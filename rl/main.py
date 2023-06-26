from fileinput import filename
import constantes as cst
import nep 
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from QLearning import *
from DeepQNetwork import Agent
import gym
import random
import time
from YokoboEnv import *
from datetime import datetime
import sys
from torch.utils.tensorboard import SummaryWriter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimStateMotor = 1 + cst.DIM_PAD + cst.INTENTION_DIM + 9 # 1 for the emotion, for the humidity IN/OUT(2), temperature IN/OUT(2), co2 (1) and atm (1), position motor (3)
print("dimStateMotor: " + str(dimStateMotor))
dimActionMotor = pow(len(cst.ACTIONS), cst.NUMBER_OF_MOTOR)

#rl_motor = QLearning(dimStateMotor, dimActionMotor)

seed = 123 # int(time.time())
#T.use_deterministic_algorithms(True)
#T.backends.cudnn.deterministic = True
#T.backends.cudnn.benchmark = False
T.cuda.manual_seed_all(seed)
T.cuda.manual_seed(seed)
T.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("##########################################")
print("###                                    ###")
print("###             - BEWODA -             ###")
print("###                                    ###")
print("##########################################")

print(cst.EPSILON_MOTOR_1)
# sys.exit()

if __name__ == '__main__':
    writer = SummaryWriter(comment="-" + "BEWODA" + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    env = YokoboEnv()
    agent = Agent(gamma=0.9, epsilon=1.0, batchSize=32, nbrActions=dimActionMotor,
                epsEnd=0.02, inputDims=dimStateMotor, lr=0.0001, epsDec=1e-2, layersDim=[cst.FC1_DIM, cst.FC2_DIM, cst.FC3_DIM])
    StepRewards, StepRewardsLight = [],[]
    scores, scoresLight, epsHistory = [],[],[]
    nbrGames = 500 + 1
    number_step_to_update_T_network = 1000
    count_T_network_steps = 0
    pyplot = rtb.backends.PyPlot.PyPlot()
    rewardOverTime = []
    best_reward = 0
    best_file = ""
    best_mean_reward = 0
    episodes_to_save = 0
    for i in range(nbrGames):

        if episodes_to_save > 20:
            avgScore = np.mean(scores[-100:]) if scores else 0
            best_mean_reward = score
            agent.save_models(reward, i, tag="bewoda")
            env.agentLight.save_models(reward, i, tag="light")

            info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
            env.saveTrajectory(i, thres=70, info=info)
            now = datetime.now() # current date and time
            with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
                fp.write(';'.join(rewardOverTime))

            break

        score = 0
        scoreLight = 0
        done = False
        observation = env.reset()
        j=0
        action_list = []

        # Number of steps per interaction
        steps_num = random.randint(300,500)

        while not done:
            j+=1
            count_T_network_steps += 1
            action = agent.chooseAction(observation)
            observation_, reward, rewardLight, done, info = env.step(action, steps_num)

            score += reward
            scoreLight += rewardLight
            rewardOverTime.append(str(reward))

            agent.storeTransition(observation, action, reward, observation_, done)

            observation = observation_
            #env.render()
            time.sleep(cst.SAMPLING_RATE)

            # remove if condition
            # if agent.memCounter >= agent.memSize:
            agent.learn()

            # if count_T_network_steps % number_step_to_update_T_network == 0:
            agent.update_t_target()
            env.agentLight.update_t_target()
            action_list.append(action)

            StepRewards.append(reward)
            StepRewardsLight.append(rewardLight)

            if j>=steps_num:
                done = True

            writer.add_scalar("epsilon", agent.epsilon, count_T_network_steps)
            writer.add_scalar("reward", reward, count_T_network_steps)
            writer.add_scalar("rewardLight", rewardLight, count_T_network_steps)

        # remove if condition for memory
        # if agent.memCounter >= agent.memSize:
        agent.update_epsilon()
        env.agentLight.update_epsilon()

        # if j > 100:
        #     episodes_to_save += 1
        # else:
        #     episodes_to_save = 0

        # if (score > best_mean_reward):
        #     best_mean_reward = score
        #     agent.save_models(score, i, tag="bewoda")
        #     env.agentLight.save_models(score, i, tag="light")

        #     # Cumulative reward
        #     avgScore = np.mean(scores[-100:]) if scores else 0
        #     info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
        #     env.saveTrajectory(i, thres=70, info=info)
        #     now = datetime.now() # current date and time
        #     with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
        #         fp.write(';'.join(rewardOverTime))
                
        # cst.ROBOT.plot(        
        #     np.transpose(np.array(env.yokobo.trajectory()[0]), (1,0)),
        #     backend='pyplot',
        #     dt=0.001,
        #     block=True,
        #     # color=color,
        #     # printEach=True
        #     )

        scores.append(score)
        scoresLight.append(scoreLight)
        epsHistory.append(agent.epsilon)

        avgScore = np.mean(scores[-100:])
        avgScoreLight = np.mean(scoresLight[-100:])

        if (avgScore > best_mean_reward):
            best_mean_reward = avgScore
            agent.save_models(score, i, tag="bewoda")
            env.agentLight.save_models(score, i, tag="light")

            # Cumulative reward
            avgScore = np.mean(scores[-100:]) if scores else 0
            info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
            env.saveTrajectory(i, thres=70, info=info)
            now = datetime.now() # current date and time
            with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
                fp.write(';'.join(rewardOverTime))

        writer.add_scalar("reward_100", avgScore, i)
        writer.add_scalar("rewardLight_100", avgScoreLight, i)
        # avgScore = np.mean(scores)
        print("episode ", i, 'score %.2f' % score,
                'average score %.2f' % avgScore,
                "epsilon %.2f" % agent.epsilon,
                "colorMatch %.2f" % env.colorMatch,
                "step number %.2f" % j)

        info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
        # if score > 0:
        #     env.saveTrajectory(i, thres=70, info=info)
        #     with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
        #         fp.write(';'.join(rewardOverTime))
        # env.saveTrajectory(i, thres=70, info=info)

        # if i%(100)==0:
        # env.plot_emotions()
        # env.plot_sensor_values()
        # plt.show()

        # plt.hist(action_list, density=True, bins=27)
        # plt.show()

        now = datetime.now() 

        if score > best_reward:
            best_reward = score
            best_file = i

        # self.file = open("./data/motors-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + '(' + str(lengthTraj) + "_pts)" + "_" + str(episode) + noColor + noPAD + ".traj", "a")
        # with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
        #     fp.write(';'.join(rewardOverTime))

    # env.saveTrajectory(i, thres=70, info=info)
    # plt.plot(scores)
    # plt.show()
    # with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
    #         fp.write(';'.join(rewardOverTime))
    # pyplot.hold()

    print(f"Best episode{best_file}, with reward {best_reward}")

    human_emotions_step = {'Human Emotion Distribution': env.human_distributions_step}
    yokobo_emotions_step = {'Yokobo Emotion Distribution': env.yokobo_distributions_step}
    human_emotions_episode = {'Human Emotion Distribution': env.human_distributions_episode}
    yokobo_emotions_episode = {'Yokobo Emotion Distribution': env.yokobo_distributions_episode}
    move_agent_rewards_step = {'Movement Agent Rewards': StepRewards}
    light_agent_rewards_step = {'Movement Agent Rewards': StepRewardsLight}
    move_agent_rewards_episode = {'Movement Agent Rewards': scores}
    light_agent_rewards_episode = {'Light Agent Rewards': scoresLight}
    TotalEmotions = {'Human Emotions': env.human_emotions_total}

    human_emotions_step_df = pd.DataFrame.from_dict(human_emotions_step)
    yokobo_emotions_step_df = pd.DataFrame.from_dict(yokobo_emotions_step)
    human_emotions_episode_df = pd.DataFrame.from_dict(human_emotions_episode)
    yokobo_emotions_episode_df = pd.DataFrame.from_dict(yokobo_emotions_episode)
    move_agent_rewards_step_df = pd.DataFrame.from_dict(move_agent_rewards_step)
    light_agent_rewards_step_df = pd.DataFrame.from_dict(light_agent_rewards_step)
    move_agent_rewards_episode_df = pd.DataFrame.from_dict(move_agent_rewards_episode)
    light_agent_rewards_episode_df = pd.DataFrame.from_dict(light_agent_rewards_episode)
    TotalEmotions_df = pd.DataFrame.from_dict(TotalEmotions)

    human_emotions_step_df.to_csv("./results/human_emotions_step.csv", index=False)
    yokobo_emotions_step_df.to_csv("./results/yokobo_emotions_step.csv", index=False)
    human_emotions_episode_df.to_csv("./results/human_emotions_episode.csv", index=False)
    yokobo_emotions_episode_df.to_csv("./results/yokobo_emotions_episode.csv", index=False)
    move_agent_rewards_step_df.to_csv("./results/move_agent_rewards_step.csv", index=False)
    light_agent_rewards_step_df.to_csv("./results/light_agent_rewards_step.csv", index=False)
    move_agent_rewards_episode_df.to_csv("./results/move_agent_rewards_episode.csv", index=False)
    light_agent_rewards_episode_df.to_csv("./results/light_agent_rewards_episode.csv", index=False)
    TotalEmotions_df.to_csv("./results/total_emotions.csv", index=False)