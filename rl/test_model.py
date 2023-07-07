from fileinput import filename
import constantes as cst
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from QLearning import *
import random
import time
from YokoboEnv import *
from datetime import datetime
from DeepQNetwork import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dimStateMotor = 1 + cst.DIM_PAD + cst.INTENTION_DIM + 9
print("dimStateMotor: " + str(dimStateMotor))
dimActionMotor = pow(len(cst.ACTIONS), cst.NUMBER_OF_MOTOR)

seed = 123
T.cuda.manual_seed_all(seed)
T.cuda.manual_seed(seed)
T.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = T.device('cuda')

path_agent_Q_net = "./Experiments/Experiment 13/models/model_Q_network_1895.7070919766927_387_bewoda.pth"
path_agent_T_net = "./Experiments/Experiment 13/models/model_T_network_1895.7070919766927_387_bewoda.pth"
path_agentlight_Q_net = "./Experiments/Experiment 13/models/model_Q_network_1895.7070919766927_387_light.pth"
path_agentlight_T_net = "./Experiments/Experiment 13/models/model_T_network_1895.7070919766927_387_bewoda.pth"


print("##########################################")
print("###                                    ###")
print("###             - BEWODA -             ###")
print("###                                    ###")
print("##########################################")

print(cst.EPSILON_MOTOR_1)

if __name__ == '__main__':
    env = YokoboEnv()
    agent = Agent(gamma=0.9, epsilon=1.0, batchSize=32, nbrActions=dimActionMotor,
                epsEnd=0.02, inputDims=dimStateMotor, lr=0.0001, epsDec=1e-2, 
                layersDim=[cst.FC1_DIM, cst.FC2_DIM, cst.FC3_DIM])
    
    model_Q_network = agent.Q_eval
    model_Q_network.load_state_dict(T.load(path_agent_Q_net))
    model_T_network = agent.T_network
    model_T_network.load_state_dict(T.load(path_agent_T_net))
    print(model_Q_network)
    print(model_T_network)
    model_Q_network.eval()
    model_T_network.eval()
    model_Q_network.to(device)
    model_T_network.to(device)

    # model_light_Q_network = agent.Q_eval.load_state_dict(T.load(path_agentlight_Q_net))
    # model_light_T_network = agent.T_network.load_state_dict(T.load(path_agentlight_T_net))

    StepRewards, StepRewardsLight = [],[]
    scores, scoresLight, epsHistory = [],[],[]
    nbrGames = 10
    number_step_to_update_T_network = 1000
    count_T_network_steps = 0
    pyplot = rtb.backends.PyPlot.PyPlot()
    rewardOverTime = []
    best_reward = 0
    best_file = ""
    best_mean_reward = -10000
    episodes_to_save = 0

    for i in range(nbrGames):
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
            observation = T.Tensor(observation).to(device)
            # print(type(observation))
            q_values = model_Q_network(observation)
            # print(q_values)
            action = T.argmax(q_values).item()
            observation_, reward, rewardLight, done, info = env.step(action, steps_num)

            score += reward
            scoreLight += rewardLight
            rewardOverTime.append(str(reward))

            # agent.storeTransition(observation, action, reward, observation_, done)

            observation = observation_
            time.sleep(cst.SAMPLING_RATE)

            # agent.update_t_target()
            # env.agentLight.update_t_target()
            action_list.append(action)

            StepRewards.append(reward)
            StepRewardsLight.append(rewardLight)

            if j>=steps_num:
                done = True

        # agent.update_epsilon()
        # env.agentLight.update_epsilon()

        scores.append(score)
        scoresLight.append(scoreLight)
        epsHistory.append(agent.epsilon)

        avgScore = np.mean(scores[-100:])
        avgScoreLight = np.mean(scoresLight[-100:])

        print("episode ", i, 'score %.2f' % score,
                'average score %.2f' % avgScore,
                "epsilon %.2f" % agent.epsilon,
                "colorMatch %.2f" % env.colorMatch,
                "step number %.2f" % j)

        info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
       
        now = datetime.now() 

        if score > best_reward:
            best_reward = score
            best_file = i

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