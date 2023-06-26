import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def load_files(dirpath):
    df = pd.read_csv(dirpath)
    return df

def process_files(arr):
    shape = np.shape(arr)
    output_arr = []
    for n in range(shape[0]):
        output_arr.append(eval(arr[n][0]))
    return np.array(output_arr)

def main():
    filenames = ["human_emotions_step.csv","yokobo_emotions_step.csv","move_agent_rewards_step.csv"]
    dirpath = os.path.abspath(os.getcwd())
    output_dict_df = {}
    for file in filenames:
        output_dict_df[file.split(".")[0]] = load_files(os.path.join(dirpath,"results//",file)).to_numpy()
        name = file.split("_")
        if "rewards" not in name:
            output_dict_df[file.split(".")[0]] = process_files(output_dict_df[file.split(".")[0]])
    
    emotions = ["neutral","happy","sad","angry"]
    plt.plot(output_dict_df["move_agent_rewards_step"][195000:196000])

    fig, axis = plt.subplots(2,2)
    for n, ax in enumerate(axis.flatten()):
        ax.plot(output_dict_df["yokobo_emotions_step"][195000:196000,n], label="robot_"+emotions[n])
        ax.plot(output_dict_df["human_emotions_step"][195000:196000,n], label="human_"+emotions[n])
        ax.legend()
    plt.show()


    
    
    


if __name__=="__main__":
    main()