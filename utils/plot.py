import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def readc(filepath, algo):
    df = pd.read_csv(filepath)
    smoothed_mean_df = df.copy()
    smoothed_std_rewards = df.copy()

    for col in df.columns[1:]:
        smoothed_mean_df[col] = df[col].rolling(window=30).mean()
        smoothed_std_rewards[col] = df[col].rolling(window=20).std()

    # plt.figure(figsize=(10, 6))
    for col in smoothed_mean_df.columns[1:]:
        if algo == 'MADeepCFR':
            plt.fill_between(smoothed_mean_df[smoothed_mean_df.columns[0]],
                             smoothed_mean_df[col] - smoothed_std_rewards[col],
                             smoothed_mean_df[col] + smoothed_std_rewards[col], color='purple', alpha=0.3)
            sns.lineplot(x=smoothed_mean_df.columns[0], y=col, color='purple',
                         data=smoothed_mean_df, label=algo)
        else:
            plt.fill_between(smoothed_mean_df[smoothed_mean_df.columns[0]],
                             smoothed_mean_df[col] - smoothed_std_rewards[col],
                             smoothed_mean_df[col] + smoothed_std_rewards[col], alpha=0.3)
            sns.lineplot(x=smoothed_mean_df.columns[0], y=col, data=smoothed_mean_df, label=algo)


input_file = r'../experiments/multi-leduc-holdem/ac/ac.csv'
readc(input_file, 'Actor-Critic')
input_file = r'../experiments/multi-leduc-holdem/ddpg/ddpg.csv'
readc(input_file, 'DDPG')
input_file = r'../experiments/multi-leduc-holdem/dqn/dqn.csv'
readc(input_file, 'DQN')
input_file = r'../experiments/multi-leduc-holdem/nfsp/nfsp.csv'
readc(input_file, 'NFSP')
input_file = r'../experiments/multi-leduc-holdem/madeepcfr/madeeepcfr.csv'
readc(input_file, 'MADeepCFR')

input_file = r'/experiments/cfr/cfr.csv'
readc(input_file, 'CFR')
input_file = r'/experiments/multi-leduc-holdem/deepcfr/deepcfr.csv'
readc(input_file, 'DeepCFR')
input_file = r'/experiments/multi-leduc-holdem/madeepcfr/madeepcfr.csv'
readc(input_file, 'MADeepCFR')

plt.xlabel('Episode')
plt.ylabel('Reward')
# plt.title('Result')
plt.legend()
plt.grid()
plt.ylim(-0., 0.7)
plt.xlim(0.0, 20000)
plt.show()
