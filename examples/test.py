import os
import csv
import matplotlib.pyplot as plt



xs = []
ys0 = []
ys1 = []
ys2 = []
ys3 = []
ys4 = []
ys5 = []

csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_cfr_result\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    xs.append(int(row['episode']))
    ys0.append(float(row['reward']))

csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_dqn_result\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    ys1.append(float(row['reward']))

csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_nfsp_result\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    ys2.append(float(row['reward']))


csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_dqn_result_1\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    ys3.append(float(row['reward']))


csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_dqn_result_2\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    ys4.append(float(row['reward']))


csv_path = r'D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_dqn_result_3\performance.csv'
csvfile=open(csv_path)
reader = csv.DictReader(csvfile)
for row in reader:
    ys5.append(float(row['reward']))




fig, ax = plt.subplots()
ax.plot(xs, ys0, label='cfr')
ax.plot(xs, ys1, label='ddqn')
ax.plot(xs, ys2, label='nfsp')
ax.plot(xs, ys3, label='dqn')
ax.plot(xs, ys4, label='actor-critic')
ax.plot(xs, ys5, label='ddpg')
ax.set(xlabel='episode', ylabel='reward')
ax.legend()
ax.grid()

save_path=r'D:\Documents\PycharmProjects\rlcard\examples\experiments_1'
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(save_path)