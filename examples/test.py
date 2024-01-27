import os
import csv
import matplotlib.pyplot as plt



xs = []
ys0 = []
ys1 = []
ys2 = []


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


fig, ax = plt.subplots()
ax.plot(xs, ys0, label='cfr')
ax.plot(xs, ys1, label='dqn')
ax.plot(xs, ys2, label='nfsp')
ax.set(xlabel='episode', ylabel='reward')
ax.legend()
ax.grid()

save_path=r'D:\Documents\PycharmProjects\rlcard\examples\experiments'
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(save_path)