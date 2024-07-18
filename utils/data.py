import csv
import json

csv_path = r"D:\Documents\PycharmProjects\rlcard\csv\dqn.csv"
json_path = r"D:\Downloads\multi-leduc-holdem_dqn_run5.json"

with open(json_path, 'r') as fcc_file:
    fcc_data = json.load(fcc_file)
    print(fcc_data)
    csv_file = open(csv_path, 'w')
    fieldnames = ['episode', 'reward']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for a in fcc_data:
        episode = int(a[1])
        reward = a[2]
        writer.writerow({'episode': int(episode), 'reward': reward})
    csv_file.close()
