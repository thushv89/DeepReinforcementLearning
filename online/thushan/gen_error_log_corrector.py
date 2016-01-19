__author__ = 'Thushan Ganegedara'

import csv
all_data = []
with open('reconstruction_error_SAE_online_cifar-10_3072, 1000, 1000, 1000, 10_pool_diff.log', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        for col in row:
            if(len(col)>0):
                all_data.append(float(col.strip()))

print(all_data)