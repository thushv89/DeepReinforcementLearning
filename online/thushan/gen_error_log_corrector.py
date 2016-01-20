__author__ = 'Thushan Ganegedara'

import csv
all_data = []
with open('reconstruction_error_DeepRL_online_mnist-var_784, 500, 10_pool_diff.log', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        for col in row:
            if(len(col)>0):
                all_data.append(float(col.strip()))

print(all_data)