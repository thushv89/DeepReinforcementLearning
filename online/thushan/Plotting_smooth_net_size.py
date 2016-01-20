__author__ = 'Thushan Ganegedara'

import matplotlib.pyplot as plt
import csv
import numpy as np
import math

chart_titles = ['CIFAR-10 [1000,1000,1000] (Layer 1)']

legends = ['MIncDAE','DeepRLNet']

all_data = []
with open('net_size_results.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        data_row = []
        for col in row:
            data_row.append(float(col))
        all_data.append(data_row)

x_axis = np.linspace(1,999,999)

plt.figure(1)

plt.plot(x_axis,all_data[0],linestyle='--',color='r',label=legends[0])
plt.plot(x_axis,all_data[1],linestyle='-',color='b',label=legends[1])

plt.xlabel('Position in the Dataset')
plt.ylabel('Number of Nodes')
plt.title(chart_titles[0])
legend = plt.legend(loc='upper left', shadow=False, fontsize='small')

plt.show()