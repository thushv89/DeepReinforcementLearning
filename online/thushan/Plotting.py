__author__ = 'Thushan Ganegedara'

import matplotlib.pyplot as plt
import csv
import numpy as np

chart_titles = ['MNIST [500] (Test Error)','MNIST [500] (Validation Error)',
                'MNIST [500,500,500] (Test Error)','MNIST [500,500,500] (Validation Error)',
                'CIFAR-10 [500] (Test Error)','CIFAR-10 [500] (Validation Error)',
                'CIFAR-10 [500,500,500] (Test Error)','CIFAR-10 [500,500,500] (Validation Error)',
                'CIFAR-100 [500] (Test Error)','CIFAR-100 [500] (Validation Error)',
                'CIFAR-100 [500,500,500] (Test Error)','CIFAR-100 [500,500,500] (Validation Error)']

all_data = []
with open('all_results.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        data_row = []
        for col in row:
            data_row.append(float(col))
        all_data.append(data_row)

x_axis = np.linspace(1,1000,1000)



plt.figure(1)

i_idx = 0
for i in [0,4*3,8*3]:
    for j in [0,3]:
        str_subplot = '23' + str(i_idx+j+1)
        plt.subplot(int(str_subplot))
        plt.plot(x_axis,all_data[i+j],'r-',
             x_axis,all_data[i+j+1],'b',
             x_axis,all_data[i+j+2],'g-')
        plt.xlabel('Position in the Dataset')
        plt.title(chart_titles[int(i/3+j/3)])
    i_idx += 1
#plt.figure(2)


plt.show()