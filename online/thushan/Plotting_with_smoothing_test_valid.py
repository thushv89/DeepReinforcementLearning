__author__ = 'Thushan Ganegedara'

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


chart_titles = ['MNIST [500] (Validation Error)','MNIST [500] (Test Error)',
                'MNIST [500,500,500] (Validation Error)','MNIST [500,500,500] (Test Error)',
                'CIFAR-10 [1000,1000,1000] (Validation Error)','CIFAR-10 [1000,1000,1000] (Test Error)',
                'CIFAR-10 [1000,1000,1000,1000,1000] (Validation Error)','CIFAR-10 [1000,1000,1000,1000,1000] (Test Error)',
                'MNIST-ROT-BACK [500] (Validation Error)','MNIST-ROT-BACK [500] (Test Error)',
                'MNIST-ROT-BACK [500,500,500] (Validation Error)','MNIST-ROT-BACK [500,500,500] (Test Error)']
legends = ['SDAE','MI-DAE','RL-DAE']
all_data = []
with open('test_valid_results_new.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        data_row = []
        for col in row:
            data_row.append(float(col))
        all_data.append(data_row)

x_axis = np.linspace(1,1000,1000)
smooth_step = 10
x_axis_short = np.linspace(1,1000,int(1000/smooth_step))

smooth_data=[]

for i,row in enumerate(all_data):
    smooth_data.append(np.convolve(all_data[i], np.ones((smooth_step,))/smooth_step, mode='same'))
    for j in range(0,int(smooth_step/2)):
        smooth_data[i][j]=all_data[i][j]
    for j in range(len(all_data[i])-1-int(smooth_step/2),len(all_data[i])):
        smooth_data[i][j]=all_data[i][j]

formatter = FuncFormatter(to_percent)

plt.figure(1)
i_idx = 0
for i in [0*6,2*6,4*6]:
    for j in [0,3]:
        str_subplot = '23' + str(i_idx+j+1)
        plt.subplot(int(str_subplot))

        plt.plot(x_axis,smooth_data[i+j]*100,linestyle='-',color='r',label=legends[0])
        plt.plot(x_axis,smooth_data[i+j+1]*100,linestyle='-',color='b',label=legends[1])
        plt.plot(x_axis,smooth_data[i+j+2]*100,linestyle='-',color='g',label=legends[2])
        plt.xlabel('Position in the Dataset')

        if str_subplot=='231' or str_subplot=='232' or str_subplot=='233':
            plt.ylabel('Validation Error %')
        else:
            plt.ylabel('Test Error %')

        plt.title(chart_titles[int(i/3+j/3)])
        if str_subplot=='231':
            legend = plt.legend(loc='upper left', shadow=False, fontsize='small')
        elif str_subplot=='232':
            legend = plt.legend(loc='lower right', shadow=False, fontsize='small')
        else:
            legend = plt.legend(loc='lower left', shadow=False, fontsize='small')

        plt.gca().yaxis.set_major_formatter(formatter)

    i_idx += 1

'''axes = plt.subplot(233)
axes.set_ylim([0.5,1.0])
axes2 = plt.subplot(236)
axes2.set_ylim([0.0,1.0])'''

plt.figure(2)
i_idx = 0
for i in [1*6,3*6,5*6]:
    for j in [0,3]:
        str_subplot = '23' + str(i_idx+j+1)
        plt.subplot(int(str_subplot))

        plt.plot(x_axis,smooth_data[i+j]*100,linestyle='-',color='r',label=legends[0])
        plt.plot(x_axis,smooth_data[i+j+1]*100,linestyle='-',color='b',label=legends[1])
        plt.plot(x_axis,smooth_data[i+j+2]*100,linestyle='-',color='g',label=legends[2])

        plt.xlabel('Position in the Dataset')
        if str_subplot=='231' or str_subplot=='232' or str_subplot=='233':
            plt.ylabel('Validation Error %')
        else:
            plt.ylabel('Test Error %')

        plt.title(chart_titles[int(math.ceil(i/3)+j/3)])

        if str_subplot=='231':
            legend = plt.legend(loc='upper left', shadow=False, fontsize='small')
        elif str_subplot=='232':
            legend = plt.legend(loc='lower right', shadow=False, fontsize='small')
        else:
            legend = plt.legend(loc='lower left', shadow=False, fontsize='small')

        plt.gca().yaxis.set_major_formatter(formatter)
    i_idx += 1

'''axes = plt.subplot(233)
axes.set_ylim([0.5,1.0])
axes2 = plt.subplot(236)
axes2.set_ylim([0.0,1.0])'''

plt.show()