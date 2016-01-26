__author__ = 'Thushan Ganegedara'

import matplotlib.pyplot as plt
import csv
import numpy as np
import math
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



chart_titles = ['MNIST (l1,500) (Predictive)','MNIST (l1,500) (Global Test Error)',
                'CIFAR-10 (l3,1500) (Predictive)','CIFAR-10 (l3,1500) (Global Test Error)',
                'MNIST-ROT-BACK (l3,1500) (Predictive)','MNIST-ROT-BACK (l3,1500) (Global Test Error)',
                'Neuron Adaptation (St vs Non-St)','Class Distribution (CIFAR-10)']
legends = ['SDAE','MI-DAE','RL-DAE']
all_data = []
with open('new_plots.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        data_row = []
        for i,col in enumerate(row):
            if i==0:
                continue
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

import matplotlib.gridspec as gridspec
gs1 = gridspec.GridSpec(2, 3)

def plot_fig(ax,X,Y, title, x_label, y_label, legends,colors,fontsize):
    for x,y,leg,c in zip(X,Y,legends,colors):
        ax.plot(x,y,c,label=leg)
    #ax.locator_params(nbins=3)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(loc='lower left', shadow=False, fontsize='small')

fig = plt.figure(1)
x_label = 'Position in the Dataset'
colors = ['r','b','y']
# MNIST [500] (Validation Error)
ax1 = fig.add_subplot(gs1[0])
plot_fig(ax1,[x_axis,x_axis,x_axis],[smooth_data[0]*100,smooth_data[1]*100,smooth_data[2]*100],
         chart_titles[0],x_label,'',legends[0:3],colors[0:3],'medium')

ax2 = fig.add_subplot(gs1[3])
plot_fig(ax2,[x_axis,x_axis,x_axis],[smooth_data[3]*100,smooth_data[4]*100,smooth_data[5]*100],
         chart_titles[1],x_label,'',legends[0:3],colors[0:3],'medium')

ax3 = fig.add_subplot(gs1[1])
plot_fig(ax3,[x_axis,x_axis,x_axis],[smooth_data[6]*100,smooth_data[7]*100,smooth_data[8]*100],
         chart_titles[2],x_label,'',legends[0:3],colors[0:3],'medium')

ax4 = fig.add_subplot(gs1[4])
plot_fig(ax4,[x_axis,x_axis,x_axis],[smooth_data[9]*100,smooth_data[10]*100,smooth_data[11]*100],
         chart_titles[3],x_label,'',legends[0:3],colors[0:3],'medium')

ax4 = fig.add_subplot(gs1[2])
plot_fig(ax4,[x_axis,x_axis,x_axis],[smooth_data[9]*100,smooth_data[10]*100,smooth_data[11]*100],
         chart_titles[3],x_label,'',legends[0:3],colors[0:3],'medium')
'''
str_subplot = '241'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[0]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[1]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[2]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[0])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# MNIST [500] (Global Test Error)
str_subplot = '245'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[3]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[4]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[5]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[1])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# CIFAR-10 [1000,1000,1000] (Validation Error)
str_subplot = '242'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[6]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[7]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[8]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[2])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# CIFAR-10 [1000,1000,1000] (Global Test Error)
str_subplot = '246'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[9]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[10]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[11]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[4])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# MNIST-ROT-BACK [1500,1500,1500] (Validation Error)
str_subplot = '243'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[12]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[13]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[14]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[5])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# MNIST-ROT-BACK [1500,1500,1500] (Global Test Error)
str_subplot = '247'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[15]*100,'r',label=legends[0])
plt.plot(x_axis,smooth_data[16]*100,'b',label=legends[1])
plt.plot(x_axis,smooth_data[17]*100,'y',label=legends[2])
plt.xlabel('Position in the Dataset')
plt.title(chart_titles[6])
legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)

# 'Neuron Adaptation (Stationary vs Non-Stationary)'
str_subplot = '244'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[18],'r',label='MI-DAE (St)')
plt.plot(x_axis,smooth_data[19],'b',label='RL-DAE (St)')
plt.plot(x_axis,smooth_data[20],'r',linestyle='--',label='MI-DAE (NonSt)')
plt.plot(x_axis,smooth_data[21],'b',linestyle='--',label='RL-DAE (NonSt)')
plt.xlabel('Position in the Dataset')
plt.ylabel('Neuron count in the first hidden layer')
plt.title(chart_titles[7])
legend = plt.legend(loc='upper left', shadow=False, fontsize='small')

# 'Class Distribution (CIFAR-10)'
str_subplot = '248'
plt.subplot(int(str_subplot))
plt.plot(x_axis,smooth_data[22]*100,'b',label='0')
plt.plot(x_axis,smooth_data[23]*100,'y',label='1')
plt.plot(x_axis,smooth_data[24]*100,'r',label='2')
plt.plot(x_axis,smooth_data[25]*100,'b',label='3')
plt.plot(x_axis,smooth_data[26]*100,'y',label='4')
plt.plot(x_axis,smooth_data[27]*100,'r',label='5')
plt.plot(x_axis,smooth_data[28]*100,'b',label='6')
plt.plot(x_axis,smooth_data[29]*100,'y',label='7')
plt.plot(x_axis,smooth_data[30]*100,'r',label='8')
plt.plot(x_axis,smooth_data[31]*100,'r',label='9')
plt.xlabel('Position in the Dataset')
plt.ylabel('Percentages of classes')
plt.title(chart_titles[7])
legend = plt.legend(loc='upper right', shadow=False, fontsize='small')
plt.gca().yaxis.set_major_formatter(formatter)'''
#gs1.tight_layout(fig, rect=[0, 0, 1, 1])
plt.show()