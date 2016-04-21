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


'''
chart_titles = ['MNIST (l1,500) (Local Error) (Non-St)','MNIST (l1,500) (Global Error) (Non-St)',
                'CIFAR-10 (l3,1000) (Local Error) (St)','CIFAR-10 (l3,1000) (Global Error) (St)',
                'MNIST-ROT-BACK (l3,1500) (Local Error)','MNIST-ROT-BACK (l3,1500) (Global Error)',
                'CIFAR-10 (l1,1000) Neuron Count (St vs Non-St)','CIFAR-10 (Class Distribution)']
legends = ['SDAE','MI-DAE','RA-DAE']
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

def plot_fig(ax,X,Y, title, x_label, y_label, legends,colors,fontsize,legend_pos='lower left'):
    for x,y,leg,c in zip(X,Y,legends,colors):
        ax.plot(x,y,c,label=leg)
    #ax.locator_params(nbins=3)
    ax.set_xlabel(x_label, fontsize='medium')
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(loc=legend_pos, shadow=False, fontsize='medium')

fig = plt.figure(1)
fig.subplots_adjust(left=0.05,right=0.95,wspace = 0.4,hspace= 0.4)

x_label = 'Position in the Dataset'
colors = ['y','b','r']
# MNIST [500] local
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

#at = AnchoredText("(a)",prop=dict(size=8), frameon=False,loc=2)

ax1 = fig.add_subplot(gs1[0])
plot_fig(ax1,[x_axis,x_axis,x_axis],[smooth_data[0]*100,smooth_data[1]*100,smooth_data[2]*100],
         chart_titles[0],x_label,'Local error %',legends[0:3],colors[0:3],'large','upper right')
#ax1.add_artist(at)
ax1.text(480,-20,'(a)',fontsize='large')

# MNIST [500] global
ax2 = fig.add_subplot(gs1[3])
plot_fig(ax2,[x_axis,x_axis,x_axis],[smooth_data[3]*100,smooth_data[4]*100,smooth_data[5]*100],
         chart_titles[1],x_label,'Global error %',legends[0:3],colors[0:3],'large','upper right')
ax2.text(480,-8,'(d)',fontsize='large')

# cifar-10 local
ax3 = fig.add_subplot(gs1[1])
plot_fig(ax3,[x_axis,x_axis,x_axis],[smooth_data[6]*100,smooth_data[7]*100,smooth_data[8]*100],
         chart_titles[2],x_label,'Local error %',legends[0:3],colors[0:3],'large','upper right')
ax3.text(480,28,'(b)',fontsize='large')

#cifar-10 global
ax4 = fig.add_subplot(gs1[4])
plot_fig(ax4,[x_axis,x_axis,x_axis],[smooth_data[9]*100,smooth_data[10]*100,smooth_data[11]*100],
         chart_titles[3],x_label,'Global error %',legends[0:3],colors[0:3],'large','upper right')
ax4.text(480,40,'(e)',fontsize='large')

#node adapt
ax5 = fig.add_subplot(gs1[2])
ax5.plot(x_axis,smooth_data[18],'b',linestyle='--',label='MI-DAE (St)')
ax5.plot(x_axis,smooth_data[19],'r',linestyle='--',label='RL-DAE (St)')
ax5.plot(x_axis,smooth_data[20],'b',linestyle='-',label='MI-DAE (Non-St)')
ax5.plot(x_axis,smooth_data[21],'r',linestyle='-',label='RL-DAE (Non-St)')
ax5.set_xlabel(x_label, fontsize='medium')
ax5.set_ylabel('Node Count in 1st Layer', fontsize='large')
ax5.set_title(chart_titles[6], fontsize='large')
ax5.legend(loc='lower right', shadow=False, fontsize='medium')
ax5.text(480,-20,'(c)',fontsize='large')

# distribution
ax6 = fig.add_subplot(gs1[5])
ax6.plot(x_axis,smooth_data[22],'b',linestyle='-',label='0')
ax6.plot(x_axis,smooth_data[23],'r',linestyle='-',label='1')
ax6.plot(x_axis,smooth_data[24],'y',linestyle='-',label='2')
ax6.plot(x_axis,smooth_data[25],'r',linestyle='--',label='3')
ax6.plot(x_axis,smooth_data[26],'b',linestyle='--',label='4')
ax6.plot(x_axis,smooth_data[27],'y',linestyle='--',label='5')
ax6.plot(x_axis,smooth_data[28],'r',linestyle=':',label='6')
ax6.plot(x_axis,smooth_data[29],'b',linestyle=':',label='7')
ax6.plot(x_axis,smooth_data[30],'y',linestyle=':',label='8')
ax6.plot(x_axis,smooth_data[31],'k',linestyle='--',label='9')

ax6.set_xlabel(x_label, fontsize='medium')
ax6.set_ylabel('Proportion of class per batch', fontsize='large')
ax6.set_title(chart_titles[7], fontsize='large')
ax6.legend(loc='upper right', shadow=False, fontsize='medium')
ax6.text(480,-0.16,'(f)',fontsize='large')
'''

chart_titles = ['State Space Comparison CIFAR-10(l3,1000) w.r.t. Global Error']
legends = [
    'State Space 1 (large)',
    'State Space 2 (large + kl_div)',
    'State Space 3 (small)',
    'State Space 4 (small + kl_div)'
           ]

x_label = 'Position in the Dataset'

state_data = {}
data_for_key = []
key = None
with open('state_comparison.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        data_row = []
        if row[0]=='Validation Error' or row[0]=='Test Error' or row[0]=='Network Size':
            if len(data_for_key)!=0:
                state_data[key]=data_for_key
            key = row[0]
            data_for_key = []
            continue
        for i,col in enumerate(row):
            if i==0:
                continue
            data_row.append(float(col)*100.)
        data_for_key.append(data_row)

state_data[key]=data_for_key

x_axis = np.linspace(1,1000,1000)
smooth_step = 10
x_axis_short = np.linspace(1,1000,int(1000/smooth_step))

smooth_state_data=[]

for i,row in enumerate(state_data['Test Error']):
    smooth_state_data.append(np.convolve(state_data['Test Error'][i], np.ones((smooth_step,))/smooth_step, mode='same'))
    for j in range(0,int(smooth_step/2)):
        smooth_state_data[i][j]=state_data['Test Error'][i][j]
    for j in range(len(state_data['Test Error'][i])-1-int(smooth_step/2),len(state_data['Test Error'][i])):
        smooth_state_data[i][j]=state_data['Test Error'][i][j]

fig = plt.figure(2)
plt.plot(x_axis,smooth_state_data[5],'b',linestyle='-',label=legends[0])
plt.plot(x_axis,smooth_state_data[6],'y',linestyle='-',label=legends[1])
plt.plot(x_axis,smooth_state_data[8],'r',linestyle='-',label=legends[2])
plt.plot(x_axis,smooth_state_data[9],'m',linestyle='-',label=legends[3])
plt.xlabel(x_label, fontsize='medium')
plt.ylabel('Global Error %', fontsize='large')
plt.title(chart_titles[0], fontsize='large')
plt.legend(loc='upper right', shadow=False, fontsize='medium')


policy_title = 'Policy (Q-Value) Comparison CIFAR-10-BIN (l3,1000)'
policy_legends = ['Q(S,$Increment$)','Q(S,$Merge$)','Q(S,$Pool$)','Class-0','Class-1']

x_label = 'Position in the Dataset'

policy_data = {}
key = None
with open('policy_comparison.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for j,row in enumerate(reader):
        data_row = []
        if row[0]=='Non Station' or row[0]=='Station':
            key = row[0]
            print(key)
            continue

        for i,col in enumerate(row):
            if i==0:
                continue
            print(j,',',i,',',col,'d')

            data_row.append(float(col))

        if key not in policy_data:
            policy_data[key] = [data_row]
        else:
            policy_data[key].append(data_row)

x_axis = np.linspace(1,139,139)

import matplotlib.gridspec as gridspec
fig = plt.figure(3)
fig.suptitle(policy_title, fontsize='large')
#plt.title(policy_title, fontsize='large')
gs3 = gridspec.GridSpec(3, 8)

ax3_1 = plt.subplot(gs3[:2, 4:])
ax3_2 = plt.subplot(gs3[2,4])
ax3_3 = plt.subplot(gs3[2, 5])
ax3_4 = plt.subplot(gs3[2,6])
ax3_5 = plt.subplot(gs3[2,7])
x3_2 = np.linspace(0,20,20)
x3_3 = np.linspace(25,35,10)
x3_4 = np.linspace(45,60,15)
x3_5 = np.linspace(110,125,15)

ax3_6 = plt.subplot(gs3[:,:4])

ax3_1.plot(x_axis,policy_data['Non Station'][0],'b',linestyle='-',label=policy_legends[0])
ax3_1.plot(x_axis,policy_data['Non Station'][1],'r',linestyle='-',label=policy_legends[1])
ax3_1.plot(x_axis,policy_data['Non Station'][2],'y',linestyle='-',label=policy_legends[2])
ax3_1.plot(x_axis,policy_data['Non Station'][3],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_1.plot(x_axis,policy_data['Non Station'][4],'r',linestyle='--',label=policy_legends[4],linewidth=2)

ax3_1.text(15, 5, '1', fontsize=20)
ax3_1.text(30, 3, '2', fontsize=20)
ax3_1.text(50, 5.5, '3', fontsize=20)
ax3_1.text(115, 3, '4', fontsize=20)

ax3_1.set_title('Non Stationary Data Distribution', fontsize='medium')
ax3_1.set_xlabel(x_label, fontsize='medium')
ax3_1.set_ylabel('Q-Values/Class distribution', fontsize='large')
ax3_1.legend(loc='upper right', shadow=False, fontsize='medium')

ax3_2.plot(x3_2,policy_data['Non Station'][3][:20],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_2.plot(x3_2,policy_data['Non Station'][4][:20],'r',linestyle='--',label=policy_legends[4],linewidth=2)
ax3_2.set_ylim([0,1])
ax3_2.set_title('Annotation 1', fontsize='medium')
ax3_2.set_xticks(np.arange(min(x3_2), max(x3_2)+1, 5.0))
ax3_2.set_ylabel('Normalized class \n distribution',fontsize='small')
ax3_3.plot(x3_3,policy_data['Non Station'][3][25:35],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_3.plot(x3_3,policy_data['Non Station'][4][25:35],'r',linestyle='--',label=policy_legends[4],linewidth=2)
ax3_3.set_ylim([0,1])
ax3_3.set_title('Annotation 2', fontsize='medium')
ax3_3.set_xticks(np.arange(min(x3_3), max(x3_3)+1, 5.0))

ax3_4.plot(x3_4,policy_data['Non Station'][3][45:60],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_4.plot(x3_4,policy_data['Non Station'][4][45:60],'r',linestyle='--',label=policy_legends[4],linewidth=2)
ax3_4.set_ylim([0,1])
ax3_4.set_title('Annotation 3', fontsize='medium')
ax3_4.set_xticks(np.arange(min(x3_4), max(x3_4)+1, 5.0))

ax3_5.plot(x3_5,policy_data['Non Station'][3][110:125],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_5.plot(x3_5,policy_data['Non Station'][4][110:125],'r',linestyle='--',label=policy_legends[4],linewidth=2)
ax3_5.set_ylim([0,1])
ax3_5.set_title('Annotation 4', fontsize='medium')
ax3_5.set_xticks(np.arange(min(x3_5), max(x3_5)+1, 5.0))

ax3_6.plot(x_axis,policy_data['Station'][0],'b',linestyle='-',label=policy_legends[0])
ax3_6.plot(x_axis,policy_data['Station'][1],'r',linestyle='-',label=policy_legends[1])
ax3_6.plot(x_axis,policy_data['Station'][2],'y',linestyle='-',label=policy_legends[2])
ax3_6.plot(x_axis,policy_data['Station'][3],'b',linestyle='--',label=policy_legends[3],linewidth=2)
ax3_6.plot(x_axis,policy_data['Station'][4],'r',linestyle='--',label=policy_legends[4],linewidth=2)

ax3_6.set_xlabel(x_label, fontsize='medium')
ax3_6.set_title('Stationary Data Distribution', fontsize='medium')
ax3_6.set_ylabel('Q-Values/Class distribution', fontsize='large')
ax3_6.legend(loc='upper left', shadow=False, fontsize='medium')

gs3.tight_layout(fig,w_pad=-2.5,h_pad=0.5,rect=[0.05,0.0,0.95,.95])


pool_title = 'Pool Size vs Global Error CIFAR-10 (l3,1000)'
pool_legends = ['MI-DAE(Pool=10000)','MI-DAE(Pool=5000)','MI-DAE(Pool=1000)','RA-DAE(Pool=10000)','RA-DAE(Pool=5000)','RA-DAE(Pool=1000)']

x_axis = np.linspace(1,999,999)
x_label = 'Position in the Dataset'

pool_data = {}
key = None
with open('pool_size.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for j,row in enumerate(reader):
        data_row = []
        if row[0]=='Validation Error' or row[0]=='Test Error':
            key = row[0]
            continue

        for i,col in enumerate(row):
            if i==0:
                continue
            data_row.append(float(col)*100)

        if key not in pool_data:
            pool_data[key] = [data_row]
        else:
            pool_data[key].append(data_row)

smooth_pool_data=[]

for i,row in enumerate(pool_data['Test Error']):
    smooth_pool_data.append(np.convolve(pool_data['Test Error'][i], np.ones((smooth_step,))/smooth_step, mode='same'))
    for j in range(0,int(smooth_step/2)):
        smooth_pool_data[i][j]=pool_data['Test Error'][i][j]
    for j in range(len(pool_data['Test Error'][i])-1-int(smooth_step/2),len(pool_data['Test Error'][i])):
        smooth_pool_data[i][j]=pool_data['Test Error'][i][j]


fig = plt.figure(4)
plt.plot(x_axis,smooth_pool_data[0],'b',linestyle='--',label=pool_legends[0],linewidth=1.5)
plt.plot(x_axis,smooth_pool_data[1],'y',linestyle='--',label=pool_legends[1],linewidth=1.5)
plt.plot(x_axis,smooth_pool_data[2],'r',linestyle='--',label=pool_legends[2],linewidth=1.5)
plt.plot(x_axis,smooth_pool_data[3],'b',linestyle='-',label=pool_legends[3])
plt.plot(x_axis,smooth_pool_data[4],'y',linestyle='-',label=pool_legends[4])
plt.plot(x_axis,smooth_pool_data[5],'r',linestyle='-',label=pool_legends[5])


plt.xlabel(x_label, fontsize='medium')
plt.ylabel('Global Error %', fontsize='large')
plt.title(pool_title, fontsize='large')
plt.legend(loc='lower left', shadow=False, fontsize='medium')

'''vis_pool_data = []
with open('state_vis_pool.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for j,row in enumerate(reader):
        data_row = []
        if j==0:
            continue
        for i,col in enumerate(row):
            data_row.append(float(col))

        vis_pool_data.append(data_row)

vis_pool = np.asarray(vis_pool_data)
fig = plt.figure(4)
ax4_1 = fig.add_subplot(231)
ax4_2 = fig.add_subplot(232)
ax4_3 = fig.add_subplot(233)
ax4_4 = fig.add_subplot(234)
ax4_5 = fig.add_subplot(235)
X, Y = np.meshgrid(vis_pool[:,0], vis_pool[:,1])
Z = []
for temp_i in range(len(X)):
    Z.append(X[i])
ax4_1.scatter(vis_pool[:,0],vis_pool[:,1])
ax4_2.scatter(vis_pool[:,0],vis_pool[:,2])
ax4_3.scatter(vis_pool[:,0],vis_pool[:,3])
ax4_4.scatter(vis_pool[:,1],vis_pool[:,2])
ax4_5.scatter(vis_pool[:,1],vis_pool[:,3])'''
#gs1.tight_layout(fig,rect=[-0.05,-0,1.05,1])
plt.show()


