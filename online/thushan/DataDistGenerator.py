__author__ = 'Thushan Ganegedara'

import sys, argparse
import pickle
import struct
import random
import math
import numpy as np
import os
from collections import defaultdict
import csv

'''-------------------------------------------------------
Takes a normal dataset and produce a non-stationary data for a given dataset
-------------------------------------------------------'''
def distribute_as(dist, n):
    cumsum = np.cumsum(dist)

    for _ in range(n):
        x = random.random()
        found = False
        for i, y in enumerate(cumsum):
            if x < y:
                yield i
                found = True
                break

        if not found:
            yield dist.shape[0] - 1

def cifar_10_load():

    train_names = ['cifar_10_data_batch_1','cifar_10_data_batch_2','cifar_10_data_batch_3','cifar_10_data_batch_4','cifar_10_data_batch_5']
    valid_name = 'cifar_10_data_batch_5'
    test_name = 'cifar_10_test_batch'

    train_x = []
    train_y = []
    for file_path in train_names:
        f = open('data' + os.sep +file_path, 'rb')
        dict = pickle.load(f,encoding='latin1')
        train_x.extend(dict.get('data')/255.)
        train_y.extend(dict.get('labels'))

    train_set = [train_x,train_y]

    f = open('data' + os.sep +valid_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    valid_set = [dict.get('data')/255.,dict.get('labels')]

    f = open('data' + os.sep +test_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    test_set = [dict.get('data')/255.,dict.get('labels')]

    f.close()

    all_data = [(np.asarray(train_x),np.asarray(train_y)),(valid_set[0],valid_set[1]),(test_set[0],test_set[1])]

    return all_data

def cifar_100_load():

    train_names = ['cifar_100_train']
    valid_name = 'cifar_100_train'
    test_name = 'cifar_100_test_batch'

    train_x = []
    train_y = []
    for file_path in train_names:
        f = open('data' + os.sep +file_path, 'rb')
        dict = pickle.load(f,encoding='latin1')
        train_x.extend(dict.get('data')/255.)
        train_y.extend(dict.get('fine_labels'))

    train_set = [train_x,train_y]

    f = open('data' + os.sep +valid_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    valid_set = [dict.get('data')/255.,dict.get('fine_labels')]

    f = open('data' + os.sep +test_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    test_set = [dict.get('data')/255.,dict.get('fine_labels')]

    f.close()

    all_data = [(np.asarray(train_x),np.asarray(train_y)),(valid_set[0],valid_set[1]),(test_set[0],test_set[1])]

    return all_data

def svhn_load():

    import scipy.io as sio

    train_name = 'svhn_train_32x32.mat'
    valid_name = 'svhn_train_32x32.mat'
    test_name = 'svhn_test_32x32.mat'

    data = sio.loadmat('data' + os.sep +train_name)
    train_x = data['X']
    # replacing label 10 with 0
    train_y = [ele[0] if ele<10 else 0 for ele in data['y']]

    res_train_x  = np.swapaxes(train_x,0,1).T.reshape((-1,3072),order='C')/255.
    #create_image_from_vector(res_train_x[534,:],'svhn','rgb')
    train_set = [res_train_x,train_y]

    vdata = sio.loadmat('data' + os.sep +valid_name)
    valid_x = vdata['X']
    valid_y = [ele[0] for ele in vdata['y']]

    res_valid_x  = np.swapaxes(valid_x,0,1).T.reshape((-1,3072),order='C')/255.
    #create_image_from_vector(res_train_x[534,:],'svhn','rgb')
    valid_set = [res_valid_x,valid_y]

    testdata = sio.loadmat('data' + os.sep +test_name)
    test_x = testdata['X']
    test_y = [ele[0] for ele in testdata['y']]

    res_test_x  = np.swapaxes(test_x,0,1).T.reshape((-1,3072),order='C')/255.
    #create_image_from_vector(res_train_x[534,:],'svhn','rgb')
    test_set = [res_test_x,test_y]

    all_data = [(train_set[0],train_set[1]),(valid_set[0],valid_set[1]),(test_set[0],test_set[1])]

    return all_data

def main(dataset='mnist', col_count=785,file_name=None, elements=100000, granularity = 20, effect = 'noise', seed = 12, mode='gauss',chunk_size=1000):

    # seed the generator (Random num generators use pseudo-random sequences)
    # seed determines the starting position in that sequence for the random numbers
    # therefore, using same seed make sure you endup with same rand sequence

    if dataset == 'mnist':
        pickle_file = 'data' + os.sep + 'mnist.pkl'
        with open(pickle_file, 'rb') as f:
            train, valid, _ = pickle.load(f, encoding='latin1')
    elif dataset == 'cifar_10':
        train, valid, _ = cifar_10_load()
    elif dataset == 'cifar_100':
        train, valid, _ = cifar_100_load()

    np.random.seed(seed)
    random.seed(seed)

    data = defaultdict(list)

    data_x, data_y = train

    # sort the data into bins depending on labels
    for i in range(data_x.shape[0]):
        data[data_y[i]].append(data_x[i])

    print(list(data.keys()))
    # randomly sample a GP
    def kernel(a, b):
        """ Squared exponential kernel """
        sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist)

    # number of samples
    n = math.ceil(elements / granularity)
    if mode=='gauss':
        Xtest = np.linspace(0, col_count, n).reshape(-1, 1)
        L = np.linalg.cholesky(kernel(Xtest, Xtest) + 1e-6 * np.eye(n))

        # massage the data to get a good distribution
        f_prior = np.dot(L, np.random.normal(size=(n, len(data))))
        f_prior -= f_prior.min()
        f_prior = f_prior ** math.ceil(math.sqrt(len(data)))
        f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)
    elif mode=='uni':
        f_prior = []
        for i in range(n):
            f_prior.append([1./len(data) for _ in range(len(data))])
    elif mode == 'gauss_bin':
        Xtest = np.linspace(0, col_count, n).reshape(-1, 1)
        L = np.linalg.cholesky(kernel(Xtest, Xtest) + 1e-6 * np.eye(n))
        f_prior = np.dot(L, np.random.normal(size=(n, 2)))
        f_prior -= f_prior.min()
        f_prior = f_prior ** math.ceil(math.sqrt(2))
        f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)
        #f_prior = np.append(f_prior,np.zeros(n,len(data)-2),axis=1)
        label_count = len(data)
        for k_i in range(2,label_count):
            data.pop(k_i)
    elif mode == 'uni_bin':
        label_count = len(data)
        for k_i in range(2,label_count):
            data.pop(k_i)
        f_prior = []
        for i in range(n):
            f_prior.append([1./len(data) for _ in range(len(data))])
    else:
        raise NotImplementedError

    fp = np.memmap(filename='data'+os.sep+file_name + '.pkl', dtype='float32', mode='w+', shape=(elements,col_count))
    # f_prior is 'n' elements long. each of that has 'col_count' elements
    for i, dist in enumerate(f_prior):
        exampleList = []
        label_dist = distribute_as(dist, granularity)
        for label in label_dist:
            example = random.choice(data[label])
            if effect == 'noise':
                example = example + 0.25 * np.random.random_sample((example.shape[0],))
            example = np.minimum(1, example).astype('float32')
            exampleList.append(np.append(example,float(label)))

        print('done dist in prior',i, ' out of ', len(f_prior))

        print('Breaking to small chunks for writing')
        for j in range(0,int(granularity/chunk_size)):
            pos_1 = int((i*granularity)+(j*chunk_size))
            pos_2 = int((i*granularity)+((j+1)*chunk_size))
            #fp[i*granularity:(i+1)*granularity,:] = exampleList[:]
            fp[pos_1:pos_2,:] = exampleList[int(j*chunk_size):int((j+1)*chunk_size)]
            print('\t\tChunk ',j,' ...')
            print('\t\tWriting to: ',pos_1,':',pos_2)
            print('\t\tReading from: ',j*chunk_size,':',(j+1)*chunk_size)

    del fp # includes flushing


    print('finished writing data ...')

def write_data_distribution(filename, col_count, row_count, row_total, label_count, mode):
    if mode == 'gauss_bin' or mode=='uni_bin':
        label_count = 2

    print('retrieving data ...')
    filename = 'data'+os.sep+filename +'.pkl'
    distribution = []
    from collections import Counter

    for i in range(row_total//row_count):
        print('dist retrieved for ', i)
        newfp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*i*row_count,shape=(row_count,col_count))
        data_new = np.empty((row_count,col_count),dtype=np.float32)
        data_new[:] = newfp[:]
        arr = data_new[:,-1]
        dist = Counter(arr)
        #for k,v in dist.items():
        #    distribution[str(k)] = distribution[str(k)] + v / sum(dist.values()) \
        #        if str(k) in distribution else v / sum(dist.values())
        distribution.append({str(k): v / sum(dist.values()) for k, v in dist.items()})
    ordered_dist = []
    for val in range(label_count):
        k = str(float(val))
        ordered_dist_i = []
        for dist_i in distribution:
            ordered_dist_i.append(dist_i[str(k)] if str(k) in dist_i else 0)
        ordered_dist.append(ordered_dist_i)

    with open('label_dist_'+file_name+'.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for i in ordered_dist:
            writer.writerow([val for val in i])

def retrive_data(file_name, col_count,dataset):

    print('retrieving data ...')
    filename = 'data'+os.sep+file_name +'.pkl'

    row_count = 1000
    #with open('test.bin', 'br') as f:
    newfp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*120000,shape=(row_count,col_count))
    data_new = np.empty((row_count,col_count),dtype=np.float32)
    data_new[:] = newfp[:]
    arr = data_new[:,-1]

    create_image_from_vector(data_new[980,:-1],dataset)

def create_image_from_vector(vec, dataset):
    if dataset == 'mnist':
        from pylab import imshow,show,cm
        imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
        show()
    elif dataset == 'cifar_10' or dataset=='cifar_100':
        import matplotlib.pyplot as plt
        new_vec = 0.2989 * vec[0:1024] + 0.5870 * vec[1024:2048] + 0.1140 * vec[2048:3072]
        rgb_vec = [np.reshape(vec[0:1024],(-1,32)),np.reshape(vec[1024:2048],(-1,32)),np.reshape(vec[2048:3072],(-1,32))]
        plt.imshow(np.transpose(np.asarray(rgb_vec),axes=(1,2,0)))
        plt.axis('off')
        plt.show()
if __name__ == '__main__':
    #logging.basicConfig(filename="labels.log", level=logging.DEBUG)

    seed = 12

    elements = 200000 # number of datapoints in final dataset
    granularity = 10000 #granularity changes how fast data changes
    suffix = 'station' #suffix used for the file name of produced data
    dataset = 'cifar_10' #dataset to use
    mode = 'uni_bin' #distribution type: gauss (gaussian distribution for multi-class data) ,uni (uniform distribution for multi-class data,
                    # gauss_bin (gaussian distribution for binary-class data),uni_bin (uniform distribution for binary-class data)

    effects = 'noise' #any special effects to have (e.g. noisy images)

    # parameters to define amount of data written to disk at a time
    row_count=1000
    chunk_size = 1000

    file_name = dataset + '_'+ suffix + '_'+str(elements) + '_' + mode

    if dataset == 'mnist':
        col_count = 784+1
        label_count = 10
    elif dataset == 'cifar_10':
        col_count = 3072+1
        label_count = 10
    elif dataset == 'cifar_100':
        col_count = 3072+1
        label_count = 100


    main(dataset, col_count,file_name, elements, granularity,effects,seed,mode,chunk_size)
    #retrive_data(file_name,col_count, dataset)

    write_data_distribution(file_name,col_count,row_count,elements, label_count,mode)
    print('done...')