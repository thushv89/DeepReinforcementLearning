__author__ = 'Thushan Ganegedara'

import sys, argparse
import pickle
import struct
import random
import math
import numpy as np
import os
from collections import defaultdict
import logging

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

    train_names = ['cifar_10_data_batch_1','cifar_10_data_batch_2','cifar_10_data_batch_3','cifar_10_data_batch_4']
    valid_name = 'cifar_10_data_batch_5'
    test_name = 'cifar_10_test_batch'

    train_x = []
    train_y = []
    for file_path in train_names:
        f = open('Data' + os.sep +file_path, 'rb')
        dict = pickle.load(f,encoding='latin1')
        train_x.extend(dict.get('data')/255.)
        train_y.extend(dict.get('labels'))

    train_set = [train_x,train_y]

    f = open('Data' + os.sep +valid_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    valid_set = [dict.get('data')/255.,dict.get('labels')]

    f = open('Data' + os.sep +test_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    test_set = [dict.get('data')/255.,dict.get('labels')]

    f.close()

    all_data = [(np.asarray(train_x),np.asarray(train_y)),(valid_set[0],valid_set[1]),(test_set[0],test_set[1])]

    return all_data

def main(dataset='mnist',file_name=None, elements=100000, granularity = 20, effect = 'noise', seed = 12):

    # seed the generator (Random num generators use pseudo-random sequences)
    # seed determines the starting position in that sequence for the random numbers
    # therefore, using same seed make sure you endup with same rand sequence

    if dataset == 'mnist':
        col_count =785
        pickle_file = 'data' + os.sep + 'mnist.pkl'
        with open(pickle_file, 'rb') as f:
            train, valid, _ = pickle.load(f, encoding='latin1')
    elif dataset == 'cifar_10':
        col_count =3072 + 1
        train, valid, _ = cifar_10_load()

    np.random.seed(seed)
    random.seed(seed)

    data = defaultdict(list)

    data_x, data_y = valid

    # sort the data into bins depending on labels
    for i in range(data_x.shape[0]):
        data[data_y[i]].append(data_x[i])

    # randomly sample a GP
    def kernel(a, b):
        """ Squared exponential kernel """
        sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist)

    # number of samples
    n = math.ceil(elements / granularity)
    Xtest = np.linspace(0, 10, n).reshape(-1, 1)
    L = np.linalg.cholesky(kernel(Xtest, Xtest) + 1e-6 * np.eye(n))

    # massage the data to get a good distribution
    f_prior = np.dot(L, np.random.normal(size=(n, len(data))))
    f_prior -= f_prior.min()
    f_prior = f_prior ** math.ceil(math.sqrt(len(data)))
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)



    fp = np.memmap(filename='data'+os.sep+file_name + '.pkl', dtype='float32', mode='w+', shape=(elements,col_count))
    for i, dist in enumerate(f_prior):
        exampleList = []
        for label in distribute_as(dist, granularity):
            example = random.choice(data[label])
            example_before = example.copy()
            if effect == 'noise':
                example = example + 0.25 * np.random.random_sample((example.shape[0],))
            example = np.minimum(1, example).astype('float32')
            exampleList.append(np.append(example,float(label)))

        print('done dist in prior',i, ' out of ', len(f_prior))
        fp[i*granularity:(i+1)*granularity,:] = exampleList[:]

    del fp # includes flushing


    print('finished writing data ...')

def retrive_data(file_name, col_count,dataset):

    print('retrieving data ...')
    filename = 'data'+os.sep+file_name +'.pkl'

    row_count = 500
    #with open('test.bin', 'br') as f:
    newfp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*1000,shape=(row_count,col_count))
    data_new = np.empty((row_count,col_count),dtype=np.float32)
    data_new[:] = newfp[:]
    arr = data_new[:,-1]

    create_image_from_vector(data_new[435,:-1],dataset)

def create_image_from_vector(vec, dataset):
    from pylab import imshow,show,cm
    if dataset == 'mnist':
        imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    elif dataset == 'cifar_10':
        new_vec = 0.2989 * vec[0:1024] + 0.5870 * vec[1024:2048] + 0.1140 * vec[2048:3072]
        imshow(np.reshape(new_vec*255,(-1,32)),cmap=cm.gray)
    show()

if __name__ == '__main__':
    #logging.basicConfig(filename="labels.log", level=logging.DEBUG)
    file_name = 'cifar_10_validation_non_station_v2'
    elements = 100000
    granularity = 200
    effects = 'noise'
    seed = 543
    main('cifar_10',file_name, elements, granularity,effects,seed)
    retrive_data(file_name,3073, 'cifar_10')
    print('done...')