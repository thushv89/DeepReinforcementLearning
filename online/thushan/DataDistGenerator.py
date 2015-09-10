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

def main():

    # seed the generator (Random num generators use pseudo-random sequences)
    # seed determines the starting position in that sequence for the random numbers
    # therefore, using same seed make sure you endup with same rand sequence
    seed = 12
    pickle_file = 'Data' + os.sep + 'mnist.pkl'
    elements = 500000
    granularity = 1000 # number of samples per distribution
    effect = 'noise'

    np.random.seed(seed)
    random.seed(seed)

    with open(pickle_file, 'rb') as f:
        train, _, _ = pickle.load(f, encoding='latin1')

    data = defaultdict(list)
    train_x, train_y = train

    # sort the data into bins depending on labels
    for i in range(train_x.shape[0]):
        data[train_y[i]].append(train_x[i])

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

    col_count =785
    for i, dist in enumerate(f_prior):
        exampleList = []
        for label in distribute_as(dist, granularity):
            example = random.choice(data[label])
            if effect == 'noise':
                example = example + np.random.random_sample((example.shape[0],))

            example = np.minimum(1, example).astype('float32')
            exampleList.append(np.append(example,float(label)))

        print('done dist in prior',i, ' out of ', len(f_prior))
        #f.write(bytes(np.asarray(byteList,dtype='S4').reshape(-1,1)))
        fp = np.memmap(filename='mnist_non_station.pkl', dtype='float32', mode='w+',
                       offset=np.dtype('float32').itemsize*len(exampleList)*col_count*i,
                       shape=(len(exampleList),col_count))
        fp[:] = exampleList[:]
        fp.flush()
        del fp


    print('finished writing data ...')

def retrive_data():

    print('retrieving data ...')
    filename = 'mnist_non_station.pkl'

    row_count = 100
    #with open('test.bin', 'br') as f:
    fp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*785*10,shape=(row_count,785))
    data_new = np.empty((row_count,785),dtype=np.float32)
    data_new[:] = fp[:]
    arr = data_new[:,-1]

    #data = np.load(filename).reshape(1000,785)
    #idx = np.where(data == data_new[0,0])
    #arr = data[:,-1]
    #logging.info(list(arr))
    #data2 = data[arr]
    print('')

if __name__ == '__main__':
    logging.basicConfig(filename="labels.log", level=logging.DEBUG)
    main()
    #retrive_data()
    print('done...')