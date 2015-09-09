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
    elements = 10000
    granularity = 250
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

    byteList = []
    exampleList = []
    labelList = []
    for i, dist in enumerate(f_prior):
        for label in distribute_as(dist, granularity):
            example = random.choice(data[label])
            if effect == 'noise':
                example = example + np.random.random_sample((example.shape[0],))

            example = np.minimum(1, example).astype('float32')
            exampleList.append(np.append(example,float(label)))
            labelList.append(label)
            #dtype S4 means a 4x8=(32-bit) byte string
            #when using numpy arrays, only deal with np.datatype classes
            byteArrForExample = np.append(example.astype('S4'),struct.pack('@f', float(label)))
            byteList.append(byteArrForExample)

        #f.write(bytes(np.asarray(byteList,dtype='S4').reshape(-1,1)))

    np.asarray(exampleList,dtype=np.float32).tofile('test.bin')
    print('finished writing data ...')
def retrive_data():

    print('retrieving data ...')
    filename = 'test.bin'

    #with open('test.bin', 'br') as f:
    data = np.fromfile(filename,dtype=np.float32).reshape(10000,785)
    #test2 = [i for i in range(data.shape[0]) if i%785==784]
    arr = data[:,-1]
    logging.info(list(arr))
    #data2 = data[arr]
    print('')
if __name__ == '__main__':
    logging.basicConfig(filename="labels.log", level=logging.DEBUG)
    #main()
    retrive_data()