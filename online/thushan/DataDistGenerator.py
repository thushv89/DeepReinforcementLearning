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
    pickle_file = 'data' + os.sep + 'mnist.pkl'
    elements = 500000
    granularity = 100 # number of samples per distribution
    effect = ''

    np.random.seed(seed)
    random.seed(seed)

    with open(pickle_file, 'rb') as f:
        train, _, _ = pickle.load(f, encoding='latin1')

    data = defaultdict(list)
    train_x, train_y = train
    create_image_from_vector(train_x[1,:],'test2')
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

    fp = np.memmap(filename='data'+os.sep+'mnist_non_station.pkl', dtype='float32', mode='w+', shape=(elements,col_count))
    for i, dist in enumerate(f_prior):
        exampleList = []
        for label in distribute_as(dist, granularity):
            example = random.choice(data[label])
            example_before = example.copy()
            if effect == 'noise':
                example = example + np.random.random_sample((example.shape[0],))
            example = np.minimum(1, example).astype('float32')
            exampleList.append(np.append(example,float(label)))

        logging.info(list(example_before))
        #create_image_from_vector(example_before,'before_'+str(i))
        logging.info(list(example))
        #create_image_from_vector(example,'after_'+str(i))
        print('done dist in prior',i, ' out of ', len(f_prior))
        fp[i*granularity:(i+1)*granularity,:] = exampleList[:]

    del fp # includes flushing


    print('finished writing data ...')

def retrive_data():

    print('retrieving data ...')
    filename = 'data'+os.sep+'mnist_non_station.pkl'

    row_count = 1000
    #with open('test.bin', 'br') as f:
    newfp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*785*81000,shape=(row_count,785))
    data_new = np.empty((row_count,785),dtype=np.float32)
    data_new[:] = newfp[:]
    arr = data_new[:,-1]

    #data = np.load(filename).reshape(1000,785)
    #idx = np.where(data == data_new[0,0])
    #arr = data[:,-1]
    granularity = 1000
    for i in range(int(row_count/granularity)):
        logging.info(list(arr[i*granularity:(i+1)*granularity]))

    create_image_from_vector(data_new[1,:-1],'test')

def create_image_from_vector(vec,filename):
    from pylab import imshow,show,cm
    new_vec = vec*255.
    #img = Image.fromarray(np.reshape(vec*255.,(28,28)).astype(int),'L')
    #img.save(filename +'.png')
    imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    show()

if __name__ == '__main__':
    logging.basicConfig(filename="labels.log", level=logging.DEBUG)
    main()
    #retrive_data()
    print('done...')