__author__ = 'Thushan Ganegedara'

import numpy as np
import math
import theano
import theano.tensor as T

class Layer(object):

    __slots__ = ['name', 'W', 'b', 'b_prime', 'idx', 'initial_size']

    def __init__(self, input_size, output_size, zero=None, W=None, b=None, b_prime=None):
        self.name = '(%d*->%d*)' %(input_size,output_size) # %d represent input_size
        self.idx = T.ivector('idx_' + self.name)

        if zero:
            self.W = theano.shared(np.zeros((input_size, output_size), dtype=theano.config.floatX))
        elif W!=None:
            self.W = theano.shared(W)
        else:
            rng = np.random.RandomState(0)
            init = 4 * np.sqrt(6.0 / (input_size + output_size))
            initial = np.asarray(rng.uniform(low=-init, high=init, size=(input_size, output_size)), dtype=theano.config.floatX)

            # randomly initalise weights
            self.W = theano.shared(initial, 'W_' + self.name)

        self.b = theano.shared(b if b != None else np.zeros(output_size, dtype=theano.config.floatX), 'b_' + self.name)
        self.b_prime = theano.shared(b_prime if b_prime != None else np.zeros(input_size, dtype=theano.config.floatX), 'b\'_' + self.name)
        self.initial_size = (input_size, output_size)

    def output(self, x):
        ''' Return the output of this layer as a MLP '''
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
