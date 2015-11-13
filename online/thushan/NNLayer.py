__author__ = 'Thushan Ganegedara'

import numpy as np
import math
import theano
import theano.tensor as T

class Layer(object):

    __slots__ = ['name', 'W', 'b', 'b_prime', 'idx', 'initial_size','output_val','mask']

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
        self.output_val = None
        self.mask = None

    def output(self, x, dropout):
        x_tilda = x
        if dropout:
            #performing dropout
            srng = T.shared_randomstreams.RandomStreams(np.random.randint(235643))
            self.mask = srng.binomial(n=1, p=1-0.15, size=(x.shape[0],x.shape[1]))
            x_tilda = x * T.cast(self.mask, theano.config.floatX)
        ''' Return the output of this layer as a MLP '''
        self.output_val = T.tanh(T.dot(x_tilda, self.W) + self.b)

        return self.output_val

    def get_mask(self):
        return self.mask

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
