__author__ = 'Thushan Ganegedara'

from enum import IntEnum
import itertools
from collections import defaultdict
import numpy as np
from theano import config
import theano
import theano.tensor as T
import math
import time

class DiscreteRL(object):
    ''' Q learning model '''
    __slots__ = [ 'history', 'q', 'states', 'actions', 'prev_state', 'prev_action', 'action_log' ]

    class State(IntEnum):
        ''' State '''
        error_plus  = 1
        error_minus = 2

        def __str__(self):
            if self.value == DiscreteRL.State.error_plus:
                return '/'
            elif self.value == DiscreteRL.State.error_minus:
                return '\\'

        def __repr__(self):
            return self.__str__()

    class Action(IntEnum):
        ''' Action '''
        reduce = 1
        increment = 2
        pool = 3

        def __repr__(self):
            return str(self)

    def __init__(self):
        self.history = 3
        self.actions = list(self.Action)
        states = [ s for s in itertools.product(*([list(self.State)] * self.history)) ]
        print("")

    def gen_dist(self):
        elements = 1000
        gran = 10
        def kernel(a, b):
            """ Squared exponential kernel """
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-0.5 * sqdist)

        # number of samples
        n = math.ceil(elements / gran)
        XtestOld = np.linspace(0, 10, n)
        Xtest = XtestOld.reshape(-1, 1)
        L = np.linalg.cholesky(kernel(Xtest, Xtest) + 1e-6 * np.eye(n))

        # massage the data to get a good distribution
        f_prior = np.dot(L, np.random.normal(size=(n, 10000)))
        f_prior -= f_prior.min()
        f_prior = f_prior ** math.ceil(math.sqrt(10000))
        f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

if __name__ == '__main__':
    '''drl = DiscreteRL()
    data = defaultdict(list)
    d = data['Joe']
    x = 25 // 2
    val = np.finfo(config.floatX)'''

    arr = np.ones((150,50))
    arr2 = np.ones((100,50))*2

    
    y = T.ivector('y')
    y_mat = theano.shared(np.zeros((5,10),dtype=theano.config.floatX),borrow=True)

    given = {
        y : np.asarray([1,2,1,2,1])
    }
    y_mat_update = [(y_mat, T.inc_subtensor(y_mat[T.arange(0,5),y],1))]

    func = theano.function(inputs=[],outputs=[y_mat], updates=y_mat_update, givens=given, on_unused_input='warn')
    y_mat_other = func()

    y_new_mat = y_mat.get_value()

    batch_scores = [(i, i+5) for i in range(5,10)]
    print(np.min([s[1] for s in batch_scores]))

