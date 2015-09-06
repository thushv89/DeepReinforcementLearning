__author__ = 'Thushan Ganegedara'

from enum import IntEnum
import itertools
from collections import defaultdict
import numpy as np
from theano import config
import theano
import theano.tensor as T

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

    idx = T.iscalar('idx')

    y_mat_update = [(y_mat, T.inc_subtensor(y_mat[[0,1,2,3,4],[2,2,3,4,2]],1))]
        #y_mat_update = [(y,y+1) for i in y_mat.shape[0] for y in y_mat[i]]
    given = {
        y : [2,3,4,5,6]
    }

    func = theano.function(inputs=[],outputs=[], updates=y_mat_update, givens=given, on_unused_input='warn')
    #func()
    func()
    y_new_mat = y_mat.get_value()
    print('')
