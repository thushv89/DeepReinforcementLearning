__author__ = 'Thushan Ganegedara'

from enum import IntEnum
import itertools
from collections import defaultdict
import numpy as np
from theano import config

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
        print ""
if __name__ == '__main__':
    drl = DiscreteRL()
    data = defaultdict(list)
    d = data['Joe']
    x = 25 // 2
    val = np.finfo(config.floatX)
    print ""
