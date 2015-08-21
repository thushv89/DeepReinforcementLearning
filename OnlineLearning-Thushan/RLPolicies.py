__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcess
import numpy as np

class Controller(object):

    def move(self, i, data, funcs):
        pass

    def end(self):
        return []


class ContinuousState(Controller):

    __slots__ = ['learning_rate', 'discount_rate', 'prev_state', 'prev_action', 'q', 'start_time', 'action_log']

    class Action(IntEnum):
        pool = 1
        reduce = 2
        increment = 3

        def __repr__(self):
            return str(self)

    def __init__(self, learning_rate=0.5, discount_rate=0.9, time_limit=1):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.prev_state = None
        self.prev_action = None
        self.prev_time = 0

        self.q = defaultdict(dict)
        self.time_limit = time_limit

    def move(self, i, data, funcs):

        if i <=30:
            funcs['pool'](1)
            return

        def ma_state(name):
            return 0 if len(data[name]) < 2 else data[name][-1] - data[name][-2]

        state = (data['r_15'][-1], data['neuron_balance'], ma_state('mea_5'), ma_state('mea_15'), ma_state('mea_30'))


        gps = {}

        for a, value_dict in self.q.items():
            if len(value_dict) < 2:
                continue

            x, y = zip(*value_dict.items())

            gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU==1, nugget=0.1)
            gp.fit(np.array(x), np.array(y))
            gps[a] = gp

        if self.prev_action or self.prev_action:

            reward = - data['error_log'][-1]

            neuron_penalty = 0

            if data['neuron_balance'] > 2 or data['neuron_balance'] < 1:
