__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import json
import random
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

        #if we haven't completed 30 iterations, keep pooling
        if i <=30:
            funcs['pool'](1)
            return

        #what does this method do?
        def ma_state(name):
            retVal = 0
            if not len(data[name]) < 2:
                retVal = data[name][-1] - data[name][-2]

            return retVal
            #return 0 if len(data[name]) < 2 else data[name][-1] - data[name][-2]

        state = (data['r_15'][-1], data['neuron_balance'], ma_state('mea_5'), ma_state('mea_15'), ma_state('mea_30'))
        print('current state %f, %f, %f, %f, %f' % (data['r_15'][-1], data['neuron_balance'], ma_state('mea_5'), ma_state('mea_15'), ma_state('mea_30')))
        print('initial_size ',data['initial_size'])

        err_diff = data['valid_error_log'][-1] - data['valid_error_log'][-2]
        curr_err = data['valid_error_log'][-1]
        print('Err log 1: ',data['valid_error_log'][-1])
        print('Err log 2: ',data['valid_error_log'][-2])
        # since we have a continuous state space, we need a regression technique to get the Q-value for prev state and action
        # for a discrete state space, this can be done by using a hashtable Q(s,a) -> value
        gps = {}

        # self.q is like this there are 3 main items Action.pool, Action.reduce, Action.increment
        # each action has (s, q) value pairs
        # use this (s, q) pairs to predict the q value for a new state
        for a, value_dict in self.q.items():
            #makes sure, if to use regression element Q(s,a) should have atleast 3 samples
            if len(value_dict) < 2:
                continue

            x, y = zip(*value_dict.items())

            gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU=1, nugget=0.1)
            gp.fit(np.array(x), np.array(y))
            gps[a] = gp

        if self.prev_state or self.prev_action:

            #reward = - data['error_log'][-1]

            reward = (1 - err_diff)*(1-curr_err)
            print('Reward: ', reward)
            neuron_penalty = 0

            if data['neuron_balance'] > 2 or data['neuron_balance'] < 1:
                # the coeff was 2.0 before
                # all tests were done with 1.5
                neuron_penalty = .5 * abs(1.5 - data['neuron_balance'])
            reward -= neuron_penalty

            print('reward', reward, 'neuron_penalty', neuron_penalty)
            #sample = reward
            #sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            # len(gps) == 0 in the first time move() is called
            if len(gps) == 0:
                sample = reward
            else:
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict([self.prev_state])[0])) for gp in gps.values())

            if self.prev_state in self.q[self.prev_action]:
                self.q[self.prev_action][self.prev_state] = (1 - self.learning_rate) * self.q[self.prev_action][self.prev_state] + self.learning_rate * sample
            else:
                self.q[self.prev_action][self.prev_state] = sample

        #action = list(self.Action)[i % len(self.Action)]
        #the first time move() is called
        #Evenly chosing actions is important because with this algorithm will learn q values for all actions
        if len(gps) == 0 or i <= 60:
            action = list(self.Action)[i % len(self.Action)]
            print('evenly chose:', action)
        else:
            # determine best action by sampling the GPs
            if random.random() <= 0.1:
                action = list(self.Action)[i % len(self.Action)]
                print('explore:', action)
            else:
                action = max((np.asscalar(gp.predict(state)[0]), action) for action, gp in gps.items())[1]
                print('chose:', action)

            for a, gp in gps.items():
                print(a, np.asscalar(gp.predict(state)[0]))

        #to_move = (data['initial_size'] * 0.1) / (data['initial_size'] * data['neuron_balance'])
        #to_move = 0.25 * np.exp(-(data['neuron_balance']-1.)**2/2.) * (1. + err_diff) * (1. + curr_err)
        # newer to_move eqn
        if random.random() > 0.1:
            to_move = np.exp(-(data['neuron_balance']-1.)**2/5.) * np.abs(err_diff)
        else:
            to_move = np.exp(-(data['neuron_balance']-1.)**2/5.) * np.abs(err_diff) * (1.+curr_err)

        print('To move: ', to_move)

        if to_move<1./data['initial_size']:
            print('To move is too small')
            funcs['pool_finetune'](1)
            action = self.Action.pool
            
        else:
            if action == self.Action.pool:
                funcs['pool_finetune'](1)
            elif action == self.Action.reduce:
                # method signature: amount, to_merge, to_inc
                # introducing division by two to reduce the impact
                funcs['merge_increment_pool'](data['pool_relevant'], to_move/2., 0)
            elif action == self.Action.increment:
                funcs['merge_increment_pool'](data['pool_relevant'], 0, to_move)

        self.prev_action = action
        self.prev_state = state

    def end(self):
        return [{'name': 'q_state.json', 'json': json.dumps({str(k):{str(tup): value for tup, value in v.items()} for k,v in self.q.items()})}]