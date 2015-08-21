''' Reinforcement learning module '''

import random
import itertools

import time
import json

from enum import IntEnum
from collections import defaultdict

import numpy as np
from sklearn.gaussian_process import GaussianProcess

class Controller(object):
    ''' Neural network architecture change controller '''
    def move(self, i, data, funcs):
        ''' Perform the next adjustment '''
        pass

    def end(self):
        ''' Returns the log '''
        return []

class DiscreteRL(Controller):
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

    def __init__(self, learning_rate=0.5, discount_rate=0.9):
        self.history = 3

        # initalise Q value map
        self.q = defaultdict(float)

        self.states = [ s for s in itertools.product(*([list(self.State)] * self.history)) ]
        self.actions = list(self.Action)

        self.prev_state = None
        self.prev_action = None
        self.action_log = []

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def move(self, i, data, funcs):
        ''' Determine the action given the state '''
        if i <= 30:
            funcs['pool'](1)
            self.action_log.append(str(self.Action.pool))
            return

        state = ( self.State.error_plus if data['mea_30'][-2] < data['mea_30'][-1] else self.State.error_minus,
                  self.State.error_plus if data['mea_15'][-2] < data['mea_15'][-1] else self.State.error_minus,
                  self.State.error_plus if data['mea_5'][-2]  < data['mea_5'][-1]  else self.State.error_minus )
        print('state', state)

        # check the reward
        if self.prev_state:
            # process the s, a, s', r tuple
            reward = - data['errors']

            neuron_penalty = 0
            if data['neuron_balance'] > 2 or data['neuron_balance'] < 1:
                neuron_penalty = 1.5 * abs(1 - data['neuron_balance'])

            reward -= neuron_penalty
            sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            print('reward', reward, 'sample', sample, self.learning_rate)
            self.q[self.prev_state, self.prev_action] = (1 - self.learning_rate) * self.q[self.prev_state, self.prev_action] +  self.learning_rate * sample

        # determine action
        if i <= 45 or random.random() < 0.1:
            action = list(self.Action)[i % len(self.Action)]
            print('evenly chose', action)
        else:
            # pick most optimal policy
            action = max((self.q[state, a], a) for a in self.actions)[1]
            print('chose', action)

            # print our thought process
            for s in self.states:
                value = max((self.q[s, a], a) for a in self.actions)
                print(s, ':', value)

        # do the action
        if action == self.Action.reduce:
            funcs['merge_increment_pool'](data['pool_relevant'], 0.1, 0)
        elif action == self.Action.increment:
            funcs['merge_increment_pool'](data['pool_relevant'], 0, 0.1)
        else:
            funcs['pool'](1)

        self.prev_action = action
        self.prev_state = state

        self.action_log.append(str(action))

    def end(self):
        return [ { 'name': 'actions.json', 'json': json.dumps(self.action_log) } ]

class Pooler(Controller):
    ''' Pool 100 all the way '''
    def move(self, i, data, funcs):
        funcs['pool'](1)

class SelectivePooler(Controller):
    ''' Hard pool all the way '''
    def move(self, i, data, funcs):
        if data['hard_pool_full']:
            funcs['hard_pool'](1)
            # clear the pool
            funcs['hard_pool_clear']()

class HardPooler(Controller):
    ''' Hard pool all the way '''
    def move(self, i, data, funcs):
        funcs['hard_pool'](1)

class PureRandom(Controller):
    ''' Randomly pick a discrete action '''
    def move(self, i, data, funcs):
        action = random.choice([0, 1, 2, 3])

        if action == 0:
            pass
        elif action == 1:
            funcs['merge_increment_pool'](1, 0.1, 0)
        elif action == 2:
            funcs['merge_increment_pool'](1, 0.1, 0.1)
        elif action == 3:
            funcs['merge_increment_pool'](1, 0, 0.1)

class NoPolicy(Controller):
    ''' No policy, only used to collect data '''
    pass

class RelevantPooler(Controller):
    ''' Only pool with relevant '''
    def move(self, i, data, funcs):
        funcs['pool'](data['pool_relevant'])

class MDAE(Controller):
    ''' Incremental autoencoders '''
    __slots__ = [ 'delta_n' ]
    def __init__(self):
        self.delta_n = 20

    def move(self, i, data, funcs):
        if i <= 30:
            return

        if data['hard_pool_full']:
            # constants for online incremental autoencoder
            e1, e2 = 0.01, 0.01

            import math
            reduction = data['mea_30'][-1] / (data['mea_30'][-2] if data['mea_30'][-2] != 0 else 0.00001)

            if reduction < 1 - e1:
                self.delta_n += 1
            elif reduction > 1 - e2:
                self.delta_n //= 2
            delta_m = math.ceil(self.delta_n * 0.5)

            funcs['merge_increment_hard_pool'](1, delta_m / data['initial_size'], self.delta_n / data['initial_size'])
            print('n', self.delta_n, 'm', delta_m)

            # clear the pool
            funcs['hard_pool_clear']()

class ContinuousState(Controller):
    ''' Q learning with a single continuous value for state and discrete action values '''
    __slots__ = [ 'learning_rate', 'discount_rate', 'prev_state', 'prev_action', 'q', 'start_time', 'action_log' ]

    class Action(IntEnum):
        ''' Action '''
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

        # q values are stored as q[action][state] = value
        self.q = defaultdict(dict)
        self.time_limit = time_limit
        self.action_log = []

    def move(self, i, data, funcs):
        if i <= 30:
            funcs['pool'](1)
            self.action_log.append(str(self.Action.pool))
            return

        # determine current state
        def ma_state(name):
            return 0 if len(data[name]) < 2 else data[name][-1] - data[name][-2]

        state = (data['r_15'][-1], data['neuron_balance'], ma_state('mea_5'), ma_state('mea_15'), ma_state('mea_30'))

        # create a GP for every discrete action
        gps = { }
        for a, value_dict in self.q.items():
            # don't bother if we don't have two samples
            if len(value_dict) < 2:
                continue

            x, y = zip(*value_dict.items())
            # create a GP with values
            gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU=1, nugget=0.1)
            gp.fit(np.array(x), np.array(y))
            gps[a] = gp

        if self.prev_state or self.prev_action:
            # determine reward
            #reward = 1 - (data['mea_5'][-1] / data['mea_5'][-2])
            reward = - data['error_log'][-1]
            #time_penalty = max(0, (self.prev_time - self.time_limit)) * 0.2
            #reward -= time_penalty

            # penalise increment
            neuron_penalty = 0
            if data['neuron_balance'] > 2 or data['neuron_balance'] < 1:
                neuron_penalty = 2 * abs(1 - data['neuron_balance'])

            reward -= neuron_penalty

            print('reward', reward, 'prev_time', self.prev_time, 'neuron_penalty', neuron_penalty)

            if len(gps) == 0:
                sample = reward
            else:
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict([self.prev_state])[0])) for gp in gps.values())

            if self.prev_state in self.q[self.prev_action]:
                self.q[self.prev_action][self.prev_state] = (1 - self.learning_rate) * self.q[self.prev_action][self.prev_state] + self.learning_rate * sample
            else:
                self.q[self.prev_action][self.prev_state] = sample

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

        print('state', state, 'action', action)
        start_time = time.time()

        # do the action
        to_move = (data['initial_size'] * 0.1) / (data['initial_size'] * data['neuron_balance'])
        if action == self.Action.pool:
            funcs['pool'](1)
        elif action == self.Action.reduce:
            funcs['merge_increment_pool'](data['pool_relevant'], to_move, 0)
        elif action == self.Action.increment:
            funcs['merge_increment_pool'](data['pool_relevant'], 0, to_move)

        self.prev_time = time.time() - start_time

        self.prev_action = action
        self.prev_state = state

        self.action_log.append(str(action))

    def end(self):
        # return the q state
        return [ { 'name': 'q_state.json', 'json': json.dumps({ str(k): { str(tup): value for tup, value in v.items() } for k, v in self.q.items() }) }, { 'name': 'actions.json', 'json': json.dumps(self.action_log) } ]
