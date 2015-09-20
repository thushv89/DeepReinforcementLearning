__author__ = 'Thushan Ganegedara'

import functools
import itertools
import theano
import theano.tensor as T
import numpy as np
import logging
import os
from math import ceil

def identity(x):
    return x

def chained_output(layers, x):
    '''
    This method is applying the given transformation (lambda expression) recursively
    to a sequence starting with an initial value (i.e. x)
    :param layers: sequence to perform recursion
    :param x: Initial value to start recursion
    :return: the final value (output after input passing through multiple neural layers)
    '''
    return functools.reduce(lambda acc, layer: layer.output(acc), layers, x)

def iterations_shim(train, iterations):
    ''' Repeat calls to this function '''
    def func(i):
        for _ in range(iterations):
            result = train(i)
        return result

    return func

def iterations_shim_early_stopping(train, validate, valid_size, iterations,frequency, validation_type='n_rand_for_train',n=10):

    def func(i):
        #we want to minimize best_valid_loss, so we shoudl start with largest
        best_valid_loss = np.inf
        patience = 0.5 * iterations # look at this many examples
        patience_increase = 1.5
        improvement_threshold = 0.995

        if validation_type == 'full':
            v_batch_idx = np.arange(0,valid_size)

        if validation_type == 'n_rand_for_train':
            v_batch_idx = np.random.uniform(low = 0, high = valid_size-1, size=n)
            print('random batches: ', list(v_batch_idx))

        for iter in range(iterations):
            print('early_stopping iteration ', str(iter))

            # the number of minibatches to go through before checking validation set
            validation_freq = min(frequency,int(patience/2))

            t_result = train(i)

            # this is an operation done in cycles. 1 cycle is iter+1/validation_freq
            # doing this every epoch
            if iter % validation_freq == 0:
                print('validating')
                if validation_type == 'n_rand_for_validation':
                    v_batch_idx = np.random.uniform(low = 0, high = valid_size-1, size=n)
                    print('random batches: ', list(v_batch_idx))

                v_results = []
                for v_i in v_batch_idx:
                    v_results.append(validate(int(v_i)))
                curr_valid_loss = np.min(v_results)
                print('curr valid loss: ', curr_valid_loss, ' best_valid_loss: ', best_valid_loss)

                if curr_valid_loss < best_valid_loss:

                    if (curr_valid_loss < best_valid_loss * improvement_threshold):
                        prev_patience = patience
                        patience = max(patience, iter * patience_increase)
                        print('patience improve: ', prev_patience, ' -> ', patience)
                    best_valid_loss = curr_valid_loss

            # patience is here to check the maximum number of iterations it should check
            # before terminating
            if patience <= iter:
                print('early stopping on iter: ', iter, ' (<', patience, ')')
                break

        return [t_result,best_valid_loss]

    return func

class Transformer(object):

    #__slots__ save memory by allocating memory only to the varibles defined in the list
    __slots__ = ['layers','arcs', '_x','_y','_logger','use_error']

    def __init__(self,layers, arcs, use_error):
        self.layers = layers
        self._x = None
        self._y = None
        self._logger = None
        self.arcs = arcs
        self.use_error = use_error


    def make_func(self, x, y, batch_size, output, updates, transformed_x = identity):
        '''
        returns a Theano function that takes x and y inputs and return the given output using given updates
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size: batch size
        :param output: the output to calculate (symbolic)
        :param update: how to get to output from input
        :return: Theano function
        '''
        idx = T.iscalar('idx')
        given = {
            self._x : transformed_x(x[idx * batch_size : (idx + 1) * batch_size]),
            self._y : y[idx * batch_size : (idx + 1) * batch_size]
        }

        return theano.function(inputs=[idx],outputs=output, updates=updates, givens=given, on_unused_input='warn')

    def process(self, x, y):
        '''
        Visit function with visitor pattern
        :param x:
        :param y:
        :return:
        '''
        pass

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        '''
        Train the network with given params
        :param learning_rate: How fast it learns
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size:
        :return: None
        '''
        pass

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        '''
        Validate the network with given parames
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        '''
        Calculate error
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass


class DeepAutoencoder(Transformer):
    ''' General Deep Autoencoder '''
    def __init__(self,layers, corruption_level, rng, lam=0.0):
        super().__init__(layers, 1, False)
        self._rng = rng
        self._corr_level = corruption_level
        self.lam = lam

        self.theta = None
        self.cost = None
        self.cost_before_reg = None
        # Need to find out what cost_vector is used for...
        self.cost_vector = None
        self.weight_regularizer = None

    def process(self, x, y):
        self._x = x
        self._y = y

        # encoding input
        for layer in self.layers:
            W, b_prime = layer.W, layer.b_prime

            #if rng is specified corrupt the inputs
            if self._rng:
                x_tilde = self._rng.binomial(size=(x.shape[0], x.shape[1]), n=1,  p=(1 - self._corr_level), dtype=theano.config.floatX) * x
                y = layer.output(x_tilde)
            else:
                y = layer.output(x)
                # z = T.nnet.sigmoid(T.dot(y, W.T) + b_prime) (This is required for regularization)

            x = y

        # decoding output and obtaining reconstruction
        for layer in reversed(self.layers):
            W, b_prime = layer.W, layer.b_prime
            x = T.nnet.sigmoid(T.dot(x,W.T) + b_prime)


        self.weight_regularizer = T.sum(T.sum(self.layers[0].W**2, axis=1))
        #weight_reg_coeff = (0.5**2) * self.layers[0].W.shape[0]*self.layers[0].W.shape[1]
        for i,layer in enumerate(self.layers[1:]):
            #self.weight_regularizer = T.dot(self.weight_regularizer**2, self.layers[i+1].W**2)
            self.weight_regularizer += T.sum(T.sum(layer.W**2, axis=1))
            #weight_reg_coeff += (0.1**2) * layer.W.shape[0]*layer.W.shape[1]

        # NOTE: weight regularizer should NOT go here. Because cost_vector is a (batch_size x 1) vector
        # where each element is cost for each element in the batch
        # costs
        # cost vector seems to hold the reconstruction error for each training case.
        # this is required for getting inputs with reconstruction error higher than average
        self.cost_vector = T.sum(T.nnet.binary_crossentropy(x, self._x),axis=1)

        self.theta = [ param for layer in self.layers for param in [layer.W, layer.b, layer.b_prime]]
        self.cost_before_reg = T.mean(self.cost_vector)
        self.cost = T.mean(self.cost_vector) + (self.lam * self.weight_regularizer)

        return None

    def train_func(self, _, learning_rate, x, y, batch_size, transformed_x=identity):
        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]
        return self.make_func(x=x,y=y,batch_size=batch_size,output=[self.cost,self.cost_before_reg], updates=updates, transformed_x=transformed_x)

    def indexed_train_func(self, arc, learning_rate, x, batch_size, transformed_x):

        nnlayer = self.layers[arc]
        # clone is used to substitute a computational subgraph
        transformed_cost = theano.clone(self.cost, replace={self._x : transformed_x(self._x)})

        # update only a set of neurons specified by index
        updates = [
            (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx], - learning_rate * T.grad(transformed_cost, nnlayer.W)[:,nnlayer.idx].T)),
            (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx], - learning_rate * T.grad(transformed_cost,nnlayer.b)[nnlayer.idx])),
            (nnlayer.b_prime, - learning_rate * T.grad(transformed_cost, nnlayer.b_prime))
        ]

        idx = T.iscalar('idx')
        givens = {self._x: x[idx * batch_size:(idx+1) * batch_size]}

        # using on_unused_inputs warn because, selected neurons could be "not depending on all the weights"
        # all the weights are a part of the cost. So it give an error otherwise
        return theano.function([idx,nnlayer.idx], None, updates=updates, givens=givens, on_unused_input='warn')

    def validate_func(self, _, x, y, batch_size, transformed_x=identity):
        return self.make_func(x=x,y=y,batch_size=batch_size,output=[self.cost,self.cost_before_reg], updates=None, transformed_x=transformed_x)

    def get_hard_examples(self, _, x, y, batch_size, transformed_x=identity):
        '''
        Returns the set of training cases (above avg reconstruction error)
        :param _:
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        # sort the values by cost and get the top half of it (above average error)
        indexes = T.argsort(self.cost_vector)[(self.cost_vector.shape[0] // 2):]
        return self.make_func(x=x, y=y, batch_size=batch_size, output=[self._x[indexes], self._y[indexes]], updates=None, transformed_x=transformed_x)

class StackedAutoencoder(Transformer):
    ''' Stacks a set of autoencoders '''
    def __init__(self, layers, corruption_level, rng):
        super().__init__(layers, len(layers), False)
        self._autoencoders = [DeepAutoencoder([layer], corruption_level, rng) for layer in layers]

    def process(self, x, y):
        self._x = x
        self._y = y

        for autoencoder in self._autoencoders:
            autoencoder.process(x,y)

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        return self._autoencoders[arc].train_func(0, learning_rate,x,y,batch_size, lambda x: chained_output(self.layers[:arc],transformed_x(x)))

    def validate_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._autoencoders[arc].validate_func(0,x,y,batch_size,lambda x: chained_output(self.layers[:arc],transformed_x(x)))

class Softmax(Transformer):

    def __init__(self, layers, iterations):
        super().__init__(layers, 1, True)

        self.theta = None
        self.results = None
        self._errors = None
        self.cost_vector = None
        self.cost = None
        self.iterations = iterations
        self.last_out = None
        self.p_y_given_x = None
        self.y_mat = None

    def process(self, x, y):
        self._x = x
        self._y = y

        self.last_out = chained_output(self.layers, x);
        self.p_y_given_x = T.nnet.softmax(chained_output(self.layers, x))

        self.results = T.argmax(self.p_y_given_x, axis=1)

        self.theta = [param for layer in self.layers for param in [layer.W, layer.b]]
        self._errors = T.mean(T.neq(self.results,y))
        self.cost_vector = -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        self.cost = T.mean(self.cost_vector)

        return None

    def get_y_as_vec_func(self, y,batch_size):

        self.y_mat = theano.shared(np.zeros((batch_size,10),dtype=theano.config.floatX))
        one_vec = T.ones_like(self._y)

        idx = T.iscalar('idx')

        y_mat_update = [(self.y_mat, T.inc_subtensor(self.y_mat[T.arange(self._y.shape[0]), self._y],1))]

        given = {
            self._y : y[idx * batch_size : (idx + 1) * batch_size]
        }

        return theano.function(inputs=[idx],outputs=[], updates=y_mat_update, givens=given, on_unused_input='warn')

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations=None):

        if iterations is None:
            iterations = self.iterations

        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]

        train = self.make_func(x,y,batch_size,self.cost,updates,transformed_x)
        ''' all print statements for this method returned None when I used iterations_shim'''
        ''' because func inside iteration_shim didn't return anything at the moment '''
        return iterations_shim(train, iterations)

    def train_with_early_stop_func(self, arc, learning_rate, x, y, v_x, v_y, batch_size, transformed_x=identity, iterations=None):
        if iterations is None:
            iterations = self.iterations

        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]

        train = self.make_func(x,y,batch_size,self.cost,updates,transformed_x)
        validate = self.make_func(v_x, v_y, batch_size, self.cost, None, transformed_x)

        valid_size = v_y.get_value().shape[0]/batch_size

        return iterations_shim_early_stopping(train, validate, valid_size, iterations, 10)

    def train_with_early_stop_func_v2(self, arc, learning_rate, x, y, v_x, v_y, batch_size, transformed_x=identity, iterations=None):
        if iterations is None:
            iterations = self.iterations

        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]

        train = self.make_func(x,y,batch_size,self.cost,updates,transformed_x)
        validate = self.make_func(v_x, v_y, batch_size, self.cost, None, transformed_x)

        return [train, validate]

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self.make_func(x,y,batch_size,self.cost,None,transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x,y,batch_size,self._errors,None, transformed_x)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x,y,batch_size,[self._y,self.results],None, transformed_x)

    def get_y_labels(self, act, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

class Pool(object):

    #theano.config.compute_test_value = 'warn'
    ''' A ring buffer (Acts as a Queue) '''
    __slots__ = ['size', 'max_size', 'position', 'data', 'data_y', '_update']

    def __init__(self, row_size, max_size):
        self.size = 0
        self.max_size = max_size
        self.position = 0

        self.data = theano.shared(np.empty((max_size, row_size), dtype=theano.config.floatX), 'pool' )
        self.data_y = theano.shared(np.empty(max_size, dtype='int32'), 'pool_y')

        x = T.matrix('new_data')
        y = T.ivector('new_data_y')
        pos = T.iscalar('update_index')

        # update statement to add new data from the position of the last data point
        update = [
            (self.data, T.set_subtensor(self.data[pos:pos+x.shape[0]],x)),
            (self.data_y, T.set_subtensor(self.data_y[pos:pos+y.shape[0]],y))]

        # function to update the data and data_y
        self._update = theano.function([pos, x, y], updates=update)

    def add(self, x, y, rows=None):

        if not rows:
            rows = x.shape[0]

        # get the latter portion of x and y (of size max_size)
        if rows > self.max_size:
            x = x[rows - self.max_size:]
            y = y[rows - self.max_size:]

        # if new data size + current position exceed max_size
        if rows + self.position > self.max_size:
            available_size = self.max_size - self.position
            self._ring_add(x[:available_size], y[:available_size])
            x = x[available_size:]
            y = y[available_size:]

        self._ring_add(x,y)

    def add_from_shared(self, index, batch_size, x, y):
        self.add(x[index * batch_size:(index+1) * batch_size].eval(), y[index * batch_size:(index+1) * batch_size].eval(), batch_size)

    # this someway returns batch indexes
    def as_size(self, new_size, batch_size):
        batches = new_size // batch_size
        starting_index = self.position // batch_size
        index_space = self.size // batch_size
        return [(starting_index - i + index_space) % index_space for i in range(batches)]

    def clear(self):
        self.size = 0
        self.position = 0

    def _ring_add(self, x, y):
        self._update(self.position, x, y)
        self.size = min(self.size + x.shape[0], self.max_size)
        self.position = (self.position + x.shape[0]) % self.max_size

class StackedAutoencoderWithSoftmax(Transformer):

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations']

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super().__init__(layers, 1, True)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._layered_autoencoders = [DeepAutoencoder([self.layers[i]], corruption_level, rng)
                                       for i, layer in enumerate(self.layers[:-1])] #[:-1] gets all items except last
        #self._softmax = Softmax(layers,iterations)
        self._softmax = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)


    def process(self, x, y):
        self._x = x
        self._y = y

        self._autoencoder.process(x,y)
        self._softmax.process(x,y)

        for ae in self._layered_autoencoders:
            ae.process(x, y)

    def train_func(self, arc, learning_rate, x, y, batch_size, early_stopping, v_x, v_y, transformed_x=identity):

        layer_greedy = [ ae.train_func(arc, learning_rate, x,  y, batch_size, lambda x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._layered_autoencoders) ]
        ae_finetune_func = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)
        if early_stopping:
            softmax_train_func,softmax_finetune_func = self._softmax.train_with_early_stop_func_v2(0, learning_rate, x, y, v_x, v_y, batch_size, transformed_x)
        else:
            softmax_train_func = self._softmax.train_func(0,learning_rate,x,y,batch_size)

        def pre_train(batch_id):
            greedy_costs = []
            for i in range(len(self.layers)-1):
                layer_greedy[i](int(batch_id))
            pre_cost,pre_cost_b4_reg = ae_finetune_func(batch_id)
            return pre_cost

        def finetune(batch_id):
            t_cost = softmax_train_func(batch_id)
            return t_cost

        def validate(batch_id):
            f_cost = softmax_finetune_func(batch_id)
            return f_cost

        return [pre_train,finetune,validate]

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y,batch_size)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size)

    def get_y_labels(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

class MergeIncrementingAutoencoder(Transformer):

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations']

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super().__init__(layers, 1, False)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._layered_autoencoders = [DeepAutoencoder([self.layers[i]], corruption_level, rng)
                                       for i, layer in enumerate(self.layers[:-1])] #[:-1] gets all items except last
        self._softmax = Softmax(layers,iterations)
        self._combined_objective = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)

    def process(self, x, y):
        self._x = x
        self._y = y
        self._autoencoder.process(x,y)
        self._softmax.process(x,y)
        self._combined_objective.process(x,y)
        for ae in self._layered_autoencoders:
            ae.process(x,y)

    def merge_inc_func(self, learning_rate, batch_size, x, y):

        m = T.matrix('m')
        # map operation applies a certain function to a sequence. This is the upper part of cosine dist eqn
        m_dists, _ = theano.map(lambda v: T.sqrt(T.dot(v, v.T)), m)
        # dimshuffle(0,'x') is converting N -> Nx1
        m_cosine = (T.dot(m, m.T)/m_dists) / m_dists.dimshuffle(0,'x')

        # T.tri gives a matrix with 1 below diagonal (including diag) and zero elsewhere
        # flatten() gives a one dim tensor having all the data in original
        # finfo gives the maximum value of floats
        # seems np.finfo(theano.config.floatX).max is used to make result
        # it is unwise to use something like m_cosine * T.tri instead of this approach,
        # as if some cosine_dist are zero they will be confused with the other zero items (resulted zeros of multiplication) in the argsort
        m_ranks = T.argsort((m_cosine - T.tri(m.shape[0]) * np.finfo(theano.config.floatX).max).flatten())[(m.shape[0] * (m.shape[0]+1)) // 2:]

        # function for getting the merge scores (cosine distance) for neurons
        score_merges = theano.function([m], m_ranks)

        # greedy layer-wise training
        layer_greedy = [ae.indexed_train_func(0, learning_rate, x, batch_size, lambda  x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._layered_autoencoders)]
        # fintune is done by optimizing cross-entropy between x and reconstructed_x
        finetune = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)
        # actual fine tuning using softmax error + reconstruction error
        combined_objective_tune = self._combined_objective.train_func(0, learning_rate, x, y, batch_size)

        # set up cost function
        mi_cost = self._softmax.cost + self.lam * self._autoencoder.cost
        mi_updates = []

        # calculating merge_inc updates
        # increment a subtensor by a certain value
        for i, nnlayer in enumerate(self._autoencoder.layers):
            # do the inc_subtensor update only for the first layer
            # update the rest of the layers normally
            if i == 0:
                # removed ".T" in the T.grad operation. It seems having .T actually
                # causes a dimension mismatch
                mi_updates += [ (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx],
                                - learning_rate * T.grad(mi_cost, nnlayer.W)[:,nnlayer.idx])) ]
                mi_updates += [ (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx],
                                - learning_rate * T.grad(mi_cost,nnlayer.b)[nnlayer.idx])) ]
            else:
                mi_updates += [(nnlayer.W, nnlayer.W - learning_rate * T.grad(mi_cost, nnlayer.W))]
                mi_updates += [(nnlayer.b, nnlayer.b - learning_rate * T.grad(mi_cost,nnlayer.b))]

            mi_updates += [(nnlayer.b_prime, -learning_rate * T.grad(mi_cost,nnlayer.b_prime))]

        softmax_theta = [self.layers[-1].W, self.layers[-1].b]

        mi_updates += [(param, param - learning_rate * grad)
                       for param, grad in zip(softmax_theta, T.grad(mi_cost, softmax_theta))]

        idx = T.iscalar('idx')

        given = {
            self._x : x[idx*batch_size : (idx+1) * batch_size],
            self._y : y[idx*batch_size : (idx+1) * batch_size]
        }

        mi_train = theano.function([idx, self.layers[0].idx], mi_cost, updates=mi_updates, givens=given)

        # the merge is done only for the first hidden layer.
        # apperantly this has been tested against doing this for all layers.
        # but doing this only to the first layer has achieved the best performance
        def merge_model(pool_indexes, merge_percentage, inc_percentage):
            '''
            Merge/Increment the batch using given pool of data
            :param pool_indexes:
            :param merge_percentage:
            :param inc_percentage:
            :return:
            '''

            prev_map = {}
            prev_dimensions = self.layers[0].initial_size[0]

            used = set()
            empty_slots = []

            # first layer
            layer_weights = self.layers[0].W.get_value().T.copy()
            layer_bias = self.layers[0].b.get_value().copy()

            # initialization of weights
            init = 4 * np.sqrt(6.0 / (sum(layer_weights.shape)))

            # number of nodes to merge or increment
            merge_count = int(merge_percentage * layer_weights.shape[0])
            inc_count = int(inc_percentage * layer_weights.shape[0])

            # if there's nothing to merge or increment
            if merge_count == 0 and inc_count == 0:
                return

            # get the nodes ranked highest to lowest (merging order)
            for index in score_merges(layer_weights):
                # all merges have been performed
                if len(empty_slots) == merge_count:
                    break

                # x and y coordinates created out of index (assume these are the two nodes
                # to merge)
                x_i, y_i = index % layer_weights.shape[0], index // layer_weights.shape[0]

                # if x_i and y_i are not in "used"`  list
                # 'used' contains merged nodes
                # 'empty_slots' are the remaining nodes after the merge
                if x_i not in used and y_i not in used:
                    # update weights and bias with avg
                    layer_weights[x_i] = (layer_weights[x_i] + layer_weights[y_i])/2
                    layer_bias[x_i] = (layer_bias[x_i] + layer_bias[y_i])/2

                    #add it to the used list
                    used.update([x_i,y_i])
                    empty_slots.append(y_i)

            #get the new size of layer
            new_size = layer_weights.shape[0] + inc_count - len(empty_slots)
            current_size = layer_weights.shape[0]

            # if new size is less than current... that is reduce operation
            if new_size < current_size:
                non_empty_slots = sorted(list(set(range(0,current_size)) - set(empty_slots)), reverse=True)[:len(empty_slots)]
                prev_map = dict(zip(empty_slots, non_empty_slots))

                for dest, src in prev_map.items():
                    layer_weights[dest] = layer_weights[src]
                    layer_weights[src] = np.asarray(self.rng.uniform(low=init, high=init, size=layer_weights.shape[1]), dtype=theano.config.floatX)

                empty_slots = []
            # increment operation
            else:
                prev_map = {}

            # new_size: new layer size after increment/reduce op
            # prev_dimension: size of input layer
            new_layer_weights = np.zeros((new_size,prev_dimensions), dtype = theano.config.floatX)
            print('Old layer 1 size: ',layer_weights.shape[0])
            print('New layer 1 size: ',new_layer_weights.shape[0])

            # the existing values from layer_weights copied to new layer weights
            # and it doesn't matter if layer_weights.shape[0]<new_layer_weights.shape[0] it'll assign values until it reaches the end
            new_layer_weights[:layer_weights.shape[0], :layer_weights.shape[1]] = layer_weights[:new_layer_weights.shape[0], :new_layer_weights.shape[1]]

            # get all empty_slots that are < new_size  +  list(prev_size->new_size)
            empty_slots = [slot for slot in empty_slots if slot < new_size] + list(range(layer_weights.shape[0],new_size))
            new_layer_weights[empty_slots] = np.asarray(self.rng.uniform(low=-init, high=init, size=(len(empty_slots), prev_dimensions)), dtype=theano.config.floatX)

            # fills missing entries with zero
            print('New layer 1 size (bias): ',layer_bias.shape[0])
            layer_bias.resize(new_size, refcheck=False)
            print('New layer 1 size (bias): ',layer_bias.shape[0])
            layer_bias_prime = self.layers[0].b_prime.get_value().copy()
            layer_bias_prime.resize(prev_dimensions)

            prev_dimensions = new_layer_weights.shape[0]

            self.layers[0].W.set_value(new_layer_weights.T)
            self.layers[0].b.set_value(layer_bias)
            self.layers[0].b_prime.set_value(layer_bias_prime)

            #if empty_slots:
            #    for _ in range(int(self.iterations)):
            #        for i in pool_indexes:
            #            layer_greedy[0](i, empty_slots)

            last_layer_weights = self.layers[1].W.get_value().copy()

            for dest, src in prev_map.items():
                last_layer_weights[dest] = last_layer_weights[src]
                last_layer_weights[src] = np.zeros(last_layer_weights.shape[1])

            last_layer_weights.resize((prev_dimensions, self.layers[1].initial_size[1]),refcheck=False)
            last_layer_prime = self.layers[1].b_prime.get_value().copy()
            last_layer_prime.resize(prev_dimensions, refcheck=False)

            self.layers[1].W.set_value(last_layer_weights)
            self.layers[1].b_prime.set_value(last_layer_prime)

            for _ in range(int(self.iterations)):
                for i in pool_indexes:
                    finetune(i)

            return empty_slots

        return merge_model


class CombinedObjective(Transformer):

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super().__init__(layers, 1, True)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._softmax = Softmax(layers,1)
        self.lam = lam
        self.iterations = iterations
        self.cost = None

    def process(self, x, yy):
        self._x = x
        self._y = yy

        self._autoencoder.process(x,yy)
        self._softmax.process(x,yy)

        self.cost = self._softmax.cost + self.lam * self._autoencoder.cost

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations=None):

        if iterations is None:
            iterations = self.iterations

        #combined_cost = self._softmax.cost + self.lam * 0.5

        theta = []
        for layer in self.layers[:-1]:
            theta += [layer.W, layer.b, layer.b_prime]
        theta += [self.layers[-1].W, self.layers[-1].b] #softmax layer

        update = [(param, param - learning_rate * grad) for param, grad in zip(theta, T.grad(self.cost,wrt=theta))]
        comb_obj_finetune_func = self.make_func(x, y, batch_size, self.cost, update, transformed_x)
        return iterations_shim(comb_obj_finetune_func, iterations)

    def train_with_early_stop_func(self, arc, learning_rate, x, y, v_x, v_y, batch_size, transformed_x=identity, iterations=None):
        if iterations is None:
            iterations = self.iterations

        combined_cost = self._softmax.cost + self.lam * self._autoencoder.cost
        #combined_cost = self._softmax.cost + self.lam * 0.5

        theta = []
        for layer in self.layers[:-1]:
            theta += [layer.W, layer.b, layer.b_prime]
        theta += [self.layers[-1].W, self.layers[-1].b] #softmax layer

        updates = [(param, param - learning_rate*grad) for param, grad in zip(theta, T.grad(combined_cost,wrt=theta))]

        train = self.make_func(x,y,batch_size,combined_cost,updates,transformed_x)
        validate = self.make_func(v_x, v_y, batch_size, combined_cost, None, transformed_x)

        valid_size = v_x.get_value().shape[0]/batch_size

        return iterations_shim_early_stopping(train, validate, valid_size, iterations, 10)

    def train_with_early_stop_func_v2(self, arc, learning_rate, x, y, v_x, v_y, batch_size, transformed_x=identity, iterations=None):
        if iterations is None:
            iterations = self.iterations

        combined_cost = self._softmax.cost + self.lam * self._autoencoder.cost
        #combined_cost = self._softmax.cost + self.lam * 0.5

        theta = []
        for layer in self.layers[:-1]:
            theta += [layer.W, layer.b, layer.b_prime]
        theta += [self.layers[-1].W, self.layers[-1].b] #softmax layer

        updates = [(param, param - learning_rate*grad) for param, grad in zip(theta, T.grad(combined_cost,wrt=theta))]

        train = self.make_func(x,y,batch_size,combined_cost,updates,transformed_x)
        validate = self.make_func(v_x, v_y, batch_size, combined_cost, None, transformed_x)

        return [iterations_shim(train,iterations), validate]

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y, batch_size, transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size, transformed_x)

    def get_y_labels(self, act, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

class DeepReinforcementLearningModel(Transformer):

    def __init__(self, layers, corruption_level, rng, iterations, epochs, lam, mi_batch_size, pool_size, valid_pool_size, controller):

        super().__init__(layers, 1, True)

        self._mi_batch_size = mi_batch_size
        self._controller = controller
        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._softmax = CombinedObjective(layers, corruption_level, rng, lam=lam, iterations=iterations)
        self._merge_increment = MergeIncrementingAutoencoder(layers, corruption_level, rng, lam=lam, iterations=iterations)

        self.epochs = epochs
        # _pool : has all the data points
        # _hard_pool: has data points only that are above average reconstruction error
        self._pool = Pool(layers[0].initial_size[0], pool_size)
        self._hard_pool = Pool(layers[0].initial_size[0], pool_size)

        self._valid_pool = Pool(layers[0].initial_size[0], valid_pool_size)
        self._valid_hard_pool = Pool(layers[0].initial_size[0], valid_pool_size)

        self.iterations = iterations
        self.lam = lam

        self.train_distribution = {}
        self.valid_distribution = {}

        self._error_log = []
        self._reconstruction_log = []
        self._neuron_balance_log = []
        self._network_size_log = []

    def process(self, x, y):
        self._x = x
        self._y = y
        self._autoencoder.process(x, y)
        self._softmax.process(x, y)
        self._merge_increment.process(x, y)

    # add_from_shared uses a queue like adding mechanism, which means everything before is deleted
    # if it run out of space
    def build_valid_pool(self, v_x, v_y, batch_size, apply_x = identity):
        hard_examples_valid_func = self._autoencoder.get_hard_examples(0, v_x, v_y, batch_size, apply_x)

        def update_pools(v_batch_id):
            self._valid_pool.add_from_shared(v_batch_id, batch_size, v_x, v_y)
            self._valid_hard_pool.add(*hard_examples_valid_func(v_batch_id))
            print('added batch: ',v_batch_id, ' valid pool size: ',self._valid_pool.size)

        return update_pools

    # pool_relevent pools all the batches from the current to the last batch that satisfies
        # cosine_dist(batch) < mean
    def pool_relevant(self, pool, distribution, batch_size, current=None):

        if current == None:
            current = distribution[-1]

        def magnitude(x):
            '''  returns sqrt(sum(v(i)^2)) '''
            return sum((v **2 for v in x.values() )) ** 0.5

        def compare(x,y):
            '''  Calculate Cosine distance between x and y '''
            top = 0

            for k in set(x) | set(y):
                xval, yval = x[k] if k in x else 0, y[k] if k in y else 0
                top += xval * yval

            return top / magnitude(x) * magnitude(y)

        # score over batches for this pool
        batches_covered = int(pool.size // batch_size)
        print('batches covered: ', batches_covered)
        print('range: ',range(-1,-1 - batches_covered))
        # the below statement get the batch scores, batch scores are basically
        # the cosine distance between a given batch and the current batch (last)
        # for i in range(-1,-1 - batches_covered) gets the indexes as minus indices as it is easier way to count from back of array
        batch_scores = [(i % batches_covered, compare(current, distribution[i])) for i in range(-1,-1 - batches_covered)]
        # mean is the mean cosine score
        mean = np.mean([ v[1] for v in batch_scores ])

        # takewhile(predicate, iterable) returns elements until the predicate is true
        # get all the batches with batch score greater than mean
        last = [0, 0]
        for last in itertools.takewhile(lambda s: s[1] > mean, batch_scores):
            pass

        return 1 - (last[0] / batches_covered)

    def train_func(self, arc, learning_rate, x, y, batch_size, early_stop, v_x, v_y, apply_x=identity):
        batch_pool = Pool(self.layers[0].initial_size[0], batch_size)
        valid_batch_pool = Pool(self.layers[0].initial_size[0], batch_size)

        layer_greedy = [ ae.train_func(arc, learning_rate, x,  y, batch_size, lambda x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._merge_increment._layered_autoencoders) ]
        ae_finetune_func = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)

        train_func = self._softmax.train_func(arc, learning_rate, x, y, batch_size, apply_x)

        reconstruction_func = self._autoencoder.validate_func(arc, x, y, batch_size, apply_x)
        error_func = self.error_func(arc, x, y, batch_size, apply_x)

        merge_inc_func_batch = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, x, y)
        merge_inc_func_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._pool.data, self._pool.data_y)
        merge_inc_func_hard_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._hard_pool.data, self._hard_pool.data_y)

        hard_examples_func = self._autoencoder.get_hard_examples(arc, x, y, batch_size, apply_x)

        train_func_pool = self._softmax.train_func(arc, learning_rate, self._pool.data, self._pool.data_y, batch_size, apply_x)
        train_func_hard_pool = self._softmax.train_func(arc, learning_rate, self._hard_pool.data, self._hard_pool.data_y, batch_size, apply_x)

        neuron_balance = 1

        def train_pool(pool, pool_func, amount):

            for i in pool.as_size(int(pool.size * amount), batch_size):
                pool_func(i)

        def moving_average(log, n):

            weights = np.exp(np.linspace(-1, 0, n))
            weights /= sum(weights)
            return np.convolve(log, weights)[n-1:-n+1]

        # get early stopping
        def train_adaptively(batch_id):

            empty_slots = None
            self._error_log.append(np.asscalar(error_func(batch_id)))
            test_rec_results = reconstruction_func(batch_id)
            test_rec_err = np.asscalar(test_rec_results[0])
            test_rec_before_reg = test_rec_results[1]
            print('Reconstruction Error (with reg): ',test_rec_err, ' (w/o reg: ', test_rec_before_reg, ') batch id: ', batch_id)
            self._reconstruction_log.append(test_rec_err)
            self._neuron_balance_log.append(neuron_balance)

            batch_pool.add_from_shared(batch_id, batch_size, x, y)
            self._pool.add_from_shared(batch_id, batch_size, x, y)
            self._hard_pool.add(*hard_examples_func(batch_id))

            data = {
                'mea_30': moving_average(self._error_log, 30),
                'mea_15': moving_average(self._error_log, 15),
                'mea_5': moving_average(self._error_log, 5),
                'pool_relevant': self.pool_relevant(self._pool,self.train_distribution,batch_size),
                'initial_size': self.layers[1].initial_size[0],
                'hard_pool_full': self._hard_pool.size == self._hard_pool.max_size,
                'error_log': self._error_log,
                'errors': self._error_log[-1],
                'neuron_balance': self._neuron_balance_log[-1],
                'reconstruction': self._reconstruction_log[-1],
                'r_15': moving_average(self._reconstruction_log, 15)
            }

            def merge_increment(func, pool, amount, merge, inc):

                nonlocal neuron_balance
                print('init size: ', self.layers[1].initial_size[0], ' curr size: ', self.layers[1].W.get_value().shape[0], ' ratio: ', np.abs(1-(self.layers[1].W.get_value().shape[0]/self.layers[1].initial_size[0])))
                change = 1 + inc - merge + 0.05 * np.abs(1- (self.layers[1].W.get_value().shape[0]/self.layers[1].initial_size[0]))
                print('neuron balance', neuron_balance, '=>', neuron_balance * change)
                neuron_balance *= change

                # pool.as_size(int(pool.size * amount), self._mi_batch_size) seems to provide indexes
                nonlocal empty_slots
                empty_slots = func(pool.as_size(int(pool.size * amount), self._mi_batch_size), merge, inc)
                print('')
            funcs = {
                'merge_increment_batch' : functools.partial(merge_increment, merge_inc_func_batch, batch_pool),
                'merge_increment_pool' : functools.partial(merge_increment, merge_inc_func_pool, self._pool),
                'merge_increment_hard_pool': functools.partial(merge_increment, merge_inc_func_hard_pool, self._hard_pool),
                'pool': functools.partial(train_pool, self._pool, train_func_pool),
                'hard_pool': functools.partial(train_pool, self._hard_pool, train_func_hard_pool),
                'hard_pool_clear': self._hard_pool.clear,
            }

            #this is where reinforcement learning comes to play

            self._controller.move(len(self._error_log), data, funcs)

            train_func(batch_id)

            self._network_size_log.append(self.layers[0].W.get_value().shape[1])

            return empty_slots

        # these methods are used for early stopping
        def finetune_adaptively(empty_slots):


            # set up cost function
            mi_cost = self._softmax.cost + self.lam * self._autoencoder.cost
            mi_updates = []

            # calculating merge_inc updates
            # increment a subtensor by a certain value
            for i, nnlayer in enumerate(self._autoencoder.layers):
                # do the inc_subtensor update only for the first layer
                # update the rest of the layers normally
                if i == 0:
                    # removed ".T" in the T.grad operation. It seems having .T actually
                    # causes a dimension mismatch
                    mi_updates += [ (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx],
                                    - learning_rate * T.grad(mi_cost, nnlayer.W)[:,nnlayer.idx])) ]
                    mi_updates += [ (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx],
                                    - learning_rate * T.grad(mi_cost,nnlayer.b)[nnlayer.idx])) ]
                else:
                    mi_updates += [(nnlayer.W, nnlayer.W - learning_rate * T.grad(mi_cost, nnlayer.W))]
                    mi_updates += [(nnlayer.b, nnlayer.b - learning_rate * T.grad(mi_cost,nnlayer.b))]

                mi_updates += [(nnlayer.b_prime, -learning_rate * T.grad(mi_cost,nnlayer.b_prime))]

            softmax_theta = [self.layers[-1].W, self.layers[-1].b]

            mi_updates += [(param, param - learning_rate * grad)
                           for param, grad in zip(softmax_theta, T.grad(mi_cost, softmax_theta))]

            idx = T.iscalar('idx')

            given = {
                self._x : self._pool.data[idx*batch_size : (idx+1) * batch_size],
                self._y : self._pool.data_y[idx*batch_size : (idx+1) * batch_size]
            }

            mi_train = theano.function([idx, self.layers[0].idx], mi_cost, updates=mi_updates, givens=given)
            combined_objective_tune = self._softmax.train_func(0, learning_rate, self._pool.data, self._pool.data_y, batch_size)

            # TODO: Add pool_relevant instead of using 1 as amount and use train_distribution as distribution
            pool_indexes = self._pool.as_size(int(self._pool.size * 1), self._mi_batch_size)

            if empty_slots:
                print('Fine tuning using mi_train, Empty slots: ', empty_slots, ' with pool indexes: ', pool_indexes)
                for _ in range(self.iterations):
                    costs = []
                    for i in pool_indexes:
                        costs.append(mi_train(i, empty_slots))

            else:
                print('Fine tuning the whole network with pool indexes: ', pool_indexes)
                for _ in range(self.iterations):
                    costs = []
                    for i in pool_indexes:
                        costs.append(np.asscalar(combined_objective_tune(i)))

            return np.mean(costs)

        def finetune_validate_adaptively(v_batch_id):
            update_valid_pool_func = self.build_valid_pool(v_x,v_y,batch_size)
            update_valid_pool_func(v_batch_id)

            #valid_pool_indexes = self._valid_pool.as_size(int(self._valid_pool.size * 1), self._mi_batch_size)
            last_rel_pool_idx = self.pool_relevant(self._valid_pool,self.valid_distribution, batch_size, self.train_distribution[-1])-1
            print('Idx of last relevant pool: ', last_rel_pool_idx, ', pool size: ', self._valid_pool.size)
            print('Fine tune validation with pool indexes (Relevant):', np.arange(last_rel_pool_idx,self._valid_pool.size//batch_size))

            combined_obj_finetune,combined_obj_valid_func = self._softmax.train_with_early_stop_func_v2(0, learning_rate, x, y, self._valid_pool.data, self._valid_pool.data_y, batch_size)

            for _ in range(self.iterations):
                valid_costs = []
                for i in range(int(last_rel_pool_idx),int(self._valid_pool.size//batch_size)):
                    valid_costs.append(np.asscalar(combined_obj_valid_func(int(i))))
            return np.mean(valid_costs)

        return train_adaptively,finetune_adaptively, finetune_validate_adaptively

    def set_train_distribution(self, t_distribution):
        self.train_distribution = t_distribution

    def set_valid_distribution(self, v_distribution):
        self.valid_distribution = v_distribution

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y,batch_size)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size)

    def get_y_labels(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.get_y_labels(arc, x, y, batch_size)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

