''' Models '''
import functools
import itertools
import random

import policies

import theano
import theano.tensor as T

import numpy as np

def identity(x):
    ''' Identity function '''
    return x

def chained_output(layers, x):
    ''' Fold the output function across all layers '''
    return functools.reduce(lambda acc, layer: layer.output(acc), layers, x)

def iterations_shim(train, iterations):
    ''' Repeat calls to this function '''
    def func(i):
        for _ in range(iterations):
            train(i)
    return func

class Transformer(object):
    ''' A compositional approach to building neural networks '''
    __slots__ = [ 'layers', 'arcs', '_x', '_y', '_logger', 'use_error', 'context' ]

    def __init__(self, layers, arcs, use_error):
        self.layers = layers
        self.arcs = arcs
        self._x = None
        self._y = None
        self._logger = None
        self.use_error = use_error

    def make_func(self, prealloc_x, prealloc_y, batch_size, output, update, apply_x=lambda x: x):
        ''' Make a function that computes the given value '''
        idx = T.iscalar('idx')
        given = { self._x: apply_x(prealloc_x[idx * batch_size : (idx + 1) * batch_size])
                , self._y: prealloc_y[idx * batch_size : (idx + 1) * batch_size]
                }
        return theano.function([idx], output, updates=update, givens=given, on_unused_input='warn')

    def begin(self, context):
        ''' Pass contextual from the trainer to the model '''
        self.context = context

    def end(self):
        ''' Called when training has finished, returns the transformer's log of events '''
        return []

    def process(self, x, y):
        ''' Visit function in visitor pattern '''
        pass

    def train_func(self, arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        ''' Train the model '''
        pass

    def validate_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        ''' Validate the model '''
        pass

    def error_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        ''' Get errors '''
        pass

class DeepAutoencoder(Transformer):
    ''' Generalised deep autoencoder '''
    def __init__(self, layers, corruption_level, rng):
        super().__init__(layers, 1, False)
        self._rng = rng
        self._corruption_level = corruption_level

        self.theta = None
        self.cost = None
        self.cost_vector = None

    def process(self, x, yy):
        self._x = x
        self._y = yy

        regulariser = 0

        # encoding and inject noise
        for layer in self.layers:
            W, b_prime = layer.W, layer.b_prime

            if self._rng:
                x_tilde = self._rng.binomial(size=(x.shape[0], x.shape[1]), n=1,  p=(1 - self._corruption_level), dtype=theano.config.floatX) * x
                y = layer.output(x_tilde)
            else:
                y = layer.output(x)
                z = T.nnet.sigmoid(T.dot(y, W.T) + b_prime)

                # let d = input layer size, h = output layer size
                # shape of x is (rows, d)
                # shape of W is (d, h)
                d2l_dy2 = T.dot(z * (1 - z), W * W) # (rows, h)
                partial_dy_dx = y * (1 - y) # (rows, h)
                approx_hessian = T.dot(d2l_dy2 * partial_dy_dx * partial_dy_dx, W.T * W.T) # (rows, d)

                regulariser += (self._corruption_level / (1 - self._corruption_level)) * T.mean(T.sum(x * x * approx_hessian, axis=1))
            x = y

        # decode
        for layer in reversed(self.layers):
            W, b_prime = layer.W, layer.b_prime
            x = T.nnet.sigmoid(T.dot(x, W.T) + b_prime)

        # cost function
        self.cost_vector = T.sum(T.nnet.binary_crossentropy(x, self._x), axis=1) + 0.5 * regulariser
        self.theta = [ param for layer in self.layers for param in [ layer.W, layer.b, layer.b_prime ] ]
        self.cost = T.mean(self.cost_vector)

        return None

    def train_func(self, _, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        updates = [ (param, param - learning_rate * grad) for param, grad in zip(self.theta, T.grad(self.cost, self.theta)) ]
        return self.make_func(prealloc_x, prealloc_y, batch_size, None, updates, apply_x)

    def indexed_train_func(self, arc, learning_rate, prealloc_x, batch_size, apply_x=identity):
        ''' Train function with indexed restriction '''
        nnlayer = self.layers[arc]
        applied_cost = theano.clone(self.cost, replace={ self._x: apply_x(self._x) })

        updates = [ (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx], - learning_rate * T.grad(applied_cost, nnlayer.W)[:,nnlayer.idx].T))
                  , (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx],   - learning_rate * T.grad(applied_cost, nnlayer.b)[nnlayer.idx]))
                  , (nnlayer.b_prime, - learning_rate * T.grad(applied_cost, nnlayer.b_prime))
                  ]

        idx = T.iscalar('idx')
        givens = { self._x: prealloc_x[idx * batch_size:(idx+1) * batch_size] }
        return theano.function([idx, nnlayer.idx], None, updates=updates, givens=givens)

    def validate_func(self, _, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self.make_func(prealloc_x, prealloc_y, batch_size, [ self.cost ], None, apply_x)

    def hard_examples(self, _, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        ''' Return a function that returns hard examples (above average reconstruction error) '''
        indexes = T.argsort(self.cost_vector)[(self.cost_vector.shape[0] // 2):]
        return self.make_func(prealloc_x, prealloc_y, batch_size, [ self._x[indexes], self._y[indexes] ], None, apply_x)

class StackedAutoencoder(Transformer):
    ''' Train autoencoders layerwise by training each one in it's own arc '''
    def __init__(self, layers, corruption_level, rng):
        super().__init__(layers, len(layers), False)
        self._autoencoders = [ DeepAutoencoder([layer], corruption_level, rng) for layer in layers ]

    def process(self, x, y):
        self._x = x
        self._y = y

        for autoencoder in self._autoencoders:
            autoencoder.process(x, y)

    def train_func(self, arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._autoencoders[arc].train_func(0, learning_rate, prealloc_x, prealloc_y, batch_size, lambda x: chained_output(self.layers[:arc], apply_x(x)))

    def validate_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._autoencoders[arc].validate_func(0, prealloc_x, prealloc_y, batch_size, lambda x: chained_output(self.layers[:arc], apply_x(x)))

class Softmax(Transformer):
    ''' Treat all visited layers as MLP, with the last layer being a softmax '''

    def __init__(self, layers, iterations):
        super().__init__(layers, 1, True)

        self.theta = None
        self._errors = None
        self._cost_vector = None
        self.cost = None
        self.iterations = iterations

    def process(self, x, y):
        self._x = x
        self._y = y

        p_y_given_x = T.nnet.softmax(chained_output(self.layers, x))

        results = T.argmax(p_y_given_x, axis=1)

        self.theta = [ param for layer in self.layers for param in [layer.W, layer.b] ]
        self._errors = T.mean(T.neq(results, y))
        self._cost_vector = -T.log(p_y_given_x)[T.arange(y.shape[0]), y]
        self.cost = T.mean(self._cost_vector)

        return None

    def train_func(self, arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity, iterations=None):
        ''' Returns a function that returns the cost in the last layer '''

        if iterations is None:
            iterations = self.iterations

        updates = [ (param, param - learning_rate * grad) for param, grad in zip(self.theta, T.grad(self.cost, self.theta)) ]

        train = self.make_func(prealloc_x, prealloc_y, batch_size, None, updates, apply_x)
        return iterations_shim(train, iterations)

    def validate_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self.make_func(prealloc_x, prealloc_y, batch_size, [ self.cost ], None, apply_x)

    def error_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self.make_func(prealloc_x, prealloc_y, batch_size, [ self._errors ], None, apply_x)

class Pool(object):
    ''' An ring buffer '''
    __slots__ = ['size', 'max_size', 'position', 'data', 'data_y', '_update']

    def __init__(self, row_size, max_size):
        self.size = 0
        self.max_size = max_size
        self.position = 0

        self.data = theano.shared(np.empty((max_size, row_size), dtype=theano.config.floatX), 'pool')
        self.data_y = theano.shared(np.empty(max_size, dtype='int32'), 'pool_y')

        x = T.matrix('new_data')
        y = T.ivector('new_data_y')
        pos = T.iscalar('update_index')

        update = [ (self.data,   T.set_subtensor(self.data[pos:pos+x.shape[0]], x))
                 , (self.data_y, T.set_subtensor(self.data_y[pos:pos+y.shape[0]], y))
                 ]
        self._update = theano.function([pos, x, y], updates=update)

    def add(self, x, y, rows=None):
        ''' Add data to the pool '''
        if not rows:
            rows = x.shape[0]

        # use the data at the end if it doesn't fit
        if rows > self.max_size:
            x = x[rows - self.max_size:]
            y = y[rows - self.max_size:]

        # split x into two if it doesn't fit
        if rows + self.position > self.max_size:
            avaliable_size = self.max_size - self.position
            self._ring_add(x[:avaliable_size], y[:avaliable_size])
            x = x[avaliable_size:]
            y = y[avaliable_size:]

        self._ring_add(x, y)

    def add_from_shared(self, index, batch_size, prealloc_x, prealloc_y):
        ''' Add data from shared variable '''
        self.add(prealloc_x[index * batch_size:(index+1) * batch_size].eval(), prealloc_y[index * batch_size:(index+1) * batch_size].eval(), batch_size)

    def clear(self):
        ''' Clear the pool '''
        self.size = 0
        self.position = 0

    def as_size(self, new_size, batch_size):
        ''' Pretend the pool is of the new_size, return relevant indices for training with the given batch_size '''
        batches = new_size // batch_size
        starting_index = self.position // batch_size
        index_space = self.size // batch_size
        return [ (starting_index - i + index_space) % index_space for i in range(batches) ]

    def _ring_add(self, x, y):
        ''' Add to ring buffer, advances the position '''
        self._update(self.position, x, y)
        self.size = min(self.size + x.shape[0], self.max_size)
        self.position = (self.position + x.shape[0]) % self.max_size

class MergeIncrementingAutoencoder(Transformer):
    ''' An autoencoder with merge and increment functions '''

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations']

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super().__init__(layers, 1, False)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._layered_autoencoders = [ DeepAutoencoder([ self.layers[i] ], corruption_level, rng) for i, layer in enumerate(self.layers[:-1]) ]
        self._softmax = Softmax(layers, 1)
        self._combined_objective = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)

    def process(self, x, y):
        self._x = x
        self._y = y
        self._autoencoder.process(x, y)
        self._softmax.process(x, y)
        self._combined_objective.process(x, y)
        for ae in self._layered_autoencoders:
            ae.process(x, y)

    def merge_inc_func(self, learning_rate, batch_size, prealloc_x, prealloc_y):
        ''' Return a function that can merge/increment the model '''
        # matrix for scoring merges
        m = T.matrix('m')
        m_dists, _ = theano.map(lambda v: T.sqrt(T.dot(v, v.T)), m)
        m_cosine = (T.dot(m, m.T) / m_dists) / m_dists.dimshuffle(0, 'x')
        m_ranks = T.argsort((m_cosine - T.tri(m.shape[0]) * np.finfo(theano.config.floatX).max).flatten())[(m.shape[0] * (m.shape[0] + 1)) // 2:]

        score_merges = theano.function([m], m_ranks)

        # greedy layerwise training
        layer_greedy = [ ae.indexed_train_func(0, learning_rate, prealloc_x, batch_size, lambda x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._layered_autoencoders) ]
        finetune = self._autoencoder.train_func(0, learning_rate, prealloc_x, prealloc_y, batch_size)
        combined_objective_tune = self._combined_objective.train_func(0, learning_rate, prealloc_x, prealloc_y, batch_size)

        # set up layered merge-increment - build a cost function
        mi_cost = self._softmax.cost + self.lam * self._autoencoder.cost
        mi_updates = []

        for i, nnlayer in enumerate(self._autoencoder.layers):
            if i == 0:
                mi_updates += [ (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx], - learning_rate * T.grad(mi_cost, nnlayer.W)[:,nnlayer.idx].T)) ]
                mi_updates += [ (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx],   - learning_rate * T.grad(mi_cost, nnlayer.b)[nnlayer.idx]))     ]
            else:
                mi_updates += [ (nnlayer.W, nnlayer.W - learning_rate * T.grad(mi_cost, nnlayer.W)) ]
                mi_updates += [ (nnlayer.b, nnlayer.b - learning_rate * T.grad(mi_cost, nnlayer.b)) ]

            mi_updates += [ (nnlayer.b_prime, - learning_rate * T.grad(mi_cost, nnlayer.b_prime)) ]

        softmax_theta = [ self.layers[-1].W, self.layers[-1].b ]

        mi_updates += [ (param, param - learning_rate * grad) for param, grad in zip(softmax_theta, T.grad(mi_cost, softmax_theta)) ]

        idx = T.iscalar('idx')
        given = {
            self._x: prealloc_x[idx * batch_size : (idx + 1) * batch_size],
            self._y: prealloc_y[idx * batch_size : (idx + 1) * batch_size]
        }
        mi_train = theano.function([idx, self.layers[0].idx ], None, updates=mi_updates, givens=given)

        def merge_model(pool_indexes, merge_percentage, inc_percentage):
            ''' Merge/increment the model using the given batch '''
            prev_map = { }
            prev_dimensions = self.layers[0].initial_size[0]

            # first layer
            used = set()
            empty_slots = []
            layer_weights = self.layers[0].W.get_value().T.copy()
            layer_bias = self.layers[0].b.get_value().copy()

            init = 4 * np.sqrt(6.0 / (sum(layer_weights.shape)))

            merge_count = int(merge_percentage * layer_weights.shape[0])
            inc_count = int(inc_percentage * layer_weights.shape[0])

            if merge_count == 0 and inc_count == 0:
                return

            for index in score_merges(layer_weights):
                if len(empty_slots) == merge_count:
                    break

                x_i, y_i = index % layer_weights.shape[0], index // layer_weights.shape[0]

                if x_i not in used and y_i not in used:
                    # merge x_i with y_i
                    layer_weights[x_i] = (layer_weights[x_i] + layer_weights[y_i]) / 2
                    layer_bias[x_i] = (layer_bias[x_i] + layer_bias[y_i]) / 2

                    used.update([x_i, y_i])
                    empty_slots.append(y_i)

            new_size = layer_weights.shape[0] + inc_count - len(empty_slots)
            current_size = layer_weights.shape[0]

            # compact weights array if neccessary
            if new_size < current_size:
                non_empty_slots = sorted(list(set(range(0, current_size)) - set(empty_slots)), reverse=True)[:len(empty_slots)]
                prev_map = dict(zip(empty_slots, non_empty_slots))

                # compact the layer weights by removing the empty slots
                for dest, src in prev_map.items():
                    layer_weights[dest] = layer_weights[src]
                    layer_weights[src] = np.asarray(self.rng.uniform(low=-init, high=init, size=layer_weights.shape[1]), dtype=theano.config.floatX)

                empty_slots = []
            else:
                prev_map = { }

            # will need to add more space for new features
            new_layer_weights = np.zeros((new_size, prev_dimensions), dtype=theano.config.floatX)
            new_layer_weights[:layer_weights.shape[0], :layer_weights.shape[1]] = layer_weights[:new_layer_weights.shape[0], :new_layer_weights.shape[1]]

            # randomly initalise new neurons
            empty_slots = [ slot for slot in empty_slots if slot < new_size] + list(range(layer_weights.shape[0], new_size))
            new_layer_weights[empty_slots] = np.asarray(self.rng.uniform(low=-init, high=init, size=(len(empty_slots), prev_dimensions)), dtype=theano.config.floatX)

            layer_bias.resize(new_size)

            layer_bias_prime = self.layers[0].b_prime.get_value().copy()
            layer_bias_prime.resize(prev_dimensions)

            prev_dimensions = new_layer_weights.shape[0]

            # set the new data
            self.layers[0].W.set_value(new_layer_weights.T)
            self.layers[0].b.set_value(layer_bias)
            self.layers[0].b_prime.set_value(layer_bias_prime)

            #if empty_slots:
                ## train this layer
                #for _ in range(self.iterations):
                    #for i in pool_indexes:
                        #layer_greedy[0](i, empty_slots)

            # update the last layer's weight matrix size
            last_layer_weights = self.layers[1].W.get_value().copy()

            # apply mapping to last layer
            for dest, src in prev_map.items():
                last_layer_weights[dest] = last_layer_weights[src]
                last_layer_weights[src] = np.zeros(last_layer_weights.shape[1])

            # fix sizes
            last_layer_weights.resize((prev_dimensions, self.layers[1].initial_size[1]))
            last_layer_prime = self.layers[1].b_prime.get_value().copy()
            last_layer_prime.resize(prev_dimensions)

            self.layers[1].W.set_value(last_layer_weights)
            self.layers[1].b_prime.set_value(last_layer_prime)

            # finetune with the deep autoencoder
            for _ in range(self.iterations):
                for i in pool_indexes:
                    finetune(i)

            # finetune with supervised
            if empty_slots:
                for _ in range(self.iterations):
                    for i in pool_indexes:
                        mi_train(i, empty_slots)
            else:
                for i in pool_indexes:
                    combined_objective_tune(i)

        return merge_model

class CombinedObjective(Transformer):
    ''' Combined objective trainer '''
    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super().__init__(layers, 1, True)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._softmax = Softmax(layers, 1)
        self.lam = lam
        self.iterations = iterations

    def process(self, x, yy):
        self._x = x
        self._y = yy

        self._autoencoder.process(x, yy)
        self._softmax.process(x, yy)

    def train_func(self, arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity, iterations=None):
        ''' Returns a function that returns the cost in the last layer '''

        if iterations is None:
            iterations = self.iterations

        combined_cost = self._softmax.cost + self.lam * self._autoencoder.cost

        # collect parameters
        theta = []
        for layer in self.layers[:-1]:
            theta += [ layer.W, layer.b, layer.b_prime ]
        theta += [ self.layers[-1].W, self.layers[-1].b ]

        # gradient descent
        updates = [ (param, param - learning_rate * grad) for param, grad in zip(theta, T.grad(combined_cost, theta)) ]
        func = self.make_func(prealloc_x, prealloc_y, batch_size, None, updates, apply_x)
        return iterations_shim(func, iterations)

    def validate_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._softmax.validate_func(arc, prealloc_x, prealloc_y, batch_size, apply_x)

    def error_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._softmax.error_func(arc, prealloc_x, prealloc_y, batch_size, apply_x)

class AdaptingCombinedObjective(Transformer):
    ''' An autoencoder that has an interchangable controller that adjusts the architecture of the neural network'''

    def __init__(self, layers, corruption_level, rng, iterations, lam, mi_batch_size, pool_size, controller):
        super().__init__(layers, 1, True)

        self._mi_batch_size = mi_batch_size
        self._controller = controller
        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)

        self._softmax = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self._merge_increment = MergeIncrementingAutoencoder(layers, corruption_level, rng, lam, iterations)

        # pool for training examples
        self._pool = Pool(layers[0].initial_size[0], pool_size)
        self._hard_pool = Pool(layers[0].initial_size[0], pool_size)

        # logging state
        self._error_log = []
        self._neuron_balance_log = []
        self._reconstruction_log = []

    def process(self, x, y):
        self._autoencoder.process(x, y)
        self._softmax.process(x, y)
        self._merge_increment.process(x, y)

    def train_func(self, arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        batch_pool = Pool(self.layers[0].initial_size[0], batch_size)

        # since merge_inc_func requires a pool
        train_func = self._softmax.train_func(arc, learning_rate, prealloc_x, prealloc_y, batch_size, apply_x)
        reconstruction_func = self._autoencoder.validate_func(arc, prealloc_x, prealloc_y, batch_size, apply_x)
        error_func = self.error_func(arc, prealloc_x, prealloc_y, batch_size, apply_x)

        merge_inc_func_batch = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, prealloc_x, prealloc_y)
        merge_inc_func_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._pool.data, self._pool.data_y)
        merge_inc_func_hard_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._hard_pool.data, self._hard_pool.data_y)

        hard_examples_func = self._autoencoder.hard_examples(arc, prealloc_x, prealloc_y, batch_size, apply_x)

        train_func_pool = self._softmax.train_func(arc, learning_rate, self._pool.data, self._pool.data_y, batch_size, apply_x)
        train_func_hard_pool = self._softmax.train_func(arc, learning_rate, self._hard_pool.data, self._hard_pool.data_y, batch_size, apply_x)

        neuron_balance = 1

        def train_pool(pool, pool_func, amount):
            ''' Train using a pool '''
            for i in pool.as_size(int(pool.size * amount), batch_size):
                pool_func(i)

        def moving_average(log, n):
            ''' Calculate moving average of n points '''
            weights = np.exp(np.linspace(-1, 0, n))
            weights /= sum(weights)
            return np.convolve(log, weights)[n-1:-n+1]

        def pool_relevant(pool):
            ''' Find the batches in the pool that have above average similarity to the current batch '''
            current = self.context['distribution'][-1]

            def magnitude(x):
                ''' Magnitude of the vector x '''
                return sum(( v ** 2 for v in x.values() )) ** 0.5

            def compare(x, y):
                ''' Cosine distance of dicts '''
                top = 0

                for k in set(x) | set(y):
                    xval, yval = x[k] if k in x else 0, y[k] if k in y else 0
                    top += xval * yval

                return top / (magnitude(x) * magnitude(y))

            # score over the batches for this pool
            batches_covered = pool.size // batch_size
            batch_scores = [ (i % batches_covered, compare(current, self.context['distribution'][i])) for i in range(-1,-1 - batches_covered,-1) ]
            mean = np.mean([ v[1] for v in batch_scores ])

            # take the last item
            last = [ 0, 0 ]
            for last in itertools.takewhile(lambda s: s[1] > mean, batch_scores):
                pass

            return 1 - (last[0] / batches_covered)

        def train_adaptively(batch_id):
            ''' Wrapper function to add adapting features '''

            # log error
            self._error_log.append(np.asscalar(error_func(batch_id)[0]))

            # get reconstruction error
            self._reconstruction_log.append(np.asscalar(reconstruction_func(batch_id)[0]))

            # calculate exponential moving average
            self._neuron_balance_log.append(neuron_balance)

            # do the pools
            batch_pool.add_from_shared(batch_id, batch_size, prealloc_x, prealloc_y)
            self._pool.add_from_shared(batch_id, batch_size, prealloc_x, prealloc_y)
            self._hard_pool.add(*hard_examples_func(batch_id))

            # collect the data required by the controllers
            data = {
                'mea_30': moving_average(self._error_log, 30),
                'mea_15': moving_average(self._error_log, 15),
                'mea_5': moving_average(self._error_log, 5),
                'pool_relevant': pool_relevant(self._pool),
                'initial_size': self.layers[1].initial_size[0],
                'hard_pool_full': self._hard_pool.size == self._hard_pool.max_size,
                'error_log': self._error_log,
                'errors': self._error_log[-1],
                'neuron_balance': self._neuron_balance_log[-1],
                'reconstruction': self._reconstruction_log[-1],
                'r_15': moving_average(self._reconstruction_log, 15)
            }

            def merge_increment(func, pool, amount, merge, inc):
                ''' Update neuron balance whenever something happens '''
                nonlocal neuron_balance
                change = 1 + inc - merge
                print('neuron balance', neuron_balance, '=>', neuron_balance * change)
                neuron_balance *= change

                func(pool.as_size(int(pool.size * amount), self._mi_batch_size), merge, inc)

            # collect the functions required
            funcs = {
                'merge_increment_batch': functools.partial(merge_increment, merge_inc_func_batch, batch_pool),
                'merge_increment_pool': functools.partial(merge_increment, merge_inc_func_pool, self._pool),
                'merge_increment_hard_pool': functools.partial(merge_increment, merge_inc_func_hard_pool, self._hard_pool),
                'pool': functools.partial(train_pool, self._pool, train_func_pool),
                'hard_pool': functools.partial(train_pool, self._hard_pool, train_func_hard_pool),
                'hard_pool_clear': self._hard_pool.clear,
            }

            # controller move
            self._controller.move(len(self._error_log), data, funcs)

            # train
            train_func(batch_id)

        return train_adaptively

    def validate_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._softmax.validate_func(arc, prealloc_x, prealloc_y, batch_size)

    def error_func(self, arc, prealloc_x, prealloc_y, batch_size, apply_x=identity):
        return self._softmax.error_func(arc, prealloc_x, prealloc_y, batch_size)

    def end(self):
        return [ { 'name': 'neurons_recon.csv',  'csv': ('neurons,reconstruction', list(zip(self._neuron_balance_log, self._reconstruction_log))) } ] + self._controller.end()
