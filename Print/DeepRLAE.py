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
