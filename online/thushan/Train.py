

__author__ = 'Thushan Ganegedara'

import pickle
import theano
import theano.tensor as T
from online.thushan import DLModels
from online.thushan import NNLayer
from online.thushan import RLPolicies
import os
import math

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh):
    '''' Load data into shared variables '''
    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh

    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_from_pickle(filename):

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        train = make_shared(train[0], train[1], 'train', True, 255.0)
        valid = make_shared(valid[0], valid[1], 'valid', True, 255.0)
        test  = make_shared(test[0], test[1], 'test', True, 255.0)

        return train, valid, test

def make_layers(in_size, hid_sizes, out_size, zero_last = False):
    layers = []
    layers.append(NNLayer.Layer(in_size, hid_sizes[0], False, None, None, None))
    for i, size in enumerate(hid_sizes,0):
        if i==0: continue
        layers.append(NNLayer.Layer(hid_sizes[i-1],hid_sizes[i], False, None, None, None))

    layers.append(NNLayer.Layer(hid_sizes[-1], out_size, False, None, None, None))
    print('Finished Creating Layers')

    return layers

def make_model(in_size, hid_sizes, out_size,batch_size):

    rng = T.shared_randomstreams.RandomStreams(0)

    layers = []
    corruption_level = 0.2
    lam = 0.2
    iterations = 10
    pool_size = 10000
    policy = RLPolicies.ContinuousState()
    layers = make_layers(in_size, hid_sizes, out_size, False)
    model = DLModels.DeepReinforcementLearningModel(
        layers, corruption_level, rng, iterations, lam, batch_size, pool_size, policy)

    model.process(T.matrix('x'), T.ivector('y'))

    return model

def run():

    distribution = []
    learning_rate = 0.1
    batch_size = 1000
    epochs = 500
    theano.config.floatX = 'float32'

    deepRLModel = make_model(784, [750,500,250], 10, batch_size)
    input_layer_size = deepRLModel.layers[0].initial_size[0]

    print('loading data ...')
    data_file, valid_file, _ = load_from_pickle('data' + os.sep + 'mnist.pkl')

    print('pooling data ...')
    # row size (layers[0] initial_size[0] and max size (batch_size)
    batch_pool = DLModels.Pool(deepRLModel.layers[0].initial_size[0], batch_size)
    print('finished pooling ...')
    for arc in range(deepRLModel.arcs):

        results_func = deepRLModel.error_func

        train_func = deepRLModel.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size)
        validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)

        print('training data ...')
        try:
            for epoch in range(epochs):
                print('Training Epoch %d ...' % epoch)
                for batch in range(math.ceil(data_file[2]/batch_size)):
                    print('')
                    print('training epoch %d and batch %d' % (epoch, batch))
                    from collections import Counter
                    dist = Counter(data_file[1][batch * batch_size : (batch + 1) * batch_size].eval())
                    distribution.append({str(k): v/ sum(dist.values()) for k, v in dist.items()})
                    deepRLModel.set_distribution(distribution)

                    train_func(batch)

                for batch in range(math.ceil(valid_file[2]/batch_size)):
                    validate_results = validate_func(batch)
                    print("epoch %d and batch %d Validation error: %f" % (epoch, batch, validate_results))

        except StopIteration:
            pass

    print('done ...')

if __name__ == '__main__':
    run()


