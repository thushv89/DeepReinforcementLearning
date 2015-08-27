from online.thushan import RLPolicies

__author__ = 'Thushan Ganegedara'

import pickle
import theano
import theano.tensor as T
import online.thushan.DLModels as DLModels
import online.thushan.NNLayer as NNLayer
import os

def make_shared(batch_x, batch_y, name):
    '''' Load data into shared variables '''
    x_shared = theano.shared(batch_x, name + '_x_pkl')
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_from_pickle(filename):

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        train = make_shared(train[0], train[1], 'train')
        valid = make_shared(valid[0], valid[1], 'valid')
        test  = make_shared(test[0], test[1], 'test')

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

def make_model():

    rng = T.shared_randomstreams.RandomStreams(0)

    layers = []
    corruption_level = 0.2
    lam = 0.2
    iterations = 1000
    in_size = 100
    hid_sizes = [500,500,500]
    out_size = 10
    batch_size = 1000
    pool_size = 10000
    policy = RLPolicies.ContinuousState()
    layers = make_layers(in_size, hid_sizes, out_size, False)
    model = DLModels.DeepReinforcementLearningModel(layers, corruption_level, rng, lam, iterations, batch_size, pool_size, policy)

    model.process(T.matrix('x'), T.ivector('y'))

    return model

def run():
    learning_rate = 0.1
    batch_size = 100
    epochs = 500
    theano.config.floatX = 'float32'

    deepRLModel = make_model()
    input_layer_size = deepRLModel.layers[0].initial_size[0]

    print('loading data ...')
    data_file, _, _ = load_from_pickle('data' + os.sep + 'mnist.pkl')

    print('pooling data ...')
    batch_pool = DLModels.Pool(deepRLModel.layers[0].initial_size[0], batch_size)

    for arc in range(deepRLModel.arcs):

        results_func = deepRLModel.validate_func

        train_func = deepRLModel.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size)
        validate_func = results_func(arc, data_file[0], data_file[1], batch_size)

        print('training data ...')
        try:
            for epoch in range(epochs):
                for batch in range(batch_size):

                    train_func(batch)
        except StopIteration:
            pass

    print('done ...')

if __name__ == '__main__':
    run()


