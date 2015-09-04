

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

def make_model(model_type,in_size, hid_sizes, out_size,batch_size):

    rng = T.shared_randomstreams.RandomStreams(0)

    layers = []
    corruption_level = 0.2
    lam = 0.2
    iterations = 3
    pool_size = 10000
    policy = RLPolicies.ContinuousState()
    layers = make_layers(in_size, hid_sizes, out_size, False)
    if model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, corruption_level, rng, iterations, lam, batch_size, pool_size, policy)
    elif model_type == 'SAE':
        model = DLModels.StackedAutoencoderWithSoftmax(
            layers,corruption_level,rng,lam,iterations)
    model.process(T.matrix('x'), T.ivector('y'))

    return model

def format_array_to_print(arr, num_ele=5):
    s = ''
    for i in range(num_ele):
        s += '%.3f' %(arr[i]) + ", "

    s += '\t...\t'
    for i in range(-num_ele,0):
        s += '%.3f' %(arr[i]) + ", "

    return s

def run():

    distribution = []
    learning_rate = 0.1
    batch_size = 1000
    epochs = 500
    theano.config.floatX = 'float32'

    model = make_model('SAE',784, [750,500,250], 10, batch_size)
    input_layer_size = model.layers[0].initial_size[0]

    print('loading data ...')
    data_file, valid_file, _ = load_from_pickle('data' + os.sep + 'mnist.pkl')

    for arc in range(model.arcs):

        get_act_vs_pred_func = model.act_vs_pred_func(arc, valid_file[0], valid_file[1], batch_size)
        results_func = model.error_func
        train_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size)

        validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)

        print('training data ...')
        try:
            for epoch in range(epochs):
                print('Training Epoch %d ...' % epoch)
                for t_batch in range(math.ceil(data_file[2]/batch_size)):
                    print('')
                    print('training epoch %d and batch %d' % (epoch, t_batch))
                    from collections import Counter
                    dist = Counter(data_file[1][t_batch * batch_size : (t_batch + 1) * batch_size].eval())
                    distribution.append({str(k): v/ sum(dist.values()) for k, v in dist.items()})
                    # model.set_distribution(distribution)

                    [greedy_costs, fine_cost, probs] = train_func(t_batch)
                    print('Greedy costs, Fine tune cost, combined cost: ', greedy_costs, ' ', fine_cost, ' ')
                    print(probs)

                    if t_batch%10==0:
                        for v_batch in range(math.ceil(valid_file[2]/batch_size)):
                            validate_results = validate_func(v_batch)
                            act_pred_results = get_act_vs_pred_func(v_batch)
                            print("epoch %d and batch %d" % (epoch, v_batch))
                            print('Actual y data for batch: ',format_array_to_print(valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].eval(),5)
                                  ,' ', valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].shape)
                            print('Data sent to DLModels: ',format_array_to_print(act_pred_results[0],5),' ', act_pred_results[0].shape)
                            print('Predicted data: ', format_array_to_print(act_pred_results[1],5), ' ', act_pred_results[1].shape)
                            print(validate_results)
        except StopIteration:
            pass

    print('done ...')

if __name__ == '__main__':
    run()


