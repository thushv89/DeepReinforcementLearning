

__author__ = 'Thushan Ganegedara'

import pickle
import theano
import theano.tensor as T
import DLModels
import NNLayer
import RLPolicies
import os
import math
import logging
import numpy as np

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

        train = make_shared(train[0], train[1], 'train', True, 1.0)
        valid = make_shared(valid[0], valid[1], 'valid', True, 1.0)
        test  = make_shared(test[0], test[1], 'test', True, 1.0)

        return train, valid, test

def load_from_memmap(filename, row_count, col_count, start_row):

    fp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*start_row,shape=(row_count,col_count))
    data = np.empty((row_count,col_count),dtype=np.float32)
    data[:] = fp[:]

    train = make_shared(data[:,:-1],data[:,-1],'train',True, 1.0)

    return train


def make_layers(in_size, hid_sizes, out_size, zero_last = False):
    layers = []
    layers.append(NNLayer.Layer(in_size, hid_sizes[0], False, None, None, None))
    for i, size in enumerate(hid_sizes,0):
        if i==0: continue
        layers.append(NNLayer.Layer(hid_sizes[i-1],hid_sizes[i], False, None, None, None))

    layers.append(NNLayer.Layer(hid_sizes[-1], out_size, True, None, None, None))
    print('Finished Creating Layers')

    return layers

def make_model(model_type,in_size, hid_sizes, out_size,batch_size):

    rng = T.shared_randomstreams.RandomStreams(0)

    corruption_level = 0.2
    lam = 0.2
    iterations = 1
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


def train_and_validate(batch_size, data_file, epochs, learning_rate, model, modelType, valid_file):
    distribution = []

    for arc in range(model.arcs):

        get_train_y_func = model.get_y_labels(arc, data_file[0], data_file[1], batch_size)
        get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)

        get_act_vs_pred_func = model.act_vs_pred_func(arc, valid_file[0], valid_file[1], batch_size)
        results_func = model.error_func
        train_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size)

        validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)

        print('training data ...')
        try:
            for epoch in range(epochs):
                print('Training Epoch %d ...' % epoch)
                for t_batch in range(math.ceil(data_file[2] / batch_size)):
                    print('')
                    print('training epoch %d and batch %d' % (epoch, t_batch))

                    if modelType == 'DeepRL':
                        from collections import Counter

                        dist = Counter(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval())
                        distribution.append({str(k): v / sum(dist.values()) for k, v in dist.items()})
                        model.set_distribution(distribution)
                        train_func(t_batch)

                    if modelType == 'SAE':
                        [greedy_costs, fine_cost, probs, y_vec] = train_func(t_batch)
                        print('Greedy costs, Fine tune cost, combined cost: ', greedy_costs, ' ', fine_cost, ' ')
                        # print(probs)
                        #for x,out,cost,y_as_vec in zip(probs[0],probs[1],probs[2],y_vec):
                        #    logging.info(list(x))
                        #    logging.info(list(out))
                        #    logging.info(list(y_as_vec))
                        #    logging.info(cost)

                    train_y_labels = get_train_y_func(t_batch)

                    act_vs_pred = get_act_vs_pred_train_func(t_batch)
                    print('Actual y data for train batch: ',
                          format_array_to_print(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval(),
                              5)
                          , ' ', data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].shape)
                    print('Data sent to DLModels train: ', format_array_to_print(train_y_labels, 5), ' ',
                          train_y_labels.shape)
                    print('Predicted data train: ', format_array_to_print(act_vs_pred[1], 5), ' ', act_vs_pred[1].shape)

                    if t_batch % 50 == 0:
                        v_errors = []
                        for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                            validate_results = validate_func(v_batch)
                            act_pred_results = get_act_vs_pred_func(v_batch)

                            # print('Actual y data for batch: ',format_array_to_print(valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].eval(),5)
                            #      ,' ', valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].shape)
                            #print('Data sent to DLModels: ',format_array_to_print(act_pred_results[0],5),' ', act_pred_results[0].shape)
                            #print('Predicted data: ', format_array_to_print(act_pred_results[1],5), ' ', act_pred_results[1].shape)
                            v_errors.append(validate_results)

                        for i, v_err in enumerate(v_errors):
                            print(i, ": ", v_err, end=', ')
                            print()

                        print(np.mean(v_errors))

        except StopIteration:
            pass
    print('done ...')


def run():

    logging.basicConfig(filename="debug.log", level=logging.DEBUG)
    learning_rate = 0.1
    batch_size = 100
    epochs = 500
    theano.config.floatX = 'float32'
    modelType = 'DeepRL'
    out_size = 10
    in_size = 784
    model = make_model(modelType,in_size, [750,500,250], out_size, batch_size)
    input_layer_size = model.layers[0].initial_size[0]

    print('loading data ...')
    _, valid_file, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')

    row_count = 1000
    col_count = in_size + 1
    row_idx = 0
    for i in range(100):
        row_idx = i * row_count
        data_file = load_from_memmap('data' + os.sep + 'mnist_non_station.pkl',row_count,col_count,row_idx)
        train_and_validate(batch_size, data_file, epochs, learning_rate, model, modelType, valid_file)

if __name__ == '__main__':
    run()


