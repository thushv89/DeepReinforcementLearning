

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

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0):
    '''' Load data into shared variables '''
    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh

    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def create_image_from_vector(vec,filename):
    from PIL import Image
    new_vec = vec*255.
    img = Image.fromarray(np.reshape(vec,(28,28),order='C').astype(int),'L')
    img.save(filename +'.png')

def load_from_pickle(filename):

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        create_image_from_vector(train[0][2,:],'test222')

        train = make_shared(train[0], train[1], 'train', False, 1.0)
        valid = make_shared(valid[0], valid[1], 'valid', False, 1.0)
        test  = make_shared(test[0], test[1], 'test', False, 1.0)

        return train, valid, test

def load_from_memmap(filename, row_count, col_count, start_row):

    fp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*start_row,shape=(row_count,col_count))
    data = np.empty((row_count,col_count),dtype=np.float32)
    data[:] = fp[:]
    test_labels = data[:,-1]
    create_image_from_vector(data[2,:-1],'test22')
    train = make_shared(data[:,:-1],data[:,-1],'train',False, 1.0)

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

def make_model(model_type,in_size, hid_sizes, out_size,batch_size, corruption_level, lam, iterations, pool_size):

    rng = T.shared_randomstreams.RandomStreams(0)

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


def train_validate_and_test(batch_size, data_file, epochs, learning_rate, model, modelType, valid_file, test_file, early_stopping):
    distribution = []

    for arc in range(model.arcs):

        get_train_y_func = model.get_y_labels(arc, data_file[0], data_file[1], batch_size)
        get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)

        get_act_vs_pred_func = model.act_vs_pred_func(arc, valid_file[0], valid_file[1], batch_size)
        results_func = model.error_func

        if early_stopping:
            train_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, True, valid_file[0],valid_file[1])
        else:
            train_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, False, None, None)

        validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)
        test_func = results_func(arc, test_file[0], test_file[1], batch_size)

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
                        train_func(t_batch)
                        #print('Greedy costs, Fine tune cost, combined cost: ', greedy_costs, ' ', fine_cost, ' ')
                        #print('Greedy costs, Fine tune cost, combined cost: ', greedy_costs, ' ', fine_cost, ' ')

                    act_vs_pred = get_act_vs_pred_train_func(t_batch)
                    #print('Actual y data for train batch: ',
                    #      format_array_to_print(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval(),
                    #          5)
                    #      , ' ', data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].shape)

                    if (t_batch + 1) % 50 == 0:
                        v_errors = []
                        test_errors = []
                        for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                            validate_results = validate_func(v_batch)
                            act_pred_results = get_act_vs_pred_func(v_batch)

                            print('Actual y data for batch: ',format_array_to_print(valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].eval(),5)
                                  ,' ', valid_file[1][v_batch * batch_size : (v_batch + 1) * batch_size].shape)
                            #print('Data sent to DLModels: ',format_array_to_print(act_pred_results[0],5),' ', act_pred_results[0].shape)
                            print('Predicted data: ', format_array_to_print(act_pred_results[1],5), ' ', act_pred_results[1].shape)
                            v_errors.append(np.asscalar(validate_results))

                        for test_batch in range(math.ceil(test_file[2] / batch_size)):
                            test_results = test_func(test_batch)
                            test_errors.append(np.asscalar(test_results))

                        for i, v_err in enumerate(v_errors):
                            print('batch ',i, ": ", v_err, end=', ')
                        print()
                        print('Mean Validation Error: ', np.mean(v_errors))
                        for i, t_err in enumerate(test_errors):
                            print('batch ',i, ": ", t_err, end=', ')
                        print()
                        print('Mean Test Error: ', np.mean(test_errors))


        except StopIteration:
            pass
    print('done ...')
    return v_errors,test_errors

def get_logger(name, folder_path):
    ''' Create a logger that outputs to `folder_path` '''

    format_string = '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'

    # create the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.formatter = logging.Formatter(format_string)
    logger.addHandler(console)

    # log to file as well
    hardcopy = logging.FileHandler(os.path.join(folder_path, name + '.log'), mode='w')
    hardcopy.setLevel(logging.DEBUG)
    hardcopy.formatter = logging.Formatter(format_string)
    logger.addHandler(hardcopy)

    return logger

def run():

    logger = get_logger('debug','logs')

    learnMode = 'offline'
    learning_rate = 0.25
    batch_size = 100
    epochs = 1
    theano.config.floatX = 'float32'
    modelType = 'SAE'
    valid_logger = get_logger('validation_'+modelType+'_'+learnMode,'logs')
    test_logger = get_logger('test_'+modelType+'_'+learnMode,'logs')
    out_size = 10
    in_size = 784
    hid_sizes = [750,500,500]

    corruption_level = 0.2
    lam = 0.2
    iterations = 50
    pool_size = 10000
    early_stop = True

    model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size)
    input_layer_size = model.layers[0].initial_size[0]

    print('---------- Model Information -------------')
    print('Learning Mode: ',learnMode)
    print('Model type: ', modelType)
    print('Batch size: ', batch_size)
    print('Epochs: ', epochs)

    layers_str = str(in_size) + ', '
    for s in hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(out_size)
    print('Network Configuration: ', layers_str)
    print('Iterations: ', iterations)
    print('Lambda Regularizing Coefficient: ', lam)
    print('Pool Size: ', pool_size)

    print('\nloading data ...')

    if learnMode == 'online':
        _, valid_file, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')

        row_count = 10000
        col_count = in_size + 1
        row_idx = 0
        validation_errors = []
        test_errors  = []

        for i in range(50):
            print('\n------------------------ New Distribution(', i,') --------------------------\n')
            row_idx = i * row_count
            data_file = load_from_memmap('data' + os.sep + 'mnist_non_station.pkl',row_count,col_count,row_idx)
            v_err,test_err = train_validate_and_test(batch_size, data_file, epochs, learning_rate, model, modelType, valid_file, test_file, early_stop)
            validation_errors.append(v_err)
            test_errors.append(test_err)

            valid_logger.info(list(v_err))
            test_logger.info(list(test_err))
    else:
        data_file, valid_file, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')
        v_err,test_err = train_validate_and_test(batch_size, data_file, epochs, learning_rate, model, modelType, valid_file, test_file, early_stop)
        valid_logger.info(list(v_err))
        test_logger.info(list(test_err))

if __name__ == '__main__':
    run()


