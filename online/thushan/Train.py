

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
import time

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0):
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

        train = make_shared(train[0], train[1], 'train', False, 1.0)
        valid = make_shared(valid[0], valid[1], 'valid', False, 1.0)
        test  = make_shared(test[0], test[1], 'test', False, 1.0)

        return train, valid, test

def load_from_memmap(filename, row_count, col_count, start_row):

    fp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*start_row,shape=(row_count,col_count))
    data = np.empty((row_count,col_count),dtype=np.float32)
    data[:] = fp[:]
    test_labesl = data[:,-1]
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

def make_model(model_type,in_size, hid_sizes, out_size,batch_size, corruption_level, lam, iterations, pool_size, valid_pool_size):

    rng = T.shared_randomstreams.RandomStreams(0)

    policy = RLPolicies.ContinuousState()
    layers = make_layers(in_size, hid_sizes, out_size, False)
    if model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, corruption_level, rng, iterations, lam, batch_size, pool_size, valid_pool_size, policy)
    elif model_type == 'SAE':
        model = DLModels.StackedAutoencoderWithSoftmax(
            layers,corruption_level,rng,lam,iterations)
    elif model_type == 'MergeInc':
        model = DLModels.MergeIncDAE(layers, corruption_level, rng, iterations, lam, batch_size, pool_size)

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


def train_validate_and_test_v2(batch_size, pool_size, data_file, pre_epochs, fine_epochs, learning_rate, model, modelType, valid_file, test_file, early_stop=True, network_size_logger = None, rec_err_logger = None, err_logger = None):
    t_distribution = []
    v_distribution = []
    start_time = time.clock()

    for arc in range(model.arcs):

        results_func = model.error_func

        if modelType == 'DeepRL':
            train_adaptive,finetune_adaptive, finetune_valid_adaptive = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, True, valid_file[0],valid_file[1])
            get_act_vs_pred_valid_func = model.act_vs_pred_func(arc, valid_file[0], valid_file[1], batch_size)
            get_act_vs_pred_test_func = model.act_vs_pred_func(arc, test_file[0], test_file[1], batch_size)
        elif modelType == 'SAE':
            pretrain_func,finetune_func,finetune_valid_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, True, valid_file[0],valid_file[1])
            get_act_vs_pred_valid_func = model.act_vs_pred_func(arc, valid_file[0], valid_file[1], batch_size)
            get_act_vs_pred_test_func = model.act_vs_pred_func(arc, test_file[0], test_file[1], batch_size)
        elif modelType == 'MergeInc':
            train_mergeinc = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, False, None, None)

        validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)
        test_func = results_func(arc, test_file[0], test_file[1], batch_size)

        print('training data ...')
        try:
            if modelType == 'SAE':
                for epoch in range(pre_epochs):
                    print('Training Epoch %d ...' % epoch)
                    for t_batch in range(math.ceil(data_file[2] / batch_size)):
                        t_cost = pretrain_func(t_batch)
                        print('training epoch %d and batch %d and cost %f' % (epoch, t_batch,t_cost))

            if modelType == 'SAE' and early_stop:
                #########################################################################
                #####                         Early-Stopping                        #####
                #########################################################################
                n_train_batches = math.ceil(data_file[2] / batch_size)
                patience = 10 * n_train_batches # look at this many examples
                patience_increase = 2.
                improvement_threshold = 0.995
                #validation frequency - the number of minibatches to go through before checking validation set
                validation_freq = min(n_train_batches,patience/2)

                #we want to minimize best_valid_loss, so we shoudl start with largest
                best_valid_loss = np.inf
                test_score = 0.

                done_looping = False

                f_epoch = 0
                while f_epoch < fine_epochs and (not done_looping):
                    f_epoch += 1
                    fine_tune_costs = []
                    for t_batch in range(n_train_batches):
                        from collections import Counter

                        t_dist = Counter(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval())
                        print('Train batch: ', t_batch, ' Distribution: ', t_dist)

                        cost = finetune_func(t_batch)
                        fine_tune_costs.append(cost)
                        #what's the role of iter? iter acts as follows
                        #in first epoch, iter for minibatch 'x' is x
                        #in second epoch, iter for minibatch 'x' is n_train_batches + x
                        #iter is the number of minibatches processed so far...
                        f_iter = (f_epoch-1) * n_train_batches + t_batch

                        # this is an operation done in cycles. 1 cycle is iter+1/validation_freq
                        # doing this every epoch
                        if (f_iter+1) % validation_freq == 0:
                            n_valid_batches =  math.ceil(valid_file[2] / batch_size)
                            valid_errs = []
                            for v_batch in range(n_valid_batches):
                                valid_errs.append(np.asscalar(finetune_valid_func(v_batch)))

                            curr_valid_loss = np.mean(valid_errs)

                            if curr_valid_loss < best_valid_loss:

                                if (curr_valid_loss < best_valid_loss * improvement_threshold):
                                    patience = max(patience, f_iter * patience_increase)
                                    print('Patience: ',patience)

                                tmp_v_errs = []
                                for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                                    tmp_v_errs.append(np.asscalar(validate_func(v_batch)))
                                print('Mean Validation Error: ', np.mean(tmp_v_errs))

                                best_valid_loss = curr_valid_loss
                            print('Iter: ', f_iter, 'Curr valid: ', curr_valid_loss, ', Best valid: ', best_valid_loss)

                    #patience is here to check the maximum number of iterations it should check
                    #before terminating
                    if patience <= f_iter:
                        print('Early stopping at iter: ', f_iter)
                        done_looping = True
                        break
            elif modelType=='SAE' and not early_stop:
                for f_epoch in range(fine_epochs):
                    finetune_func(t_batch)

            elif modelType == 'DeepRL' and early_stop:

                all_empty_slots = set()
                last_empty_slots = []
                from collections import Counter

                for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                    v_dist = Counter(valid_file[1][v_batch * batch_size: (v_batch + 1) * batch_size].eval())
                    v_distribution.append({str(k): v / sum(v_dist.values()) for k, v in v_dist.items()})
                    model.set_valid_distribution(v_distribution)

                n_train_batches = math.ceil(data_file[2] / batch_size)
                patience = 10 * n_train_batches # look at this many examples
                patience_increase = 2.
                improvement_threshold = 0.995
                validation_freq = min(n_train_batches,patience/2)

                best_valid_loss = np.inf

                f_epoch = 0
                while f_epoch < fine_epochs:
                    epoch_start_time = time.clock()
                    print ('\n Fine Epoch: ', f_epoch)
                    fine_tune_costs = []
                    for t_batch in range(math.ceil(data_file[2] / batch_size)):
                        train_batch_start_time = time.clock()
                        f_epoch += 1
                        f_iter = (f_epoch -1 ) * n_train_batches + t_batch
                        t_dist = Counter(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval())
                        t_distribution.append({str(k): v / sum(t_dist.values()) for k, v in t_dist.items()})
                        model.set_train_distribution(t_distribution)
                        print('Train batch: ', t_batch, ' Distribution: ', t_dist)
                        empty_slots = train_adaptive(t_batch)

                        if empty_slots:
                            last_empty_slots = empty_slots
                            print('Fine tuning for specific neurons ...')
                            all_empty_slots.update(empty_slots)

                        fine_tune_costs.append(finetune_adaptive(empty_slots))

                        if (f_iter+1) % validation_freq == 0:
                            print('\nEarly stopping validation (',f_iter,')')
                            n_valid_batches =  math.ceil(valid_file[2] /batch_size)
                            #v_batch_idx = np.random.uniform(low = 0, high = n_valid_batches-1, size=10)
                            valid_errs = []
                            for v_batch in range(n_valid_batches):
                                valid_errs.append(np.asscalar(finetune_valid_adaptive(int(v_batch))))
                                print('v_batch: ', v_batch, ' valid errs: ', valid_errs)
                            curr_valid_loss = np.mean(valid_errs)

                            print('Iter: ', f_iter, 'Curr valid: ', curr_valid_loss, ', Best valid: ', best_valid_loss)
                            if curr_valid_loss < best_valid_loss:

                                if curr_valid_loss < best_valid_loss * improvement_threshold:
                                    patience = max(patience, f_iter * patience_increase)
                                    print('Patience: ',patience)

                                tmp_v_errs = []
                                for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                                    tmp_v_errs.append(np.asscalar(validate_func(v_batch)))
                                print('Mean Validation Error: ', np.mean(tmp_v_errs))

                                best_valid_loss = curr_valid_loss

                        train_batch_stop_time = time.clock()
                        print('\nTime for train batch ', t_batch, ': ', (train_batch_stop_time-train_batch_start_time), ' (secs)')
                    #patience is here to check the maximum number of iterations it should check
                    #before terminating
                    if patience <= f_iter:
                        print('\nEarly stopping at iter: ', f_iter)
                        break

                    epoch_stop_time = time.clock()
                    print('Time for epoch ',f_epoch, ': ',(epoch_stop_time-epoch_start_time)/60,' (mins)')

                # fine tune all empty slots. thi function uses the pool to fintune
                refined_empty_slots = [i for i in list(all_empty_slots) if i<=np.max(last_empty_slots)]
                #if refined_empty_slots and not len(refined_empty_slots)==0:
                #    print('Finetuning all empty slots for this stream')
                #    finetune_adaptive(refined_empty_slots)

                network_size_logger.info(model._network_size_log)
                model._network_size_log = []
                rec_err_logger.info(model._reconstruction_log)
                err_logger.info(model._error_log)

            elif modelType == 'MergeInc':
                prev_train_err = np.inf
                inc = 0.1
                improvement_threshold = 0.995
                t_costs = []
                for t_batch in range(math.ceil(data_file[2] / batch_size)):
                    curr_train_err = train_mergeinc(t_batch, inc, inc*.5)
                    t_costs.append(curr_train_err)
                    print('Train batch: ', t_batch, ' Train cost: ', curr_train_err)
                    # when hard_pool is full ...
                    if ((t_batch+1)*batch_size) % (pool_size*2) == 0:
                        print('Hard pool is full...')
                        mean_train_error = np.mean(t_costs)
                        if mean_train_error > prev_train_err * (1 + (1-improvement_threshold)):
                            inc = 1 - prev_train_err/mean_train_error
                        else:
                            inc = 0.

                        print(modelType, ' inc: ', inc, ' merge: ', inc*0.5)
                        print('Prev TrainErr: ', prev_train_err, ' Mean TrainErr: ', mean_train_error)
                        prev_train_err = mean_train_error
                        t_costs = []


            v_errors = []
            test_errors = []
            print('\nValidation phase ...\n')
            for v_batch in range(math.ceil(valid_file[2] / batch_size)):
                validate_results = validate_func(v_batch)
                v_errors.append(np.asscalar(validate_results))
                if modelType == 'DeepRL' or modelType=='SAE':
                    act_vs_pred_valid = get_act_vs_pred_valid_func(v_batch)
                    print('Actual: ', act_vs_pred_valid[0])
                    print('Predicted: ', act_vs_pred_valid[1])
            print('\nTesting phase ...\n')
            for test_batch in range(math.ceil(test_file[2] / batch_size)):
                test_results = test_func(test_batch)
                test_errors.append(np.asscalar(test_results))
                if modelType == 'DeepRL' or modelType=='SAE':
                    act_vs_pred_test = get_act_vs_pred_test_func(test_batch)
                    print('Actual: ', act_vs_pred_test[0])
                    print('Predicted: ', act_vs_pred_test[1])
            for i, v_err in enumerate(v_errors):
                print('batch ',i, ": ", v_err, end=', ')
            print()
            print('Mean Validation Error: ', np.mean(v_errors),'\n')
            for i, t_err in enumerate(test_errors):
                print('batch ',i, ": ", t_err, end=', ')
            print()
            print('Mean Test Error: ', np.mean(test_errors),'\n')


        except StopIteration:
            pass
    end_time = time.clock()
    print('\nTime taken for the data stream: ', (end_time-start_time)/60, ' (mins)')
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

    dataset = 'cifar-10'
    in_size = 3072
    out_size = 10

    learnMode = 'online'
    modelType = 'DeepRL'

    learning_rate = 0.25
    batch_size = 1000
    epochs = 1
    theano.config.floatX = 'float32'

    hid_sizes = [500,500,500]

    corruption_level = 0.2
    lam = 0.1
    iterations = 8
    pool_size = 10000
    valid_pool_size = pool_size/2
    early_stop = True

    pre_epochs = 5
    finetune_epochs = 10

    valid_logger = get_logger('validation_'+modelType+'_'+learnMode+'_'+dataset,'logs')
    test_logger = get_logger('test_'+modelType+'_'+learnMode+'_'+dataset,'logs')

    network_size_logger, reconstruction_err_logger, error_logger = None, None, None

    if modelType == 'DeepRL':
        network_size_logger = get_logger('network_size_'+modelType+'_'+learnMode+'_'+dataset,'logs')
        reconstruction_err_logger = get_logger('reconstruction_error_'+modelType+'_'+learnMode+'_'+dataset,'logs')
        error_logger = get_logger('error_'+modelType+'_'+learnMode+'_'+dataset,'logs')
    model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size, valid_pool_size)
    input_layer_size = model.layers[0].initial_size[0]


    model_info = '---------- Model Information -------------\n'
    model_info += 'Learning Mode: ' + learnMode + '\n'
    model_info += 'Model type: ' + modelType + '\n'
    model_info += 'Batch size: ' + str(batch_size) + '\n'
    model_info += 'Epochs: ' + str(epochs) + '\n'

    layers_str = str(in_size) + ', '
    for s in hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(out_size)
    model_info += 'Network Configuration: ' + layers_str + '\n'
    model_info += 'Iterations: ' + str(iterations) + '\n'
    model_info += 'Lambda Regularizing Coefficient: ' + str(lam) + '\n'
    model_info += 'Pool Size (Train): ' + str(pool_size) + '\n'
    model_info += 'Pool Size (Valid): ' + str(valid_pool_size) + '\n'

    print(model_info)
    valid_logger.info(model_info)
    test_logger.info(model_info)
    print('\nloading data ...')

    if learnMode == 'online':
        if dataset == 'mnist':
            _, _, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')
        elif dataset == 'cifar-10':
            f = open('data' + os.sep + 'cifar_10_test_batch', 'rb')
            dict = pickle.load(f,encoding='latin1')
            test_file = make_shared(np.asarray(dict.get('data'), dtype=np.float32), np.asarray(dict.get('labels'), dtype=np.float32), 'test', True, 255.0)

        train_row_count = 20000
        valid_row_count = 4000
        col_count = in_size + 1
        validation_errors = []
        test_errors  = []


        for i in range(int(500000/train_row_count)):
            print('\n------------------------ New Distribution(', i,') --------------------------\n')
            if dataset == 'mnist':
                data_file = load_from_memmap('data' + os.sep + 'mnist_non_station.pkl',train_row_count,col_count,i * train_row_count)
                valid_file = load_from_memmap('data' + os.sep + 'mnist_validation_non_station.pkl',valid_row_count,col_count,i * valid_row_count)
            elif dataset == 'cifar-10':
                data_file = load_from_memmap('data' + os.sep + 'cifar_10_non_station.pkl',train_row_count,col_count,i * train_row_count)
                valid_file = load_from_memmap('data' + os.sep + 'cifar_10_validation_non_station.pkl',valid_row_count,col_count,i * valid_row_count)

            v_err,test_err = train_validate_and_test_v2(batch_size, pool_size, data_file, pre_epochs, finetune_epochs, learning_rate, model, modelType, valid_file, test_file, early_stop, network_size_logger,reconstruction_err_logger,error_logger)
            validation_errors.append(v_err)
            test_errors.append(test_err)

            valid_logger.info(list(v_err))
            test_logger.info(list(test_err))
    else:
        if dataset == 'mnist':
            data_file, valid_file, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')
        elif dataset == 'cifar-10':
            train_names = ['cifar_10_data_batch_1','cifar_10_data_batch_2','cifar_10_data_batch_3','cifar_10_data_batch_4']
            valid_name = 'cifar_10_data_batch_5'
            test_name = 'cifar_10_test_batch'

            data_x = []
            data_y = []
            for file_path in train_names:
                f = open('data' + os.sep +file_path, 'rb')
                dict = pickle.load(f,encoding='latin1')
                data_x.extend(dict.get('data'))
                data_y.extend(dict.get('labels'))

            data_file = make_shared(np.asarray(data_x,dtype=theano.config.floatX),np.asarray(data_y,theano.config.floatX),'train',True, 255.)

            f = open('data' + os.sep +valid_name, 'rb')
            dict = pickle.load(f,encoding='latin1')
            valid_file = make_shared(np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX),'valid',True, 255.)

            f = open('data' + os.sep +test_name, 'rb')
            dict = pickle.load(f,encoding='latin1')
            test_file = make_shared(np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX),'test',True, 255.)

            f.close()

        v_err,test_err = train_validate_and_test_v2(batch_size, pool_size, data_file, pre_epochs, finetune_epochs, learning_rate, model, modelType, valid_file, test_file, early_stop, network_size_logger,reconstruction_err_logger,error_logger)
        valid_logger.info(list(v_err))
        test_logger.info(list(test_err))

if __name__ == '__main__':
    run()


