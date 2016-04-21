

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

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0,turn_bw=False):
    '''' Load data into shared variables '''
    if turn_bw:
        dims = batch_x.shape[1]
        bw_data = 0.2989*batch_x[:,0:dims/3] + 0.5870 * batch_x[:,dims/3:(2*dims)/3] + 0.1140 * batch_x[:,(dims*2)/3:dims]
        batch_x = bw_data

    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh
    max_val = np.max(x_shared.eval())
    assert 0.004<=max_val<=1.
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

def load_from_memmap(filename, row_count, in_size, start_row,turn_bw):
    if turn_bw:
        col_count = in_size*3 + 1
    else:
        col_count = in_size + 1

    fp = np.memmap(filename,dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*col_count*start_row,shape=(row_count,col_count))
    data = np.empty((row_count,col_count),dtype=np.float32)
    data[:] = fp[:]

    if turn_bw:
        #0.2989 * R + 0.5870 * G + 0.1140 * B
        bw_data = 0.2989*data[:,0:(col_count-1)/3] + 0.5870 * data[:,(col_count-1)/3:(2*(col_count-1))/3] + 0.1140 * data[:,((col_count-1)*2)/3:(col_count-1)]
        data_y = np.asarray([data[:,-1]]).reshape((row_count,1))
        data = np.concatenate((bw_data,data_y),1)

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

def make_model(model_type,in_size, hid_sizes, out_size,batch_size, corruption_level, lam, iterations, pool_size, simi_thresh,rl_logger,rl_state_logger):

    rng = T.shared_randomstreams.RandomStreams(0)

    policy = RLPolicies.ContinuousState(q_logger=rl_logger,state_vis_logger=rl_state_logger)
    layers = make_layers(in_size, hid_sizes, out_size, False)
    if model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, corruption_level, rng, iterations, lam, batch_size, pool_size, policy,simi_thresh)
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

def create_image_from_vector(vec, dataset,turn_bw):
    from pylab import imshow,show,cm
    if dataset == 'mnist':
        imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    elif dataset == 'cifar-10':
        if not turn_bw:
            new_vec = 0.2989 * vec[0:1024] + 0.5870 * vec[1024:2048] + 0.1140 * vec[2048:3072]
        else:
            new_vec = vec
        imshow(np.reshape(new_vec*255,(-1,32)),cmap=cm.gray)
    show()

def train_validate_and_test_v2(batch_size, data_file, next_data_file, learning_rate, model, modelType, valid_file, test_file, time_logger):
    t_distribution = []
    v_distribution = []
    start_time = time.clock()

    for arc in range(model.arcs):
        v_errors = []
        results_func = model.error_func

        if modelType == 'DeepRL':
            train_adaptive = model.train_func(arc, learning_rate, data_file[0], data_file[1], next_data_file[0], next_data_file[1], batch_size)
            get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)
            get_act_vs_pred_test_func = model.act_vs_pred_func(arc, test_file[0], test_file[1], batch_size)
        elif modelType == 'SAE':
            pretrain_func,finetune_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], next_data_file[0], next_data_file[1], batch_size)
            get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)
            get_act_vs_pred_test_func = model.act_vs_pred_func(arc, test_file[0], test_file[1], batch_size)

        if valid_file[0] and valid_file[1]:
            validate_func = results_func(arc, valid_file[0], valid_file[1], batch_size)
        test_func = results_func(arc, test_file[0], test_file[1], batch_size)

        print('training data ...')
        try:
            if modelType == 'SAE':
                print('Pre-training ...')
                for t_batch in range(math.ceil(data_file[2] / batch_size)):
                    pretrain_func(t_batch)

                print('Fine tuning ...')
                for t_batch in range(math.ceil(data_file[2] / batch_size)):
                    v_errors.append(finetune_func(t_batch))

            elif modelType == 'DeepRL':

                from collections import Counter

                fine_tune_costs = []
                for t_batch in range(math.ceil(data_file[2] / batch_size)):
                    train_batch_start_time = time.clock()

                    t_dist = Counter(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval())
                    t_distribution.append({str(k): v / sum(t_dist.values()) for k, v in t_dist.items()})
                    model.set_train_distribution(t_distribution)
                    print('Train batch: ', t_batch, ' Distribution: ', t_dist)

                    v_errors.append(train_adaptive(t_batch))

                    train_batch_stop_time = time.clock()
                    print('\nTime for train batch ', t_batch, ': ', (train_batch_stop_time-train_batch_start_time), ' (secs)')

            test_errors = []
            for test_batch in range(math.ceil(test_file[2] / batch_size)):
                test_results = test_func(test_batch)
                test_errors.append(np.asscalar(test_results))


        except StopIteration:
            pass

    end_time = time.clock()
    print('\nTime taken for the data stream: ', (end_time-start_time)/60, ' (mins)')
    time_logger.info((end_time-start_time))
    return np.mean(v_errors),test_errors

def train_validate_mergeinc(batch_size, pool_size, data_file, next_data_file, learning_rate, model, modelType, valid_file, test_file,time_logger):
    start_time = time.clock()


    for arc in range(model.arcs):

        train_mergeinc = model.train_func(arc, learning_rate, data_file[0], data_file[1], next_data_file[0], next_data_file[1], batch_size)

        results_func = model.error_func
        test_func = results_func(arc, test_file[0], test_file[1], batch_size)

        improvement_threshold = 0.995

        v_errors = []
        for t_batch in range(math.ceil(data_file[2] / batch_size)):
            curr_train_err = train_mergeinc(t_batch)
            v_errors.append(curr_train_err)

            if (not v_errors) or len(v_errors)==0:
                pass

        test_errors = []

        print('Testing phase ...\n')
        for test_batch in range(math.ceil(test_file[2] / batch_size)):
            test_results = test_func(test_batch)
            test_errors.append(np.asscalar(test_results))

    end_time = time.clock()
    print('\nTime taken for the data stream: ', (end_time-start_time)/60, ' (mins)')
    time_logger.info(end_time-start_time)
    return np.mean(v_errors),test_errors


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
    log_suffix = ''
    import getopt
    import sys
    try:
        opts,args = getopt.getopt(sys.argv[1:],"",["suffix="])
    except getopt.GetoptError:
        print('<filename>.py --suffix <log file prefix>')
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--suffix':
                log_suffix = arg

    #####################################################################

    dataset = 'cifar-10' #mnist, mnist-var, cifar-10,cifar-100,svhn,cifar-10-bin
    dataset_type = 'non_station' # station or non_station
    iterations = 10

    in_size = 3072 #cifar-10:3072, mnist:784, mnist-var:784
    if dataset == 'cifar-10-bin':
        out_size = 2
    else:
        out_size = 10

    hid_sizes = [1000,1000,1000]

    modelType = 'DeepRL'
    simi_thresh = 0.8 #choose between 0.7 (non-station) or 0.995 (station)

    pool_size = 10000

    #######################################################################
    learnMode = 'online'
    learning_rate = 0.2
    batch_size = 1000
    epochs = 1
    theano.config.floatX = 'float32'

    turn_bw = False # turn images black and white (for cifar-10 & cifar-100)
    corruption_level = 0.2
    lam = 0.1

    if not dataset=='cifar-10-bin':
        online_total_rows = 1000000
    else:
        online_total_rows = 200000
    online_train_row_count = batch_size

    layers_str = str(in_size) + ', '
    for s in hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(out_size)

    valid_logger = get_logger('validation_'+modelType+'_'+learnMode+'_'+dataset +'_'+dataset_type + '_' + layers_str+log_suffix,'logs')
    test_logger = get_logger('test_'+modelType+'_'+learnMode+'_'+dataset +'_'+dataset_type + '_' + layers_str+log_suffix,'logs')
    reconstruction_err_logger = get_logger('reconstruction_error_'+modelType+'_'+learnMode+'_'+dataset + '_' + dataset_type + '_' + layers_str+log_suffix,'logs')
    rl_logger = get_logger('policy_'+modelType+'_'+learnMode+'_'+dataset + '_' + dataset_type + '_' + layers_str+log_suffix,'logs')
    state_vis_logger = get_logger('state_vis'+modelType+'_'+learnMode+'_'+dataset + '_' + dataset_type + '_' + layers_str+log_suffix,'logs')
    time_logger = get_logger('time'+modelType+'_'+learnMode+'_'+dataset + '_' + dataset_type + '_' + layers_str+log_suffix,'logs')
    network_size_logger, error_logger = None, None

    if modelType == 'DeepRL' or modelType == 'MergeInc':
        network_size_logger = get_logger('network_size_'+modelType+'_'+learnMode+'_'+dataset +'_'+dataset_type + '_' + layers_str+log_suffix,'logs')
    model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size,simi_thresh,rl_logger=rl_logger,rl_state_logger=state_vis_logger)


    model_info = '---------- Model Information -------------\n'
    model_info += 'Dataset: ' + dataset + '\n'
    model_info += 'Dataset type: ' + dataset_type + '\n'
    model_info += 'Learning Mode: ' + learnMode + '\n'
    if learnMode == 'online':
        model_info += 'Total Examples: ' + str(online_total_rows) + '\n'
    model_info += 'Model type: ' + modelType + '\n'
    model_info += 'Batch size: ' + str(batch_size) + '\n'
    model_info += 'Epochs: ' + str(epochs) + '\n'

    model_info += 'Network Configuration: ' + layers_str + '\n'
    model_info += 'Iterations: ' + str(iterations) + '\n'
    model_info += 'Lambda Regularizing Coefficient: ' + str(lam) + '\n'
    model_info += 'Pool Size (Train): ' + str(pool_size) + '\n'
    model_info += 'Simi Thresh (For DeepRL): ' + str(simi_thresh) + '\n'

    print(model_info)
    valid_logger.info(model_info)
    test_logger.info(model_info)
    print('\nloading data ...')

    if learnMode == 'online':

        if dataset == 'mnist':
            _, _, test_file = load_from_pickle('data' + os.sep + 'mnist.pkl')
        elif dataset == 'mnist-var':
            f = open('data'+os.sep+'mnist_rot_back_test.pkl','rb')
            test_x,test_y = pickle.load(f,encoding='latin1')
            test_file = make_shared(test_x,test_y,'test',False,1.0,turn_bw)
        elif dataset == 'cifar-10':
            f = open('data' + os.sep + 'cifar_10_test_batch', 'rb')
            dict = pickle.load(f,encoding='latin1')
            test_file = make_shared(np.asarray(dict.get('data'), dtype=np.float32), np.asarray(dict.get('labels'), dtype=np.float32), 'test', True, 255.0, turn_bw)
        elif dataset == 'cifar-100':
            f = open('data' + os.sep + 'cifar_100_test_batch', 'rb')
            dict = pickle.load(f,encoding='latin1')
            test_file = make_shared(np.asarray(dict.get('data'), dtype=np.float32), np.asarray(dict.get('fine_labels'), dtype=np.float32), 'test', True, 255.0, turn_bw)
        elif dataset == 'svhn':
            import scipy.io as sio
            testdata = sio.loadmat('data' + os.sep +'svhn_test_32x32.mat')
            test_x = testdata['X']
            test_y = [ele[0] for ele in testdata['y']]
            res_test_x  = np.swapaxes(test_x,0,1).T.reshape((-1,3072),order='C')
            test_file = make_shared(np.asarray(res_test_x, dtype=np.float32), np.asarray(test_y, dtype=np.float32), 'test', True, 255.0, turn_bw)
        elif dataset == 'cifar-10-bin':
            f = open('data' + os.sep + 'cifar_10_test_batch', 'rb')
            dict = pickle.load(f,encoding='latin1')
            labels = np.asarray(dict.get('labels'),dtype=np.int8)
            data = dict.get('data')

            labels_0 = np.where(labels==0)[0]
            labels_1 = np.where(labels==1)[0]
            labels_01 = np.append(labels_0,labels_1)

            test_file = make_shared(np.asarray(data, dtype=np.float32)[labels_01,:],np.asarray(labels,dtype=np.float32)[labels_01],'test',True, 255.0, turn_bw)

        validation_errors = []
        mean_test_errors  = []
        curr_data_file = None

        for i in range(int(online_total_rows/online_train_row_count)):
            v_err,test_err = None,None
            print('\n------------------------ New Distribution(', i,') --------------------------\n')
            if dataset == 'mnist':
                next_data_file = load_from_memmap('data' + os.sep + 'mnist_'+dataset_type+'_1000000.pkl',online_train_row_count,in_size,i * online_train_row_count,False)
                valid_file = [None,None]
            elif dataset == 'mnist-var':
                next_data_file = load_from_memmap('data'+os.sep + 'mnist_rot_back_'+dataset_type+'_1000000.pkl',online_train_row_count,in_size,i*online_train_row_count,False)
                valid_file = [None,None]
            elif dataset == 'cifar-10':
                next_data_file = load_from_memmap('data' + os.sep + 'cifar_10_'+dataset_type+'_1000000.pkl',online_train_row_count,in_size,i * online_train_row_count,turn_bw)
                valid_file = [None,None]
            elif dataset == 'cifar-100':
                next_data_file = load_from_memmap('data' + os.sep + 'cifar_100_'+dataset_type+'_1000000.pkl',online_train_row_count,in_size,i * online_train_row_count,turn_bw)
                valid_file = [None,None]
            elif dataset == 'svhn':
                next_data_file = load_from_memmap('data' + os.sep + 'svhn_non_station_1000000.pkl',online_train_row_count,in_size,i * online_train_row_count,turn_bw)
                valid_file = [None,None]
            elif dataset == 'cifar-10-bin':
                if dataset_type == 'non_station':
                    next_data_file = load_from_memmap('data' + os.sep + 'cifar_10_'+dataset_type+'_200000_gauss_bin.pkl',online_train_row_count,in_size,i*online_train_row_count,turn_bw)
                elif dataset_type == 'station':
                    next_data_file = load_from_memmap('data' + os.sep + 'cifar_10_'+dataset_type+'_200000_uni_bin.pkl',online_train_row_count,in_size,i*online_train_row_count,turn_bw)

                valid_file = [None,None]
            else:
                raise NotImplementedError

            if curr_data_file and not modelType == 'MergeInc':
                v_err,test_err = train_validate_and_test_v2(batch_size, curr_data_file, next_data_file, learning_rate, model, modelType, valid_file, test_file,time_logger)
            elif curr_data_file and modelType == 'MergeInc':
                v_err,test_err = train_validate_mergeinc(batch_size, pool_size, curr_data_file, next_data_file, learning_rate, model, modelType, valid_file, test_file,time_logger)

            if v_err and test_err:
                validation_errors.append(v_err)
                mean_test_errors.append(np.mean(test_err))

                print('Validation Error: ',v_err)
                for t_idx, err in enumerate(test_err):
                    print('batch ',t_idx, ": ", err, end=', ')
                print()
                print('Mean Test Error: ', np.mean(test_err),'\n')


            if (i+1) % 100 == 0:
                valid_logger.info(validation_errors)
                test_logger.info(list(mean_test_errors))
                reconstruction_err_logger.info(list(np.asarray(model._reconstruction_log)[:]))
                if network_size_logger:
                    network_size_logger.info(list(np.asarray(model._network_size_log)))

            curr_data_file = next_data_file

        if modelType == 'DeepRL' or modelType == 'MergeInc':
            err_log_arr = np.asarray(model._error_log).reshape(-1,len(model._error_log))


        rec_log_arr = np.asarray(model._reconstruction_log)
        reconstruction_err_logger.info(list(rec_log_arr[:]))
        valid_logger.info(validation_errors)
        test_logger.info(list(mean_test_errors))
    else:

        prev_train_err = np.inf
        inc = 0.

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



        if not modelType == 'MergeInc':
            v_err,test_err = train_validate_and_test_v2(batch_size, data_file, learning_rate, model, modelType, valid_file, test_file)
        else:
            v_err,test_err,prev_train_err,inc = train_validate_mergeinc(batch_size, data_file, learning_rate, model, modelType, valid_file, test_file, prev_train_err, inc)

        valid_logger.info(list(v_err))
        test_logger.info(list(test_err),', mean:', np.mean(test_err))

if __name__ == '__main__':
    run()


