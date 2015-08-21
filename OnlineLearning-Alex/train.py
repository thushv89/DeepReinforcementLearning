#!/usr/bin/env python3
""" Trains neural networks """

import math
import os, time
import itertools, functools
import logging

import numpy as np

import theano
import theano.tensor as T

import models
import common
import nnet_layer
import policies

# output logging data to the data folder
OUTPUT_LOCATION = 'output/'

LAYERS_FOLDER = 'layers/'
FILTERS_FOLDER = 'filters/'

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

def make_shared(batch_x, batch_y, name):
    '''' Load data into shared variables '''
    x_shared = theano.shared(batch_x, name + '_x_pkl')
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_from_file(input_layer_size, filename):
    ''' Load examples from the file, load into shared variables '''
    with open(filename, 'rb') as handle:
        batch_x, batch_y = common.from_bytes(input_layer_size, handle.read())

    return make_shared(batch_x, batch_y, 'file')

def load_from_pickle(filename):
    ''' Load data to shared memory from a pickle file '''
    import pickle

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        train = make_shared(train[0], train[1], 'train')
        valid = make_shared(valid[0], train[1], 'valid')
        test  = make_shared(test[0], test[1], 'test')

        return train, valid, test

def make_layers(layer_files, additional_layers, zero_last):
    ''' Create layers from layer files '''
    layers = [ nnet_layer.Layer.from_npz(filename) for filename in layer_files ]

    if layers:
        additional_layers = [ layers[-1].initial_size[1] ] + additional_layers

    for i, size in enumerate(zip(additional_layers, additional_layers[1:])):
        zero = zero_last and i == len(additional_layers) - 1
        layers.append(nnet_layer.Layer(size[0], size[1], zero))

    return layers

def make_model(args):
    ''' Create the neural network model as required '''
    model = args.model
    theano_rng = theano.tensor.shared_randomstreams.RandomStreams(0)
    rng = None if args.marginalised else theano_rng

    if model == 'autoencoder':
        # pure autoencoder model, typically used for pretraining
        layers = make_layers(args.layer_files, args.additional_layers, False)
        model = models.StackedAutoencoder(layers, args.corruption_level, rng)
    elif model == 'deep_autoencoder':
        # autoencoder, except trains multiple layers at a time
        layers = make_layers(args.layer_files, args.additional_layers, False)
        model = models.DeepAutoencoder(layers, args.corruption_level, rng)
    elif model == 'softmax':
        # pure multilayer perceptron model with softmax layer on bottom
        layers = make_layers(args.layer_files, args.additional_layers, True)
        model = models.Softmax(layers, args.iterations)
    elif model == 'adapting':
        # create the policy
        policy = getattr(policies, args.policy)

        # self adapting with a policy
        layers = make_layers(args.layer_files, args.additional_layers, True)
        model = models.AdaptingCombinedObjective(layers, args.corruption_level, rng, args.iterations, args.lam, args.mi_batch_size, args.pool_size, policy())
    elif model == 'combined':
        # pure multilayer perceptron model with softmax layer on bottom
        layers = make_layers(args.layer_files, args.additional_layers, True)
        model = models.CombinedObjective(layers, args.corruption_level, rng, args.lam, args.iterations)
    else:
        raise ValueError(args.model + ' is not a known model')

    # initialise the model
    model.process(T.matrix('x'), T.ivector('y'))

    return model

def get_arguments():
    ''' Parse arguments '''
    import argparse

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default=4, help='Iterations per batch')
    parser.add_argument('-r', '--learning-rate', dest='learning_rate', type=float, default=0.1, help='Learning rate')

    parser.add_argument('-m', '--model', dest='model', type=str, required=True, help='Which model to use')

    architecture_group = parser.add_argument_group('Architecture')
    architecture_group.add_argument('-lf', '--layer-files', nargs='+', dest='layer_files', type=str, default=[], help='Create the layers in front using these files')
    architecture_group.add_argument('-al', '--additional-layers', nargs='+', dest='additional_layers', type=int, default=[], help='Add new empty layers with the following sizes')

    input_group = parser.add_argument_group('Data input')
    input_group.add_argument('-if', '--data-file', dest='data_file', type=str, help='Specifies the file used to load input data')
    input_group.add_argument('-pkf', '--pickle-file', dest='pickle_file', type=str, help='Get training, validation and test from pkl')

    output_group = parser.add_argument_group('Data output')
    output_group.add_argument('-o', '--output', dest='output_folder', type=str, default=str(time.time()), help='Name of output folder')

    training_group = parser.add_argument_group('Training')
    training_group.add_argument('-e', '--epoches', dest='epoches', type=int, default=1, help='Number of training epoches, only usable if a data file was specified')

    method_specific_group = parser.add_argument_group('Method specific options, some may be ignored depending on the model specified')
    method_specific_group.add_argument('-ma', '--marginalised', dest='marginalised', action='store_true', help='Affects: autoencoders. Use marginalised autoencoders')
    method_specific_group.add_argument('-mc', '--corruption-level', dest='corruption_level', type=float, default=0.2, help='Affects: autoencoder. Corruption level to use')
    method_specific_group.add_argument('-mt', '--theirs', dest='theirs', action='store_true', help='Affects: adapting, use incremental autoencoder')
    method_specific_group.add_argument('-ml', '--lambda', dest='lam', type=float, default=0.2, help='Affects: combiend objective')
    method_specific_group.add_argument('-mebs', '--merge-increment-batch-size', dest='mi_batch_size', type=int, default=20, help='Affects, adapting')
    method_specific_group.add_argument('-mps', '--pool-size', dest='pool_size', type=int, default=10000, help='Affects, adapting')

    # different policies for adapting
    method_specific_group.add_argument('--policy', dest='policy', type=str, default=None, help='Affects adapting')

    display_group = parser.add_argument_group('Display')
    display_group.add_argument('-deo', '--display-on-epoch-only', dest='epoch_only', action='store_true', help='Only show output per epoch')
    display_group.add_argument('-dfs', '--display-filter-size', dest='filter_size', nargs=2, type=float, default=[], help='Only show output per epoch')
    display_group.add_argument('-df', '--display-filters', dest='filters', action='store_true')

    args = parser.parse_args()

    # verify arguments are valid
    if not args.layer_files and not args.additional_layers:
        raise ValueError('You need to specify the architecture')

    if len(args.layer_files) + len(args.additional_layers) < 2:
        raise ValueError('Must have at least 2 layers')

    if args.pickle_file and args.data_file:
        raise ValueError('Pickle file overrides data, validation and test files')

    return args

def main():
    ''' Entry point '''
    # initialisation
    args = get_arguments()
    args.start_time = str(time.time())

    # create folders for the output data
    output_folder = os.path.join(OUTPUT_LOCATION, args.output_folder, '')

    layers_folder = os.path.join(output_folder, LAYERS_FOLDER, '')
    filters_folder = os.path.join(output_folder, FILTERS_FOLDER, '')

    # create sub folders
    for folder in [output_folder, layers_folder, filters_folder]:
        os.makedirs(folder, exist_ok=True)

    # set up logging
    logger = get_logger('trainer', output_folder)
    logger.info(str(args))

    # build the model
    logger.info('forcing 32 bit floats')
    theano.config.floatX = 'float32'

    nnet_model = make_model(args)
    input_layer_size = nnet_model.layers[0].initial_size[0]
    args.architecture = [ layer.initial_size for layer in nnet_model.layers ]
    logger.info('architecture: ' + str(nnet_model.layers))

    # prepare load training data if applicable
    if args.pickle_file:
        data_file, _, _ = load_from_pickle(args.pickle_file)
    else:
        data_file = args.data_file and load_from_file(input_layer_size, args.data_file)

    # a buffer for the batch of data
    batch_pool = models.Pool(nnet_model.layers[0].initial_size[0], args.batch_size)
    use_stdin = False

    if not data_file:
        logger.info('expecting stdin input')
        use_stdin = True
        data_file = [ batch_pool.data, batch_pool.data_y, 0 ]

    if nnet_model.arcs > 1 and not data_file:
        raise ValueError('This model has multiple arcs, you must specify a data file')

    # runtime statistics
    validation_runs = []
    distribution = []
    batch_count = 0

    nnet_model.begin({ 'distribution': distribution })

    def write_filters(series):
        ''' Dump layer filters '''
        # output filters of the first layers
        if int(math.sqrt(nnet_model.layers[0].initial_size[0])) ** 2 != nnet_model.layers[0].initial_size[0] and not args.filter_size:
            pass
        else:
            first_layer_filter = os.path.join(filters_folder, str(series) + '.png')
            nnet_model.layers[0].to_png(first_layer_filter, args.filter_size)

    for arc in range(nnet_model.arcs):
        # use the error function for validation and test if the model provides it
        results_func = nnet_model.error_func if nnet_model.use_error else nnet_model.validate_func

        # create the training and validation function
        train_func = nnet_model.train_func(arc, args.learning_rate, data_file[0], data_file[1], args.batch_size)
        validate_func = results_func(arc, data_file[0], data_file[1], args.batch_size)

        # start training
        try:
            for epoch in range(args.epoches):
                for batch in itertools.count() if use_stdin else range(math.ceil(data_file[2] / args.batch_size)):
                    # load a batch from stdin if neccessary
                    if use_stdin:
                        batch_x, batch_y = common.stdin_batch(input_layer_size, args.batch_size)
                        batch_pool.add(batch_x, batch_y)

                    def format_results(npl):
                        ''' Format results if they are avaliable '''
                        return 'n/a' if not npl else '%10.3f' % np.asscalar(npl[0])

                    # don't bother validating and testing if we don't need to
                    if not args.epoch_only or batch == 0:
                        # validate
                        validate_results = validate_func(0 if use_stdin else batch)

                        if use_stdin:
                            from collections import Counter
                            dist = Counter(batch_y)
                            distribution.append({ str(k): v / sum(dist.values()) for k, v in dist.items() })

                            validation_runs.append((time.time(), validate_results[0]))

                    # train
                    train_func(0 if use_stdin else batch)
                    if args.filters:
                        write_filters(batch_count)

                    # display output
                    if not args.epoch_only or batch == 0:
                        logger.info('arc: %4d, epoch: %4d, batch: %4d%s, %s: validate (%s): %s',
                                    arc, epoch, batch, '' if use_stdin else '/' + str(math.ceil(data_file[2] / args.batch_size))
                                    , 'errors' if nnet_model.use_error else 'cost', 'stream' if use_stdin else 'batch', format_results(validate_results))

                    batch_count += 1

        except StopIteration:
            pass

    if validation_runs:
        logger.info('validation average: %f', sum(v[-1] for v in validation_runs) / len(validation_runs))
    model_logs = nnet_model.end()

    # output layer weights
    for i, layer in enumerate(nnet_model.layers):
        path = os.path.join(layers_folder, str(i) + '.npz')
        logger.info('dumping layer %d to %s', i, path)
        layer.to_npz(path)

    if args.filters:
        logger.info('dumping 1st layer filters')
        write_filters('end')

    def write_log(log):
        ''' Output data '''
        name = log['name']
        logger.info('writing ' + os.path.join(output_folder, name))
        if 'csv' in log:
            csv = log['csv']
            columns, data = csv
            with open(os.path.join(output_folder, name), 'w') as f:
                f.write(columns + '\n')
                for item in data:
                    f.write(','.join(str(cell) for cell in item))
                    f.write('\n')
        elif 'json' in log:
            with open(os.path.join(output_folder, name), 'w') as f:
                f.write(log['json'])

    # output config
    import json
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # output data distribution
    with open(os.path.join(output_folder, 'distribution.json'), 'w') as f:
        json.dump(distribution, f)

    # output validation log
    write_log({ 'name': 'validation.csv', 'csv': ('time,error', validation_runs) })

    # output the model's log
    for log in model_logs:
        write_log(log)

if __name__ == '__main__':
    main()
