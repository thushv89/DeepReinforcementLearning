#!/usr/bin/env python3

import sys, argparse
import pickle
import struct
import random
import math
import numpy as np

from collections import defaultdict

def distribute_as(dist, n):
    cumsum = np.cumsum(dist)

    for _ in range(n):
        x = random.random()
        found = False
        for i, y in enumerate(cumsum):
            if x < y:
                yield i
                found = True
                break

        if not found:
            yield dist.shape[0] - 1

def main():
    parser = argparse.ArgumentParser(description='Generate distribution')

    parser.add_argument('-tf', '--text-format', dest='text_format', action='store_true', help='Write in text format')
    parser.add_argument('-q', '--silent', dest='silent', action='store_true', help='No stderr output')
    parser.add_argument('pickle_file', type=str, help='Pickle file to load data from')
    parser.add_argument('granularity', type=int, help='Granularity of distribution')
    parser.add_argument('elements', type=int, help='How much data to generate')
    parser.add_argument('effect', type=str, help='What filter to apply')
    parser.add_argument('seed', type=int, default=None, help='Seed')
    parser.add_argument('--header', action='store_true', dest='header', help='Make a header')
    parser.add_argument('--even', action='store_true', dest='even', help='even')

    args = parser.parse_args()

    # pipe death
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # seed the generator
    if args.seed != None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    with open(args.pickle_file, 'rb') as f:
        train, _, _ = pickle.load(f, encoding='latin1')

    data = defaultdict(list)
    train_x, train_y = train

    # sort the data into bins depending on labels
    for i in range(train_x.shape[0]):
        data[train_y[i]].append(train_x[i])

    # randomly sample a GP
    def kernel(a, b):
        """ Squared exponential kernel """
        sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist)

    # number of samples
    n = math.ceil(args.elements / args.granularity)
    Xtest = np.linspace(0, 10, n).reshape(-1, 1)
    L = np.linalg.cholesky(kernel(Xtest, Xtest) + 1e-6 * np.eye(n))

    # massage the data to get a good distribution
    f_prior = np.dot(L, np.random.normal(size=(n, len(data))))
    f_prior -= f_prior.min()
    f_prior = f_prior ** math.ceil(math.sqrt(len(data)))
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    if args.header:
        meta = { 'input_dimension': 28 * 28, 'column': 'mnist_1', 'column_id': 0, 'start': 0, 'gap': 3600, 'labels': 10 }
        import json
        blob = json.dumps(meta).encode('utf-8')
        sys.stdout.buffer.write(struct.pack('<I', len(blob)))
        sys.stdout.buffer.write(blob)

    for i, dist in enumerate(f_prior):
        if not args.silent:
            print('>>', i, '/', f_prior.shape[0], dist, file=sys.stderr)
        # generate a data of args.granularity with the specified distribution
        if args.even:
            dist = np.array([ 1 / len(dist) for _ in dist])
        for label in distribute_as(dist, args.granularity):
            example = random.choice(data[label])
            if args.effect == 'noise':
                example = example + np.random.random_sample((example.shape[0],))
            elif args.effect == 'none':
                pass
            else:
                raise Exception('Unknown effect')

            example = np.minimum(1, example).astype('float32')

            if args.text_format:
                print(' '.join(map(str, example)), label)
            else:
                sys.stdout.buffer.write(example.view('b'))
                sys.stdout.buffer.write(struct.pack('@f', float(label)))

if __name__ == '__main__':
    main()
