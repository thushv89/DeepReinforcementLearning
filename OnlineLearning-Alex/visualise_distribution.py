#!/usr/bin/env python3
''' Output the distribution of a binary blob in the desired format '''
import sys, argparse
import itertools
import numpy as np

from collections import Counter
from mdae.common import stdin_batch

def main():
    ''' Entry point '''
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('dimensions', type=int, help='Number of dimensions')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('labels', type=int, help='Total number of labels')
    parser.add_argument('--csv', action='store_true', help='Write as CSV')

    args = parser.parse_args()

    try:
        for i in itertools.count():
            _, batch_y = stdin_batch(args.dimensions, args.batch_size)
            counts = Counter(batch_y)
            data = np.array([ counts[k] / sum(counts.values()) for k in range(args.labels) ])
            if args.csv:
                print(','.join((str(k) + ':' + str(v) for k, v in zip(range(args.labels), data))))
            else:
                print('<<', i, data, file=sys.stderr)
    except StopIteration:
        pass

if __name__ == '__main__':
    main()
