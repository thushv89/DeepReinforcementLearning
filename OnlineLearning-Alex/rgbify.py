#!/usr/bin/env python3

import png
import math
import sys
import numpy as np
import argparse
import itertools

parser = argparse.ArgumentParser(description='MDAE')
parser.add_argument('filename', metavar='filename', type=str, help='File name')
parser.add_argument('fw', metavar='width', type=int, default=None, help='Filter width')
parser.add_argument('fh', metavar='height', type=int, default=None, help='Filter height')
args = parser.parse_args()

r = png.Reader(args.filename)
w, h, rows_iter, _ = r.read()

n_columns = math.ceil((w + 1)/(args.fw + 1))
n_rows = math.ceil((h + 1)/(args.fh + 1))

output = []
empty_row = [0] * w * 3
row_group = math.ceil(args.fh / 3)
try:
    while True:
        r = np.concatenate(list(itertools.islice(rows_iter, 0, row_group)))
        g = np.concatenate(list(itertools.islice(rows_iter, 0, row_group)))
        b = np.concatenate(list(itertools.islice(rows_iter, 0, row_group)))
        output += [ val for tup in zip(r, g, b) for val in tup ]
        next(rows_iter)
        output += empty_row
except StopIteration:
    pass

writer = png.Writer(width=w, height=n_columns * row_group + n_columns - 1)
with open(args.filename + '.rgb.png', 'wb') as f:
    writer.write_array(f, output)
