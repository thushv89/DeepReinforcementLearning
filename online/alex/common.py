''' Data input and layer creation '''

import sys
import numpy as np

def stdin_batch(input_layer_size, batch_size):
    ''' Load a batch from stdin '''
    batch_x, batch_y = from_bytes(input_layer_size, sys.stdin.buffer.read((input_layer_size + 1) * 4 * batch_size))
    if batch_x.shape[0] == 0:
        raise StopIteration
    else:
        return batch_x, batch_y

def from_bytes(input_layer_size, block):
    ''' Load a raw block of bytes as a training example '''
    data = np.frombuffer(block, 'float32')
    data = data.reshape((-1, input_layer_size + 1))

    batch_x = data[:,:input_layer_size]
    batch_y = data[:,input_layer_size].astype(np.int32)

    return batch_x, batch_y
