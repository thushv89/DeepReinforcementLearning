import numpy as np
import math

#import png

import theano
import theano.tensor as T

class Layer(object):
    ''' Represents a layer in the neural network '''
    __slots__ = [ 'name', 'W', 'b', 'b_prime', 'idx', 'initial_size' ]

    def __init__(self, input_layer_size, output_layer_size, zero=None, W=None, b=None, b_prime=None):
        ''' Initalise the layer '''
        self.name = '(%d*->%d*)' % (input_layer_size, output_layer_size)
        self.idx = T.ivector('idx_' + self.name)

        if zero:
            self.W = theano.shared(np.zeros((input_layer_size, output_layer_size)), dtype=theano.config.floatX)
        elif W != None:
            self.W = theano.shared(W)
        else:
            rng = np.random.RandomState(0)
            init = 4 * np.sqrt(6.0 / (input_layer_size + output_layer_size))
            initial = np.asarray(rng.uniform(low=-init, high=init, size=(input_layer_size, output_layer_size)), dtype=theano.config.floatX)

            # randomly initalise weights
            self.W = theano.shared(initial, 'W_' + self.name)

        # check explicitly aganist none since np arrays don't like to converted to a boolean value
        self.b = theano.shared(b if b != None else np.zeros(output_layer_size, dtype=theano.config.floatX), 'b_' + self.name)
        self.b_prime = theano.shared(b_prime if b_prime != None else np.zeros(input_layer_size, dtype=theano.config.floatX), 'b\'_' + self.name)
        self.initial_size = (input_layer_size, output_layer_size)

    def output(self, x):
        ''' Return the output of this layer as a MLP '''
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b)

    def _draw_filters(self, size=None):
        ''' Dump filters to a 2D array '''
        W = self.W.get_value().T

        if size:
            row_w, row_h = size
            in_width, in_height = row_h, row_w
            out_width = math.ceil(W.shape[0] ** 0.5)
            out_height = math.ceil(W.shape[0] / out_width)
        else:
            out_width, in_width   = [ math.ceil(unit ** 0.5) for unit in W.shape ]
            out_height, in_height = [ math.ceil(xy[0] / xy[1]) for xy in zip(W.shape, [out_width, in_width]) ]

        image = np.zeros((in_width * out_width + out_width - 1, in_height * out_height + out_height - 1), theano.config.floatX)

        for i, row in enumerate(W):
            row -= row.min()
            row /= row.max()

            x, y = (i % out_width) * (in_width + 1), (i // out_width) * (in_height + 1)
            image[x:x+in_width, y:y+in_height] = np.reshape(row, (in_width, in_height))

        return image

    '''def to_png(self, path, size=None):
         Write the filters array to a png
        with open(path, 'wb') as f:
            data = (255 * self._draw_filters(size)).astype(np.uint8)
            writer = png.Writer(data.shape[1], data.shape[0], greyscale=True)
            writer.write(f, data)'''

    def to_npz(self, filename):
        ''' Save data to npz format '''
        np.savez(filename, W=self.W.get_value(), b=self.b.get_value(), b_prime=self.b_prime.get_value())

    @staticmethod
    def from_npz(filename):
        ''' Create a layer using numpy npz file format'''
        npz = np.load(filename)
        return Layer(npz['W'].shape[0], npz['W'].shape[1], False, npz['W'], npz['b'], npz['b_prime'])

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
