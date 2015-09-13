__author__ = 'Thushan Ganegedara'
import numpy as np
import gzip
import pickle
from pylab import *

def create_image_from_vector(vec,filename):
    from PIL import Image
    new_vec = vec*255.
    #img = Image.fromarray(np.reshape(vec*255,(-1,28)).astype(int),'L')
    #img.save(filename +'.png')
    imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    show()

if __name__ == '__main__':

    #f = gzip.open("data\mnist.pkl.gz", 'rb')
    with open('data\mnist.pkl', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()
    create_image_from_vector(train_set[0][2,:],'test2222')