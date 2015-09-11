__author__ = 'Thushan Ganegedara'
import numpy as np
import gzip
import pickle

def create_image_from_vector(vec,filename):
    from PIL import Image
    new_vec = vec*255.
    img = Image.fromarray(np.reshape(vec*255,(-1,28)).astype(int),'L')
    img.save(filename +'.png')


if __name__ == '__main__':

    f = gzip.open("data\mnist.pkl.gz", 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()
    create_image_from_vector(train_set[0][2,:],'test2222')