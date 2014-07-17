import os
import sys
import time
import pandas as pd
from PIL import Image
import numpy
import random
import math
import theano
import theano.tensor as T
#from sklearn.utils import shuffle
#from random import shuffle

def load_training_data(directory,nSample):
    #############
    # LOAD DATA #
    #############
    print '... loading data'	
    trainLabels = pd.read_csv(os.path.join(directory,'trainLabels.csv'))
    trainLabels['numeric']=0
    labelGroups = trainLabels.groupby(['label'])
    grpKeys = list(labelGroups.groups.keys())
    print '... number of labels is ', len(grpKeys)
    grpIndex = dict(zip(grpKeys,range(len(grpKeys))))
    trainLabels['numeric']=trainLabels['label'].apply(lambda x: grpIndex[x])
    numericLabels = numpy.asarray(trainLabels['numeric'])
    nTrain = trainLabels['id'].shape[0]
    print '... number of images is ', nTrain
    nTrain = numpy.minimum(nTrain,nSample)
    labels = numericLabels[0:nTrain]
    images = numpy.ndarray(shape=(nTrain,32*32),dtype=float)
    trainId = range(nTrain)
    for iImg in range(nTrain):
        fileName = os.path.join(directory, "train", str(iImg+1)+".png")
        img = numpy.array(Image.open(fileName))
        #images[iImg] = numpy.reshape(0.114*img[:,:,0]+0.436*img[:,:,1]+0.615*img[:,:,2],32*32)
        images[iImg] = numpy.reshape(0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2],32*32)


    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
		
#    random_state = numpy.random.RandomState(243)
#    images, labels, trainId = shuffle(images,labels,trainId,random_state=random_state)
    random.seed(243)
    rndmState = random.getstate()
    trainId = range(1,nTrain+1)
    random.shuffle(trainId)
    random.setstate(rndmState)
    random.shuffle(images)
    random.setstate(rndmState)
    random.shuffle(labels)
    nOneThird = nSample/3
    test_set_x, test_set_y = shared_dataset(images[0:nOneThird],labels[0:nOneThird])
    valid_set_x, valid_set_y = shared_dataset(images[nOneThird:2*nOneThird],labels[nOneThird:2*nOneThird])
    train_set_x, train_set_y = shared_dataset(images[2*nOneThird:3*nOneThird],labels[2*nOneThird:3*nOneThird])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y),grpKeys,trainId[nOneThird:2*nOneThird],labels[nOneThird:2*nOneThird]]
    return rval

def load_test_data(directory,iStart,iStop):
    #############
    # Load Data #
    #############
    print '... loading test data',iStart,',',iStop
    nTest = iStop-iStart+1
    images = numpy.ndarray(shape=(nTest,32*32),dtype=float)
    for iImg in range(iStart,iStop+1):
        fileName = os.path.join(directory, "test", str(iImg)+".png")
        img = numpy.array(Image.open(fileName))
        images[iImg-iStart-1] = numpy.reshape(0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2],32*32)

    def shared_dataset(data_x, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        return shared_x

    test_set_x = shared_dataset(images)
    return test_set_x

if __name__ == '__main__':
    load_data('../',50);

