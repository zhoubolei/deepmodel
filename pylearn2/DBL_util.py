import cPickle
import os
import sys
import time

import csv

import numpy as np
import Image

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.utils.string_utils import preprocess

class EmotionsDataset(DenseDesignMatrix):
    """
    A Pylearn2 Dataset class for accessing the data for the
    facial expression recognition Kaggle contest for the ICML
    2013 workshop on representation learning.
    """

    def __init__(self, which_set,
            base_path = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions',
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),
            fit_test_preprocessor = False,
            randindex=None,
            trainindex=None):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                This directory should be writable; if the .csv files haven't
                already been converted to npy, this class will convert them
                to save memory the next time they are loaded.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.test_args = locals()
        self.test_args['which_set'] = 'public_test'
        self.test_args['fit_preprocessor'] = fit_test_preprocessor
        del self.test_args['start']
        del self.test_args['stop']
        del self.test_args['self']

        files = {'train': 'train.csv', 'public_test' : 'test.csv'}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        path = base_path + '/' + filename

        path = preprocess(path)
        
        X, y = self._load_data(path, which_set == 'train')
        

        if start is not None:
            assert which_set != 'test'
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            if y is not None:
                y = y[start:stop, :]
        if trainindex:
            X_list_flipLR, X_list_flipUD = self.flipData(X)
            X = X + X_list_flipLR
            y = y + y
        
        view_converter = DefaultViewConverter(shape=[48,48,1], axes=axes)

        super(EmotionsDataset, self).__init__(X=X, y=y, view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

    def get_test_set(self):
        return EmotionsDataset(**self.test_args)

    def _load_data(self, path, expect_labels):
        assert path.endswith('.csv')
        X_path = path[:-4] + '.X.npy'
        Y_path = path[:-4] + '.Y.npy'	
	"""        

        if os.path.exists(X_path):
            X = np.load(X_path)
            if expect_labels:
                y = np.load(Y_path)
            else:
                y = None
            return X, y
	"""

        # Convert the .csv file to numpy
        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        row = reader.next()

        y_list = []
        X_list = []

        for row in reader:
            if expect_labels:
                y_str, X_row_str = row
                y = int(y_str)
                y_list.append(y)
            else:
                X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list).astype('float32')
        if expect_labels:
            y = np.asarray(y_list)

            one_hot = np.zeros((y.shape[0],7),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot
        else:
            y = None

        #np.save(X_path, X)
        #if y is not None:
        #    np.save(Y_path, y)

        return X, y
        
    def flipData(self, sampleList, sizeImg = [48, 48]):
        # flip the image set from left to right, and upside down
        print "flip the images"
        sampleList_LR = []
        sampleList_UD = []
        for i in range(len(sampleList)):
            curSampleVector = sampleList[i]
            singleImg = np.asarray(curSampleVector).astype('uint8').reshape((sizeImg[0],sizeImg[1]))
            #singleImg = singleImg.reshape((sizeImg[0],sizeImg[1]))
            singleImg_ud = np.flipud(singleImg)
            singleImg_lr = np.fliplr(singleImg)
            sampleList_UD.append(list(singleImg_ud.reshape((sizeImg[0]*sizeImg[1]))))
            sampleList_LR.append(list(singleImg_lr.reshape((sizeImg[0]*sizeImg[1]))))
        return sampleList_LR, sampleList_UD

class DataPylearn2(DenseDesignMatrix):
    def __init__(self,ds,ishape,numclass=-1,axes = ('b', 0, 1, 'c'),fit_preprocessor=True):
        X = ds[0]
        y = ds[1]        
        y_mat = y
        if numclass>0:
            y_mat=[]
            for yi in y:
                tmp = np.zeros(numclass)
                tmp[yi] = 1
                y_mat.append(tmp)
            y_mat = np.asarray(y_mat).astype('float32')
           
        view_converter = DefaultViewConverter(shape=ishape, axes=axes)
        super(DataPylearn2, self).__init__(X=X, y=y_mat, view_converter=view_converter)

    def preprocess(self):
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)
        self = pre.apply(self)


class LoadData(object):
    def __init__(self,base_path='',valid_r=2/3):
        self.base_path = base_path
        self.files = {'train': 'train.csv', 'test' : 'test.csv', "truth": 'test_all_truth.csv'}
        self.validationSetRatio = valid_r # the size of validation set
        self.data = []

    def loadTest(self, preprocessFLAG = 0,cutFLAG=-1):
        print '... loading testing data'
        csv_file = open(self.base_path + self.files['test'], 'r')
        reader = csv.reader(csv_file)
        # Discard header
        row = reader.next()
        X_list = []
        cc = 0
        for row in reader:
            X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            if preprocessFLAG ==1:
                X_list.append(self.histeq(X_row))
            else:
                X_list.append(X_row)
            cc += 1
            if cc==cutFLAG:
                break

        csv_file = open(self.base_path + self.files['truth'], 'r')
        reader = csv.reader(csv_file)
        # Discard header
        #row = reader.next()
        y_list = []
        cc = 0
        for row in reader:
            y_str, = row
            y = int(y_str)
            y_list.append(y)
            cc += 1
            if cc==cutFLAG:
                break

        assert (len(X_list) == len(y_list))
        X = np.asarray(X_list).astype('float32')
        y = np.asarray(y_list).astype('float32')
        return [X,y]
   
    def loadTrain(self, preprocessFLAG=0, flipFLAG=2,cutFLAG=-1):
        # preprocessFLAG: whether to do preprocess on the data
        # flipFLAG: whether augment training data with the flipped samples   
        print '... loading training data'
        csv_file = open(self.base_path + self.files['train'], 'r')
        reader = csv.reader(csv_file)
        # Discard header
        row = reader.next()
        y_list = []
        X_list = []
        cc = 0
        for row in reader:
            y_str, X_row_str = row
            y = int(y_str)
            y_list.append(y)
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            if preprocessFLAG ==1:
                X_list.append(self.histeq(X_row))
            else:
                X_list.append(X_row)
            cc += 1
            if cc==cutFLAG:
                break

        assert (len(X_list) == len(y_list))
        print '... randomly separting training set and validation set'

        if flipFLAG == 1:
            print "including flipped training data..."
            X_list_flipLR, X_list_flipUD = self.flipData(X_list)
            X_list = X_list + X_list_flipLR
            y_list = y_list + y_list
        elif flipFLAG == 2:
            print "including flipped training data..."
            X_list_flipLR, X_list_flipUD = self.flipData(X_list)
            X_list = X_list + X_list_flipLR + X_list_flipUD
            y_list = y_list + y_list + y_list
        else:
            pass

        sizeTrainSet = len(X_list)
        startIndex_validation = int(self.validationSetRatio*sizeTrainSet)
        randomIndex = np.random.permutation([i for i in range(sizeTrainSet)])
        randomIndex = range(sizeTrainSet)
        X_list_validation = [X_list[i] for i in randomIndex[:startIndex_validation]]
        y_list_validation = [y_list[i] for i in randomIndex[:startIndex_validation]]
        X_list_train = [X_list[i] for i in randomIndex[startIndex_validation:]]
        y_list_train = [y_list[i] for i in randomIndex[startIndex_validation:]]
        del X_list

        X1 = np.asarray(X_list_train).astype('float32')
        y1 = np.asarray(y_list_train).astype('float32')
        X2 = np.asarray(X_list_validation).astype('float32')
        y2 = np.asarray(y_list_validation).astype('float32')
        return  [(X1,y1),(X2,y2)]

    def histeq(self, face_vector,nbr_bins=256):
      ## histogram normalization for the face images.
      # get image histogram
        imhist,bins = np.histogram(face_vector,nbr_bins,normed=True)
        cdf = imhist.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

      # use linear interpolation of cdf to find new pixel values
        face_vector_normalized = np.interp(face_vector,bins[:-1],cdf)

        return face_vector_normalized

    def flipData(self, sampleList, sizeImg = [48, 48]):
        # flip the image set from left to right, and upside down
        print "flip the images"
        sampleList_LR = []
        sampleList_UD = []
        for i in range(len(sampleList)):
            curSampleVector = sampleList[i]
            singleImg = np.asarray(curSampleVector).astype('uint8').reshape((sizeImg[0],sizeImg[1]))
            singleImg_ud = np.flipud(singleImg)
            singleImg_lr = np.fliplr(singleImg)
            sampleList_UD.append(list(singleImg_ud.reshape((sizeImg[0]*sizeImg[1]))))
            sampleList_LR.append(list(singleImg_lr.reshape((sizeImg[0]*sizeImg[1]))))
        return sampleList_LR, sampleList_UD

