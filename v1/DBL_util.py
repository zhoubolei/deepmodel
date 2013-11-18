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


class DataPylearn2(DenseDesignMatrix):
    def __init__(self,ds,ishape,numclass=-1,preprocess=0,axes = ('b', 0, 1, 'c'),fit_preprocessor=True):
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
        if preprocess:
            preprocessor.apply(self, can_fit=fit_preprocessor)

        super(DataPylearn2, self).__init__(X=X, y=y_mat, view_converter=view_converter)

class LoadData(object):
    def __init__(self,base_path='',valid_r=0.2):
        self.base_path = base_path
        self.files = {'train': 'train.csv', 'test' : 'test.csv'}
        self.validationSetRatio = valid_r # the size of validation set
        self.data = []

    def loadTest(self, preprocessFLAG = 0,cutFLAG=-1):
        print '... loading testing data'
        file_data = 'test.csv'
        file_label = 'test_all_truth.csv'
        csv_file = open(self.base_path + file_data, 'r')
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
        csv_file = open(self.base_path + file_label, 'r')
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
        setType = 'train'
        csv_file = open(self.base_path + self.files[setType], 'r')
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

