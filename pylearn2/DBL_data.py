import cPickle
import os
import sys
import time
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


"""
parent class with preprocessor
"""
class DataIO(DenseDesignMatrix):
    def __init__(self,X, y, view_converter):        
        super(DataIO, self).__init__(X=X, y=y, view_converter=view_converter)

    def loadFile(self,filename=None,start= 0, stop=-1,trainindex=0):
        if not os.path.exists(filename):
            print filename+" : doesn't exist"
            return None
        else:
            # pre-computed
            X_path = filename[:filename.rfind('.')] + '.X'+str(trainindex)+'.npy'
            Y_path = filename[:filename.rfind('.')] + '.Y'+str(trainindex)+'.npy'   
            if os.path.exists(X_path):
                X = np.load(X_path)
                if trainindex:
                    y = np.load(Y_path)
                else:
                    y = None
            else:            
                X, y = self._load_data(filename, trainindex!=0)
            # default: X=(m,n), m instances of n dimensional feature
            num = X.shape[0]
            if stop==-1 or stop>num:
                stop = num
            X = X[start:stop, :]

            if not os.path.exists(X_path):
                np.save(X_path, X)
                print "save: "+X_path
                if y is not None:
                    y = y[start:stop]
                    np.save(Y_path, y)         
                    print "save: "+Y_path 
            """
            print y[:10]
            print X[:10,:]                              
            """
            return X,y

    def _load_data(self, path, expect_labels):
        return
    
    def label_id2arr(self,y,numclass):
        one_hot = np.zeros((y.shape[0],numclass),dtype='float32')
        for i in xrange(y.shape[0]):
            one_hot[i,y[i]] = 1.
        return one_hot

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

class Occ(DataIO):    
    def __init__(self,which_set,numclass,
            base_path = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions',
            start = 0,
            stop = -1,
            preprocessor = None,
            trainindex=0,
            ishape=None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),            
            fit_test_preprocessor = False,                        
            flip=0
            ):
        files = {'train': 'occ_train.csv', 'public_test' : 'test.csv'}
        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)
        
        X, y = self.loadFile(base_path + '/' + filename, start,stop,trainindex)
        # train_index
        if flip:
            X_list_flipLR, X_list_flipUD = self.flipData(X)
            X = X + X_list_flipLR
            y = y + y        
        
        view_converter = DefaultViewConverter(shape=np.append(ishape.shape,ishape.num_channels), axes=axes)
        super(Occ, self).__init__(X=X, y=self.label_id2arr(y,numclass), view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
                

    def _load_data(self, path, expect_labels):
        assert path.endswith('.csv')
        # Convert the .csv file to numpy
        csv_file = open(path, 'r')
        import csv
        reader = csv.reader(csv_file)
        # Discard header
        # row = reader.next()
        y_list = []
        X_list = []
        for row in reader:
            row = row[0]
            if expect_labels:                
                y = int(row[:row.find(' ')])
                y_list.append(y)
                X_row_str = row[row.find(' ')+1:]
            else:
                X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list).astype('float32')
        if expect_labels:
            y = np.asarray(y_list)            
        else:
            y = None

        return X, y


class ICML_emotion(DataIO):
    """
    A Pylearn2 Dataset class for accessing the data for the
    facial expression recognition Kaggle contest for the ICML
    2013 workshop on representation learning.
    """
    def __init__(self,which_set,numclass,
            base_path = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions',
            start = 0,
            stop = -1,
            preprocessor = None,
            trainindex=0,
            ishape=None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),            
            fit_test_preprocessor = False,                        
            flip=0
            ):
        files = {'train': 'train.csv', 'public_test' : 'test.csv'}
        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)
        
        X, y = self.loadFile(base_path + '/' + filename, start,stop,trainindex)
        # train_index
        if flip:
            X_list_flipLR, X_list_flipUD = self.flipData(X)
            X = X + X_list_flipLR
            y = y + y    

        view_converter = DefaultViewConverter(shape=np.append(ishape.shape,ishape.num_channels), axes=axes)
        super(ICML_emotion, self).__init__(X=X, y=self.label_id2arr(y,numclass), view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def _load_data(self, path, expect_labels):
        assert path.endswith('.csv')
        # Convert the .csv file to numpy
        csv_file = open(path, 'r')
        import csv
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
        else:
            y = None
        return X, y
            