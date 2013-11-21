"""
building convolutional network on facial emotion dataset

with IO module

usage: test it on GPU
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cNetwork.py

"""


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

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer


class Dataset(object):
    def __init__(self):        
        #self.base_path = os.environ['PYLEARN2_DATA_PATH'] + '/icml_2013_emotions/'
        #self.base_path = '/afs/csail.mit.edu/u/b/bzhou/data/pylearn2/icml_2013_emotions/'
        self.base_path = '/home/bolei/data/icml_2013_emotions/'
        self.files = {'train': 'train.csv', 'test' : 'test.csv', 'test_label':'test_all_truth.csv' }
        self.validationSetRatio = 0.2 # the size of validation set
        self.data = []
    
    def loadTest(self, preprocessFlag = 0):
        print '... loading testing data'
        csv_file = open(self.base_path + self.files['test'], 'r')
        reader = csv.reader(csv_file)        
        # Discard header
        row = reader.next()
        
        X_list = []
        
        for row in reader:
            X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            if preprocessFlag ==1:                
                X_list.append(self.histeq(X_row))   
            else:
                X_list.append(X_row)
            
        csv_file = open(self.base_path + self.files['test_label'], 'r')
        reader = csv.reader(csv_file)  
        
        # Discard header       
        y_list = []
        
        for row in reader:
            y_str, = row
            y = int(y_str)
            y_list.append(y)  
        
        assert (len(X_list) == len(y_list))     
        test_set_x, test_set_y = self.shared_dataset(X_list, y_list)
        rval = [(test_set_x, test_set_y)]

        return rval
        
    def loadTrain(self, preprocessFlag=0, flipFlag=0): 
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

        for row in reader:
            y_str, X_row_str = row
            y = int(y_str)
            y_list.append(y)

            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            if preprocessFlag ==1:
                X_list.append(self.histeq(X_row))    
            else:
                X_list.append(X_row)
        assert (len(X_list) == len(y_list))
        print '... randomly separting training set and validation set' 

        if flipFlag == 1:
            print "including flipped training data..."
            X_list_flipLR, X_list_flipUD = self.flipData(X_list)
            X_list = X_list + X_list_flipLR
            y_list = y_list + y_list
        elif flipFlag == 2:
            print "including flipped training data..."
            X_list_flipLR, X_list_flipUD = self.flipData(X_list)
            X_list = X_list + X_list_flipLR + X_list_flipUD
            y_list = y_list + y_list + y_list
        else:
            pass        
            
        
        sizeTrainSet = len(X_list)        
        startIndex_validation = int(self.validationSetRatio*sizeTrainSet)
        randomIndex = np.random.permutation([i for i in range(sizeTrainSet)])
        

        X_list_validation = [X_list[i] for i in randomIndex[:startIndex_validation]]
        y_list_validation = [y_list[i] for i in randomIndex[:startIndex_validation]]    
            
        
        X_list_train = [X_list[i] for i in randomIndex[startIndex_validation:]]
        y_list_train = [y_list[i] for i in randomIndex[startIndex_validation:]]   
        
        del X_list
        train_set_x, train_set_y = self.shared_dataset(X_list_train, y_list_train)
        valid_set_x, valid_set_y = self.shared_dataset(X_list_validation, y_list_validation)
        
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
        return rval

        
    def shared_dataset(self, data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables
            for the convenience of GPU CUDA computation
        """ 

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')  
        
    def histeq(self, face_vector,nbr_bins=256):
      ## histogram normalization for the face images.
      # get image histogram
        imhist,bins = np.histogram(face_vector,nbr_bins,normed=True)
        cdf = imhist.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

      # use linear interpolation of cdf to find new pixel values
        face_vector_normalized = np.interp(face_vector,bins[:-1],cdf)

        return face_vector_normalized

    def plotSample(self, dataList, labelList,  sampleIndex, saveFile='errorSample.jpg', outputNum = 100, sizeImg=[48,48]):
        #TODO
        # plotting out the samples according to the sample Index
        # dataList = [[3,4,5,5..],[4,5,6,7]...,...] 
        # sampleIndex = [13,4,5,3]
        # output a image with [sqrt(otuputNum), sqrt(outputNum)]

        nRow = int(np.sqrt(outputNum))
        img_plot = Image.new('L', (sizeImg[0]*nRow,sizeImg[1]*nRow))
        imgIndex = 0
        for i in xrange(0, sizeImg[0]*nRow, sizeImg[0]):
            for j in xrange(0, sizeImg[1]*nRow, sizeImg[0]):
                if len(sampleIndex)<= imgIndex:
                    break
                sampleVector = dataList[sampleIndex[imgIndex]]
                singleImg = np.asarray(sampleVector).astype('uint8').reshape((sizeImg[0],sizeImg[1]))
                im = Image.fromarray(singleImg)
                #im.save('testSample.jpg')
                #raw_input('wait')
                img_plot.paste(im, (i,j))
                imgIndex += 1
        print 'save misclassified sample'
        img_plot.save(saveFile)
    
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
 
class DeepLearner():
    def __init__(self):	
        self.tag = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
        self.preprocessFlag = 0
        self.flipFlag = 2
		
		### Model parameters
        self.nkerns=[30, 40, 40] # number of kernal at each layer, current best performance is 50.0% on testing set, kernal number is [30,40,40]
        self.batch_size = 500
        self.n_epochs = 2000 # number of iterations
        self.ishape = (48, 48)  # size of input images
        self.nClass = 7
        self.rng = np.random.RandomState(23455)	     
        self.save_as_file = 'dl.pkl'
        self.read_from_file = 'dl.pkl'  					

    def Train(self):	
        batch_size = self.batch_size
        learning_rate =  0.02

        n_epochs = self.n_epochs
        nkerns = self.nkerns		
        ishape = self.ishape
        nClass = self.nClass
        rng = self.rng    
    
        myDataset = Dataset()
        datasets = myDataset.loadTrain(preprocessFlag = self.preprocessFlag, flipFlag = self.flipFlag)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        dataset_test = myDataset.loadTest(preprocessFlag = self.preprocessFlag)
        test_set_x, test_set_y = dataset_test[0] 


        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                       
		######################
		# BUILD ACTUAL MODEL #
		######################

        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[0]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=(batch_size, 1, ishape[0], ishape[0]),
                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], 22, 22),
                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
                image_shape=(nkerns[0], nkerns[1], 9, 9),
                filter_shape=(nkerns[2], nkerns[1], 2, 2), poolsize=(2, 2))    

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer3_input = layer2.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[2] * 4 * 4,
                             n_out=500, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=nClass)

        # the cost we minimize during training is the NLL of the model
        cost = layer4.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model

        validate_model = theano.function([index], layer4.errors(y),
                givens={
                    x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                    y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
                    
        test_model = theano.function([index], layer4.errorsLabel(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})

        # create a list of all model parameters to be fit by gradient descent
        params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        train_model = theano.function([index], cost, updates=updates,
              givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})

        test_set_truth = theano.function([],test_set_y)()
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_train_batches, \
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        
                        #test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_output = [test_model(i) for i in xrange(n_test_batches)]
                        test_losses = [item[0] for item in test_output]
                        #test_y_gt = [label[0] for label in item[1] for item in test_output] #
                        test_y_pred = np.array([item[1] for item in test_output]) 
                        test_y_pred = np.concatenate(test_y_pred)
                        #test_y_model_gt = np.array([item[2] for item in test_output]) 
                        #test_y_model_gt = np.concatenate(test_y_model_gt)
                        test_y_gt = [test_set_truth[i*batch_size:(i+1)*batch_size] for i in xrange(n_test_batches)]
                        test_y_gt = np.concatenate(test_y_gt)
                            

                        #print test_y_pred
                        #print test_y_model_gt
                        #print test_y_gt

                        
                        errorNum = np.count_nonzero(test_y_gt - test_y_pred)
                        errorSampleIndex = [i for i in range(len(test_y_pred)) if test_y_pred[i]!=test_y_gt[i]] 
                        #print errorNum, len(errorSampleIndex)

                        test_score = np.mean(test_losses)
                        print(('  epoch %i, minibatch %i/%i, test error of best '
                               'model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                        print((' on all test sample %f %%')%((float(errorNum)/float(len(test_y_pred))*100.)))
                        
                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        
        #TODO: write the code to save the trained model and test the trained model on test data
        
        print('Best validation score of %f %% obtained at iteration %i,'\
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        
        print "saving model.."		
        params_value = []
        for param in params:
            params_value.append(param.get_value())

        pkl_file = file(self.save_as_file, 'wb')
        cPickle.dump(params_value, pkl_file, protocol=cPickle.HIGHEST_PROTOCOL)
        pkl_file.close()		
        
    def Test(self):
   
		pkl_file = file(self.read_from_file, 'rb')
		params_value = cPickle.load(pkl_file)
		pkl_file.close()

		myDataset = Dataset()
		batch_size = self.batch_size
		dataset_test = myDataset.loadTest(self.preprocessFlag)			
		test_set_x, test_set_y = dataset_test[0]
		# compute number of minibatches for testing
		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_test_batches /= batch_size
	
		# allocate symbolic variables for the data
		index = T.lscalar()  
		x = T.matrix('x')  
		y = T.ivector('y') 
		
		### Model parameters
		nkerns = self.nkerns
		ishape = self.ishape  
		nClass = self.nClass		
		rng = self.rng

		print '... rebuilding the model'

		layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[0]))
		layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
		        image_shape=(batch_size, 1, ishape[0], ishape[0]),
		        filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
		layer0.W.set_value(params_value[8])
		layer0.b.set_value(params_value[9])

		layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
		        image_shape=(batch_size, nkerns[0], 22, 22),
		        filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
		layer1.W.set_value(params_value[6])
		layer1.b.set_value(params_value[7])

		layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
		        image_shape=(nkerns[0], nkerns[1], 9, 9),
		        filter_shape=(nkerns[2], nkerns[1], 2, 2), poolsize=(2, 2))    
		layer2.W.set_value(params_value[4])
		layer2.b.set_value(params_value[5])

		layer3_input = layer2.output.flatten(2)
		layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[2] * 4 * 4,
		                     n_out=500, activation=T.tanh)
		layer3.W.set_value(params_value[2])
		layer3.b.set_value(params_value[3])

		layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=nClass)
		layer4.W.set_value(params_value[0])
		layer4.b.set_value(params_value[1])

        
		test_model = theano.function([index], layer4.errors(y),
		        givens={
		            x: test_set_x[index * batch_size: (index + 1) * batch_size],
		            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

		test_losses = [test_model(i) for i in xrange(n_test_batches)]
		test_score = np.mean(test_losses)

		print('Test performance %f %%' % (test_score * 100.))

    def TestSingle(self, faceArray):  
        pkl_file = file(self.read_from_file, 'rb')
        params_value = cPickle.load(pkl_file)
        pkl_file.close()

	
		# allocate symbolic variables for the data
        index = T.lscalar()  
        x = T.matrix('x')  
        y = T.ivector('y') 
		
		### Model parameters
        nkerns = self.nkerns
        ishape = self.ishape  
        nClass = self.nClass		
        rng = self.rng
        batch_size = self.batch_size       
        print '... rebuilding the model'

        layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[0]))
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=(batch_size, 1, ishape[0], ishape[0]),
                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
        layer0.W.set_value(params_value[8])
        layer0.b.set_value(params_value[9])

        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], 22, 22),
                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
        layer1.W.set_value(params_value[6])
        layer1.b.set_value(params_value[7])

        layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
                image_shape=(nkerns[0], nkerns[1], 9, 9),
                filter_shape=(nkerns[2], nkerns[1], 2, 2), poolsize=(2, 2))    
        layer2.W.set_value(params_value[4])
        layer2.b.set_value(params_value[5])

        layer3_input = layer2.output.flatten(2)
        layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[2] * 4 * 4,
                             n_out=500, activation=T.tanh)
        layer3.W.set_value(params_value[2])
        layer3.b.set_value(params_value[3])

        layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=nClass)
        layer4.W.set_value(params_value[0])
        layer4.b.set_value(params_value[1])

        myDataset = Dataset()
        
        ### duplicate the input array x batch_size time
        data_x = [faceArray] * batch_size
        data_y = [1] * batch_size
        data_x, data_y = myDataset.shared_dataset(data_x, data_y, borrow=True)

        test_model = theano.function([index], layer4.errorsLabel(y),
                givens={
                    x: data_x[index * batch_size: (index + 1) * batch_size],
                    y: data_y[index * batch_size: (index + 1) * batch_size]})

        test_output = test_model(0)
        test_score = np.mean(test_output[0])
        test_y_pred = test_output[1]
        #print test_y_pred
        return self.tag[test_y_pred[0]]
		
if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    learner = DeepLearner()
    if sys.argv[1] == 'train':
        learner.Train()
    if sys.argv[1] == 'test':
        learner.Test()
