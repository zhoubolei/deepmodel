"""
building convolutional network on facial emotion dataset

Bolei Zhou

usage: test it on GPU
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cNetwork.py

"""
#TODO: 
# - the module of IO for loading and saving model parameters.
# - the module of testing the best model on testing dataset.
# - the module of plotting error samples and confusion matrix
# - the module of handeling the output of face detection 

import cPickle
import os
import sys
import time

import csv

import numpy as np

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
        self.base_path = '/afs/csail.mit.edu/u/b/bzhou/data/pylearn2/icml_2013_emotions/'
        #self.base_path = '/home/bolei/data/icml_2013_emotions/'
        self.files = {'train': 'train.csv', 'test' : 'test.csv'}
        self.validationSetRatio = 0.2 # the size of validation set
        self.data = []
    
    def loadTest(self, preprocess = 1):
        print '... loading testing data'
        file_data = 'test.csv'
        file_label = 'test_all_truth.csv'
        csv_file = open(self.base_path + file_data, 'r')
        reader = csv.reader(csv_file)        
        # Discard header
        row = reader.next()
        
        X_list = []
        
        for row in reader:
            X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            if preprocess ==1:                
                X_list.append(self.histeq(X_row))   
            else:
                X_list.append(X_row)
            
        csv_file = open(self.base_path + file_label, 'r')
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
        
    def loadTrain(self, preprocess=1):    
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
            if preprocess ==1:
                X_list.append(self.histeq(X_row))    
            else:
                X_list.append(X_row)
        assert (len(X_list) == len(y_list))
        print '... randomly separting training set and validation set' 

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

    def plotExample(self, dataList, labelList,  sampleIndex, outputNum = 10000):
        #TODO
        # plotting out the samples according to the sample Index
        # dataList = [[3,4,5,5..],[4,5,6,7]...,...] 
        # sampleIndex = [13,4,5,
        # output a image with [sqrt(otuputNum), sqrt(outputNum)]
        """
        Hint: 
        blank_image = Image.new("RGB", (800, 600))
        blank_image.paste(image64, (0,0))
        blank_image.paste(fluid128, (400,0))
        blank_image.paste(fluid512, (0,300))
        blank_image.paste(fluid1024, (400,300))
        blank_image.save(out)
        
        
        import Image

        #opens an image:
        im = Image.open("1_tree.jpg")
        #creates a new empty image, RGB mode, and size 400 by 400.
        new_im = Image.new('RGB', (400,400))

        #Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((100,100))
        #Iterate through a 4 by 4 grid with 100 spacing, to place my image
        for i in xrange(0,500,100):
            for j in xrange(0,500,100):
                #I change brightness of the images, just to emphasise they are unique copies.
                im=Image.eval(im,lambda x: x+(i+j)/30)
                #paste the image at location i,j:
                new_im.paste(im, (i,j))

        new_im.show()
        """
        sampleList = [dataList[i] for i in sampleIndex]
        
        imgVector = sampleList[0]
        singleImg = np.asarray(imgVector).astype('uint8').reshape((48,48))
        im = Image.fromarray(singleImg)
        im.save('myImg.jpg')
        pass
        
def runDeepLearning():
### Loading training set and separting it into training set and testing set
   
    myDataset = Dataset()
    preprocess = 0
    datasets = myDataset.loadTrain(preprocess)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    dataset_test = myDataset.loadTest(preprocess)
    test_set_x, test_set_y = dataset_test[0]

    
### Model parameters
    learning_rate = 0.02
    n_epochs = 2000
    nkerns=[30, 40, 40] # number of kernal at each layer, current best performance is 50.0% on testing set, kernal number is [30,40,40]
    batch_size = 800
    
    # compute number of minibatches for training, validation and testing
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
                        # [int] labels


    ishape = (48, 48)  # size of input images
    nClass = 7
    
    rng = np.random.RandomState(23455)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

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
                
    test_model = theano.function([index], layer4.errors(y),
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

    while (epoch < n_epochs) and (not done_looping):
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
                    
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    
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

if __name__ == '__main__':
    runDeepLearning()

