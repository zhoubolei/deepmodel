# integrate the face detector module and trained deep model
# running THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python DBL_integration.py

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

from DBL_layer import DBL_ConvLayers
from DBL_model import DBL_model
import os
import cPickle
import numpy as np

from faceModule import faceDetector
import cv2.cv as cv


def pipeline():
    tagEmotion = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    print 'load facedetector...'
    faceXMLpath =  "/scratch/opencv/opencv-2.4.6.1/data/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)
    
    #img_folder = "/scratch/face/thumbnails_features_deduped_publish/thumbnails_features_deduped_publish/bill clinton/"
    img_folder = "/data/vision/billf/manifold-learning/DL/Data/pubfig/images/"
    imgs = os.listdir(img_folder)
    print len(imgs)

    ## reconstruct model
    print "loading trained deep learning model..."
    FF = 'train.pkl'
    a=cPickle.load(open(FF))
    X = a.model.get_input_space().make_batch_theano()
    Y = a.model.fprop(X)
    from theano import tensor as T
    y = T.argmax(Y, axis=1)
    from theano import function
    f = function([X], y)

    
    for i in range(len(imgs)):
        input_name = img_folder + imgs[i]
        print imgs[i]
        if input_name.endswith('jpg'):
            face_set, img_rectangle = myDetector.detectImg(input_name)
            if face_set!=0:
                print face_set
                xx = np.array(face_set)
                x_500 = np.concatenate((xx, np.zeros((499, xx.shape[1]),
                dtype=xx.dtype)), axis=0)                
                from pylearn2.space import Conv2DSpace  
                ishape = Conv2DSpace(
                    shape = [48, 48],
                    num_channels = 1
                    )
                from DBL_util import DataPylearn2
                x_input = DataPylearn2([x_500, np.zeros(500)], (48,48,1))
                x_arg = x_input.X
                if X.ndim > 2:
                   x_arg = x_input.get_topological_view(x_arg)               
                y_pred = f(x_arg.astype(X.dtype))
                
                print tagEmotion[y_pred[0]]
                
                cv.ShowImage("result", img_rectangle)
                cv.WaitKey()
    

    

                
if __name__ == "__main__": 

    loadModel()




