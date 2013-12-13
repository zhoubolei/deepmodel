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
import time
from DBL_util import DataPylearn2

from pylearn2.space import Conv2DSpace  
ishape = Conv2DSpace(
    shape = [48, 48],
    num_channels = 1
    )
    
def loadModel(pklname):

    ishape = Conv2DSpace(
        shape = [48, 48],
        num_channels = 1
        )
    nclass = 7
    # create layers

    #nk = [30, 40]
    nk = [32,20,10]
    ks = [[8,8],[5,5],[3,3]]
    ir = [0.05,0.05,0.05]
    ps = [[4,4],[4,4],[2,2]]
    pd = [[2,2],[2,2],[2,2]]
    kn = [0.9,0.9,0.9]
    layers = DBL_ConvLayers(nk,ks,ir,ps,pd,kn)   
    layer_soft = Softmax(
        layer_name='y',
        #max_col_norm = 1.9365,
        n_classes = nclass,
        #init_bias_target_marginals=DBL.ds_train,
        #istdev = .05
        irange = .0
    )
    layers.append(layer_soft)  
    
    # create DBL_model      
    model = MLP(layers, input_space=ishape)  
    layer_params = cPickle.load(open(pklname))
    layer_id = 0
    for layer in model.layers:
        if layer_id < len(layers) - 1:
            layer.set_weights(layer_params[layer_id][0])
            layer.set_biases(layer_params[layer_id][1])
        else:
            layer.set_weights(layer_params[layer_id][1])
            layer.set_biases(layer_params[layer_id][0])       
        layer_id = layer_id + 1    
    return model

def loadModel2(pklname):

    ishape = Conv2DSpace(
        shape = [48, 48],
        num_channels = 1
        )
    nclass = 7
    # create layers
    nk = [30, 40] # train3040.pkl.cpu
    #nk = [32, 20, 10]
    #nk = [40,30,20]
    ks = [[8,8],[5,5],[3,3]]
    ir = [0.05,0.05,0.05]
    ps = [[4,4],[4,4],[2,2]]
    pd = [[2,2],[2,2],[2,2]]
    kn = [0.9,0.9,0.9]
    layers = DBL_ConvLayers(nk,ks,ir,ps,pd,kn)   
    layer_soft = Softmax(
        layer_name='y',
        #max_col_norm = 1.9365,
        n_classes = nclass,
        #init_bias_target_marginals=DBL.ds_train,
        #istdev = .05
        irange = .0
    )
    #layers.append(layer_soft)  
    
    # create DBL_model      
    model = MLP(layers, input_space=ishape)  
    layer_params = cPickle.load(open(pklname))
    layer_id = 0
    for layer in model.layers:
        if layer_id < len(layers) - 1:
            layer.set_weights(layer_params[layer_id][0])
            layer.set_biases(layer_params[layer_id][1])
            layer_id = layer_id + 1    
    return model

def readWholeImg(imgname):
    curFrame = cv.LoadImage(imgname, 1)      
    gray = cv.CreateImage((curFrame.width,curFrame.height), 8, 1)   
    cv.CvtColor(curFrame, gray, cv.CV_BGR2GRAY)
    img48 = cv.CreateImage((48, 48), 8, 1)
    cv.Resize(gray, img48, cv.CV_INTER_LINEAR)
    cv.EqualizeHist(img48, img48)
    face_vector = np.asarray(img48[:,:])
    face_vector = face_vector.reshape(48*48)
    return [[face_vector], curFrame]

def pipelineImg():
    tagEmotion = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    print 'load facedetector...'
    faceXMLpath =  "/data/vision/fisher/data1/face_celebrity/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)
    
    #img_folder = "/scratch/face/thumbnails_features_deduped_publish/thumbnails_features_deduped_publish/bill clinton/"
    #img_folder = "/data/vision/billf/manifold-learning/DL/Data/pubfig/images/"
    img_folder = "/data/vision/fisher/data1/face_celebrity/thumbnails_features_deduped_publish/thumbnails_features_deduped_publish/bill gates/"
    imgs = os.listdir(img_folder)
    print len(imgs)

    ## reconstruct model
    print "loading trained deep learning model..."
    
    """ old way to load model
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
            if face_set[0]!=0:
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
    """
    
    
    #model = loadModel('train3040.pkl.cpu')
    model = loadModel('train322010.pkl.cpu')

    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    from theano import tensor as T
    y = T.argmax(Y, axis=1)
    from theano import function
    f = function([X], y)

    for i in range(len(imgs)):
        input_name = img_folder + imgs[i]
        
        if input_name.endswith('jpg'):
            face_set, img_rectangle = myDetector.detectImg(input_name)
            #face_set, img_rectangle = readWholeImg(input_name)
            
            if face_set!=None:
                xx = np.array(face_set)
                x_500 = np.concatenate((xx, np.zeros((499, xx.shape[1]),
                dtype=xx.dtype)), axis=0)                

                x_input = DataPylearn2([x_500, np.zeros(500)], (48,48,1))
                x_arg = x_input.X
                if X.ndim > 2:
                   x_arg = x_input.get_topological_view(x_arg)               
                y_pred = f(x_arg.astype(X.dtype))
                #print y_pred.shape
                #print y_pred[0]
                
                print y_pred[0]
                print imgs[i], tagEmotion[y_pred[0]]
                cv.ShowImage("result", img_rectangle)
                cv.WaitKey(0)
                #if cv.WaitKey(10) >= 0:
                #    break  
                
def pipelineCam():
    tagEmotion = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    print 'load facedetector...'
    faceXMLpath =  "/data/vision/fisher/data1/face_celebrity/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)

    ## reconstruct model
    print "loading trained deep learning model..."
  
    model = loadModel('train3040.pkl.cpu')
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    from theano import tensor as T
    y = T.argmax(Y, axis=1)
    from theano import function
    f = function([X], y)


    capture = myDetector.openCam()
    if capture:
        while 1:
            
            face_set, img_rectangle = myDetector.retrieveCam()
            print img_rectangle
            if face_set!=None:
                xx = np.array(face_set)
                x_500 = np.concatenate((xx, np.zeros((499, xx.shape[1]),
                dtype=xx.dtype)), axis=0)                

                x_input = DataPylearn2([x_500, np.ones(500)], (48,48,1))
                x_arg = x_input.X
                if X.ndim > 2:
                   x_arg = x_input.get_topological_view(x_arg)               
                y_pred = f(x_arg.astype(X.dtype))
                
                print tagEmotion[y_pred[0]]

            cv.ShowImage("result", img_rectangle)
            print img_rectangle.shape
            cv.WaitKey()
            if cv.WaitKey(10) >= 0:
                break      


    

                
if __name__ == "__main__": 
    pipelineImg()
    #pipelineCam()




