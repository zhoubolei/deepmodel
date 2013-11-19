# emotion dataset

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

from DBL_layer import DBL_ConvLayers
from DBL_model import DBL_model
import os
import cPickle
import numpy as np


import cPickle
from theano import tensor as T


if __name__ == "__main__": 
    #DD = '/home/Stephen/Desktop/Bird/DLearn/Data/icml_2013_emotions/'
    DD = '/afs/csail.mit.edu/u/b/bzhou/data/faceexpression/fer2013/'
    
    #pretrainModel = '/afs/csail.mit.edu/u/b/bzhou/code/deepmodel/v1/train.pkl'
    pretrainModel = '/afs/csail.mit.edu/u/b/bzhou/code/deepmodel/v1/mlp_cpu.pkl'
    #data = None
    #T1_v,T1_t = DBL_model_test1(DD,[-1,-1],pretrainModel)
    #print T1_v[1]
    #print T1_t[1]
    
  
    a=cPickle.load(open(pretrainModel))
    X = a.model.get_input_space().make_batch_theano()
    Y = a.model.fprop(X)
    y = T.argmax(Y, axis=1)
    f = function([X], y)
    f2 = function([X], Y)
    xx = a.ds_test.get_topological_view(a.ds_test.X)
    f(xx)
    f2(xx)
    """
    X = a.model.get_input_space().make_batch_theano()
    Y = a.model.fprop(X)
    from theano import tensor as T
    y = T.argmax(Y, axis=1)
    from theano import function
    f = function([X], y)
    f2 = function([X], Y)
    xx = a.ds_test.get_topological_view(a.ds_test.X)
    f(xx)
    f2(xx)
    # show weight
    python /home/Stephen/Desktop/Bird/DLearn/Dis/pylearn2/pylearn2/scripts/show_weights.py /home/Stephen/Desktop/Bird/DLearn/Dis/deepmodel/v1/ha2.pkl 
    mm=a.model
    cPickle.dump(mm,open('ha2.pkl','wb'))

    python /home/Stephen/Desktop/Bird/DLearn/Dis/pylearn2/pylearn2/scripts/plot_monitor.py /home/Stephen/Desktop/Bird/DLearn/Dis/deepmodel/v1/ha2.pkl 


THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python DBL_test.py
    """
