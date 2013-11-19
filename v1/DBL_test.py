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

def DBL_model_test1(basepath,cutoff=[-1,-1],pklname='',newdata=None):
    # convolutional_net.yaml baseline
    if pklname!='' and os.path.isfile(pklname):
        DBL = cPickle.load(open(pklname))
        DBL.test_raw(newdata)
    else:
        # data 
        ishape = Conv2DSpace(
                shape = [48, 48],
                num_channels = 1
                )
        preproc=[0,0]
        nclass = 7
        DBL = DBL_model(basepath,nclass,np.append(ishape.shape,1),preproc,cutoff)

        # create layers
        nk = [32]
        #nk = [40,30,20]
        ks = [[8,8],[8,8],[3,3]]
        ir = [0.05,0.05,0.05]
        ps = [[4,4],[4,4],[2,2]]
        pd = [[2,2],[2,2],[2,2]]
        kn = [0.9,0.9,0.9]
        layers = DBL_ConvLayers(nk,ks,ir,ps,pd,kn)
        layer_soft = Softmax(
            layer_name='y',
            #max_col_norm = 1.9365,
            n_classes = nclass,
            init_bias_target_marginals=DBL.ds_train,
            #istdev = .05
            irange = .0
        )
        layers.append(layer_soft)
        
        # create DBL_model
        #model = MLP(layer_soft, input_space=ishape)
        model = MLP(layers, input_space=ishape)
        from pylearn2.termination_criteria import MonitorBased
        algo_term = MonitorBased(            
                channel_name = "y_misclass",
                prop_decrease = 0.,
                N = 10)
        algo = SGD(learning_rate = 0.001,
                batch_size = 100,
                init_momentum = .5,
                monitoring_dataset = DBL.ds_valid,
                termination_criterion=algo_term
                )
        DBL.run_model(model,algo)
        if pklname!='':
            DBL.ds_train = []
            DBL.ds_test = []
            DBL.ds_valid = []
            cPickle.dump(DBL,open(pklname, 'wb'))

    return DBL.result_valid,DBL.result_test

if __name__ == "__main__": 
    DD = '/home/Stephen/Desktop/Bird/DLearn/Data/icml_2013_emotions/'
    #DD = '/afs/csail.mit.edu/u/b/bzhou/data/faceexpression/fer2013/'
    
    data = None
    T1_v,T1_t = DBL_model_test1(DD,[1500,500],DD+'train.pkl')
    print T1_v[1]
    print T1_t[1]
    """
    import cPickle
    FF = '/home/Stephen/Desktop/Bird/DLearn/Data/Emotion/train.pkl'
    a=cPickle.load(open(FF))
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
