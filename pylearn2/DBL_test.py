# emotion dataset
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python DBL_test.py
# 

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

from DBL_layer import DBL_ConvLayers
from DBL_model import DBL_model
import os
import cPickle
import numpy as np

# convolutional_net.yaml baseline
def DBL_model_test1(basepath,cutoff=[-1,-1],pklname='',newdata=None):

    # data
    ishape = Conv2DSpace(
            shape = [48, 48],
            num_channels = 1
            )
    preproc=[0,0]
    nclass = 7
    
    DBL = DBL_model(basepath,nclass,np.append(ishape.shape,1),preproc,cutoff)
        
    # create layers
    nk = [30]
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
        init_bias_target_marginals=DBL.ds_train,
        #istdev = .05
        irange = .0
    )
    layers.append(layer_soft)  
    
    # create DBL_model      
    model = MLP(layers, input_space=ishape)
  
    if pklname!='' and os.path.isfile(pklname):
    
        # load and rebuild model
        layer_params = cPickle.load(open(pklname + '.cpu'))
        layer_id = 0
        for layer in model.layers:
            if layer_id < len(layers) - 1:
                layer.set_weights(layer_params[layer_id][0])
                layer.set_biases(layer_params[layer_id][1])
            else:
                layer.set_weights(layer_params[layer_id][1])
                layer.set_biases(layer_params[layer_id][0])
            
            layer_id = layer_id + 1
        
        DBL.model = model                
        DBL.test_raw(newdata)
        
    else:
                
        algo_term = EpochCounter(500) # number of epoch iteration
        algo = SGD(learning_rate = 0.001,
                batch_size = 500,
                init_momentum = .5,
                monitoring_dataset = DBL.ds_valid,
                termination_criterion=algo_term
                )
        DBL.run_model(model,algo)
        
        # save the model
        if pklname!='':
            layer_params = []
            for layer in layers:
                param = layer.get_params()      
                print param
                print param[0].get_value()
                layer_params.append([param[0].get_value(), param[1].get_value()])
                
            #cPickle.dump(DBL,open(pklname, 'wb'))
            #cPickle.dump(layer_params, open(pklname + '.cpu', 'wb'))
            cPickle.dump(layer_params, open(pklname + '.cpu', 'wb'))

        print DBL.result_valid[1], DBL.result_test[1]
    return DBL.result_valid[1], DBL.result_test[1]

if __name__ == "__main__": 
    DD = '/home/Stephen/Desktop/Data/Classification/icml_2013_emotions/'
    T1_v,T1_t = DBL_model_test1(DD,[-1,-1],'')    
    """
    DD = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions/'
    T1_v,T1_t = DBL_model_test1(DD,[-1,-1],'train30.pkl')
    """
    
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
