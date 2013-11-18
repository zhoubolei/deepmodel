# emotion dataset

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

from DBL_layer import DBL_ConvLayers
from DBL_model import DBL_model
import os
import cPickle
def DBL_model_test1(basepath,pklname='',newdata=None):
    if pklname!='' and os.path.isfile(pklname):
        if newdata!=None:
            DBL = cPickle.load(open(pklname))
            DBL.test_raw(newdata)
    else:
        # data 
        ishape = Conv2DSpace(
                shape = [48, 48],
                num_channels = 1
                )
        preproc=[0,3]
        # create layers
        nk = [40,40,40]
        ks = [[5,5],[5,5],[5,5]]
        ir = [0.05,0.05,0.05]
        ps = [[2,2],[2,2],[2,2]]
        pd = [[2,2],[2,2],[2,2]]
        kn = [0.9,0.9,0.9]
        layers = DBL_ConvLayers(nk,ks,ir,ps,pd,kn)
        layer_soft = Softmax(
            layer_name='y',
            max_col_norm = 1.9365,
            n_classes = 7,
            istdev = .05
        )
        layers.append(layer_soft)
        
        model = MLP(layers, input_space=ishape)
        algo = SGD(learning_rate = 1e-1,
                batch_size = 99,
                batches_per_iter = 1,
                termination_criterion=EpochCounter(2)
                )

        # create DBL_model
        DBL = DBL_model(model,algo,basepath,preproc)
        if pklname!='':
            cPickle.dump(DBL,open(pklname, 'wb'))

    return DBL.result_valid,DBL.result_test

if __name__ == "__main__": 
    DD = '/home/Stephen/Desktop/Bird/DLearn/Data/Emotion_small/'
    data = None
    T1_v,T1_t = DBL_model_test1(DD)
    print T1_v[1]
    print T1_t[1]
