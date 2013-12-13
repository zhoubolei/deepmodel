from DBL_data import *
import numpy as np
class DBL_model(object):
    def __init__(self,basepath,numclass,ishape,preproc=[0,0],cutoff=[-1,-1]): 
        # 1. data 
        self.numclass = numclass
        self.ishape = ishape

    def run_model(self,model,algo,ds_train):
        # 2. model/algo
        self.model = model
        self.algo  = algo
        # 3. train/test
        self.algo.setup(self.model, ds_train)
        self.train(ds_train)        

    def train(self,d_train):
        while True:
            #print d_train.X.shape,d_train.y.shape
            self.algo.train(dataset = d_train)
            self.model.monitor.report_epoch()            
            self.model.monitor()
            """
            # hack the monitor
            print "monior:\n"
            self.test(self.ds_valid)
            """
            if not self.algo.continue_learning(self.model):
                break    
    
    def test(self,ds2):
        # https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/icml_2013_wrepl/emotions/make_submission.py
        batch_size = 500 #self.algo.batch_size
        m = ds2.X.shape[0]
        extra = (batch_size - m % batch_size) % batch_size
        #print extra,batch_size,m
        assert (m + extra) % batch_size == 0
        if extra > 0:
            ds2.X = np.concatenate((ds2.X, np.zeros((extra, ds2.X.shape[1]),
                    dtype=ds2.X.dtype)), axis=0)
            assert ds2.X.shape[0] % batch_size == 0
        X = self.model.get_input_space().make_batch_theano()
        Y = self.model.fprop(X)

        from theano import tensor as T
        y = T.argmax(Y, axis=1)
        from theano import function
        f = function([X], y)
        y = []
        for i in xrange(ds2.X.shape[0] / batch_size):
            x_arg = ds2.X[i*batch_size:(i+1)*batch_size,:]
            if X.ndim > 2:
                x_arg = ds2.get_topological_view(x_arg)
            y.append(f(x_arg.astype(X.dtype)))

        y = np.concatenate(y)
        y = y[:m]
        ds2.X = ds2.X[:m,:]
        """
        print y
        print ds2.y
        
        """
        if ds2.y.ndim>1:
            yy = np.argmax(ds2.y,axis=1)
        else:
            yy = ds2.y
        print len(y)
        print len(yy)
        acc = 0
        if len(yy)>0: 
            assert len(y)==len(yy)
            acc = float(np.sum(y-yy==0))/len(yy)
        print acc
        return [[y],[acc]]


