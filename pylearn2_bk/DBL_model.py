from DBL_util import *
import numpy as np
class DBL_model(object):
    def __init__(self,basepath,numclass,ishape,preproc=[0,0],cutoff=[-1,-1]): 
        # 1. data 
        self.numclass = numclass
        self.ishape = ishape
        self.preprocess(basepath,preproc[0],preproc[1],cutoff)

    def run_model(self,model,algo):
        # 2. model/algo
        self.model = model
        self.algo  = algo
        # 3. train/test
        self.algo.setup(self.model, self.ds_train)
        self.train()
        self.result_valid = self.test(self.ds_valid)
        self.result_test = self.test(self.ds_test)

    def preprocess(self,basepath,train_pre=0,flip=3,cutoff=[-1,-1]):
        """
        myDataset = LoadData(basepath)
        datasets = myDataset.loadTrain(preprocessFLAG=train_pre,flipFLAG=flip,cutFLAG=cutoff[0])
        self.ds_train = DataPylearn2(datasets[0],self.ishape,self.numclass)
        self.ds_valid = DataPylearn2(datasets[1],self.ishape)
        self.ds_test = DataPylearn2(myDataset.loadTest(train_pre,cutoff[1]),self.ishape)
        """

        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)
        
        
        self.ds_train = EmotionsDataset(which_set='train',base_path=basepath,start=0,stop=10000,preprocessor=pre, trainindex=1)        
        self.ds_valid = EmotionsDataset(which_set='train',base_path=basepath,start=10000,stop=15000,preprocessor=pre)        
        #self.ds_test = EmotionsDataset(which_set='public_test')        

        myDataset = LoadData(basepath)
        self.ds_test = DataPylearn2(myDataset.loadTest(train_pre,cutoff[1]),self.ishape)
    def train(self):
        while True:
            self.algo.train(dataset = self.ds_train)
            self.model.monitor.report_epoch()            
            self.model.monitor()
            """
            # hack the monitor
            print "monior:\n"
            self.test(self.ds_valid)
            """
            if not self.algo.continue_learning(self.model):
                break
    def test_raw(self,ds_raw):
        if ds_raw==None:
            ds = self.ds_test
        else:
            ds = DataPylearn2(ds_raw,self.ishape,-1)
        return self.test(ds)

    
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


