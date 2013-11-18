from pylearn2 import train
from pylearn2.models.mlp import ConvRectifiedLinear,RectifiedLinear,Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import EpochCounter

from DBL_util import LoadData
from DBL_util import DataPylearn2
import numpy as np

#%load_ext autoreload
#%autoreload 2

# 1. Data
myLoadData = LoadData('/home/Stephen/Desktop/Bird/DLearn/Data/Emotion_small/')
preprocess = 0
datasets = myLoadData.loadTrain(preprocessFLAG=preprocess,flipFLAG=3)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

dataset_test = myLoadData.loadTest(preprocess)
test_set_x, test_set_y = dataset_test[0]
ishape = Conv2DSpace(
        shape = [48, 48],
        num_channels = 1
        )
rng = np.random.RandomState(23455)

# 2. Model
# conv layer
layer0 = ConvRectifiedLinear(
        layer_name = 'h1',
        output_channels = 30,
        irange = .05,
        kernel_shape = [5, 5],
        pool_shape = [2, 2],
        pool_stride = [2, 2],
        max_kernel_norm = 0.9
        )
layer1 = ConvRectifiedLinear(
        layer_name = 'h2',
        output_channels = 40,
        irange = .05,
        kernel_shape = [8, 8],
        pool_shape = [4, 4],
        pool_stride = [2, 2],
        max_kernel_norm = 0.9
        )
# mlp
"""
layer2 = RectifiedLinear(
        layer_name = 'h3',
        dim = 49,
        sparse_init = 15
)
"""

# softmax
layer3 = Softmax(
        max_col_norm = 1.9365,
        layer_name = 'y',
        n_classes = 7,
        istdev = .05
)
layers = [layer0, layer1, layer3]
#layers = [layer0, layer2, layer3]
ann = MLP(layers, input_space=ishape)
t_algo = SGD(learning_rate = 1e-1,
        batch_size = 99,
        batches_per_iter = 1,
        termination_criterion=EpochCounter(2)
        )
     
ds = DataPylearn2(train_set_x,train_set_y,)
t_algo.setup(ann, ds)
       
while True:
    t_algo.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    if not t_algo.continue_learning(ann):
        break

# test: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/icml_2013_wrepl/emotions/make_submission.py
ds2 = DataPylearn2(test_set_x)
m = ds2.X.shape[0]
batch_size = 99 
extra = (batch_size - m % batch_size) % batch_size
assert (m + extra) % batch_size == 0
if extra > 0:
    ds2.X = np.concatenate((ds2.X, np.zeros((extra, ds2.X.shape[1]),
            dtype=ds2.X.dtype)), axis=0)
    assert ds2.X.shape[0] % batch_size == 0
X = ann.get_input_space().make_batch_theano()
Y = ann.fprop(X)

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
