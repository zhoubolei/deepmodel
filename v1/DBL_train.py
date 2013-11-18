from pylearn2 import train
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.energy_functions.rbm_energy import grbm_type_1
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.corruption import GaussianCorruptor
from pylearn2.termination_criteria import MonitorBased

from DBL_util import Dataset
from DBL_util import Dataset2

def runDeepLearning2():
### Loading training set and separting it into training set and testing set

    myDataset = Dataset('/home/Stephen/Desktop/Bird/DLearn/Data/Emotion_small/')
    preprocess = 0
    datasets = myDataset.loadTrain(preprocessFLAG=preprocess,flipFLAG=3)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    dataset_test = myDataset.loadTest(preprocess)
    test_set_x, test_set_y, test_set_y_array = dataset_test[0]
    # temporary solution to get the ground truth of sample out to test_set_y_array.
    # the reason is that after T.cast, test_set_y becomes TensorVariable, which I do not find way to output its
    # value...anyone can help?

### Model parameterso
    """
    learning_rate = 0.02
    n_epochs = 3000
    nkerns=[30, 40, 40] # number of kernal at each layer, current best performance is 50.0% on testing set, kernal number is [30,40,40]
    batch_size = 500

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
    """
    rng = np.random.RandomState(23455)


    # Import yaml file that specifies the model to train
    # conv layer
    layer0 = ConvRectifiedLinear(
            layer_name = 'h2',
            output_channels = 64,
            irange = .05,
            kernel_shape = [8, 8],
            pool_shape = [4, 4],
            pool_stride = [2, 2],
            max_kernel_norm = 0.9
            )
    # mlp
    layer2 = RectifiedLinear(
            layer_name = 'h1',
            dim = 1000,
            sparse_init = 15
    )


    # softmax
    layer3 = Softmax(
            max_col_norm = 1.9365,
            layer_name = 'y',
            n_classes = 7,
            istdev = .05
    )
    ds = Dataset2(train_set_x,train_set_y)
    layers = [layer0, layer2, layer3]
    ann = mlp.MLP(layers, nvis=3)
    t_algo = SGD(learning_rate = 1e-1,
            batch_size = 500,
            termination_criterion=EpochCounter(400)
            )
         
    t_algo.setup(ann, ds)
           
    while True:
        trainer.train(dataset=ds)
        ann.monitor.report_epoch()
        ann.monitor()
        if not trainer.continue_learning(ann):
            break

    #test_data = np.array([[0, 1]])
    #t_train.model.fprop(theano.shared(test_data, name='test_data')).eval()
if __name__ == '__main__':    
    runDeepLearning2()
