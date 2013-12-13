import numpy as np
import cPickle
import os
import theano
import theano.tensor as T
# define constants
EMOTION='/home/Stephen/Desktop/Bird/DLearn/Data/Emotion/'
MNIST='/home/Stephen/Desktop/Bird/DLearn/Data/'
VALID = 0.1
def load(dataset):
    if dataset=='emotion':
        if not os.path.isfile(EMOTION+"train_x.csv"):
            Preprocess(dataset)
        train_set_xall = np.loadtxt(open(EMOTION+"train_x.csv","rb"),delimiter=" ",skiprows=0) 
        train_set_yall = np.loadtxt(open(EMOTION+"train_y.csv","rb"),delimiter=" ",skiprows=0) 
        num_train = len(train_set_yall)
        valid_sz = int(VALID*num_train)
        train_set_x = train_set_xall[valid_sz:,:]
        valid_set_x = train_set_xall[:valid_sz,:]
        train_set_y = train_set_yall[valid_sz:]
        valid_set_y = train_set_yall[:valid_sz]
        test_set_x = np.loadtxt(open(EMOTION+"test_pub.csv","rb"),delimiter=" ",skiprows=0) 
        test_set_y = np.loadtxt(open(EMOTION+"test_pub_truth.csv","rb"),delimiter=" ",skiprows=0) 
        return [[train_set_x,train_set_y],[valid_set_x,valid_set_y],[test_set_x,test_set_y]]
    elif dataset=='mnist':
       return cPickle.load(open(MNIST+'mnist.pkl','rb'))

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_share(dataset):
    datasets = load(dataset)
    for i in range(3):
        datasets[i] = shared_dataset(datasets[i])
    return datasets


def Preprocess(dataset):
    if dataset=='emotion':
        a=open(EMOTION+"train.csv","rb")
        b=open(EMOTION+"train_x.csv","w")
        c=open(EMOTION+"train_y.csv","w")
        # skip the first line
        line = a.readline()
        while line:
            line = a.readline()
            if len(line)>1:
                lines= line.split(',')
                c.write(lines[0]+'\n')
                b.write(lines[1][1:-3]+'\n')

if __name__ == '__main__':
    Preprocess('emotion')
