from random import randint
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np

class XOR(DenseDesignMatrix):
    def __init__(self):
        self.class_names = ['0', '1']
        X = [[randint(0, 1), randint(0, 1)] for _ in range(1000)]
        y = []
        for a, b in X:
            if a + b == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])
        X = np.array(X)
        y = np.array(y)
        super(XOR, self).__init__(X=X, y=y)
 
ds = XOR()
