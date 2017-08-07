import mxnet as mx
import numpy as np
import cv2
import os
import sklearn.neighbors as nn
import math
from cfn.operator_py.additional import *

class NonGrayMaskOperator(mx.operator.CustomOp):
    def __init__(self):
        super(NonGrayMaskOperator, self).__init__()
        self.thresh = 5

    def forward(self, is_train, req, in_data, out_data, aux):
        output = np.zeros(out_data[0].shape)
        output[...] = (np.sum(np.sum(np.sum(np.abs(in_data[0].asnumpy()) > self.thresh, axis=1),axis=1),axis=1) > 0)[:,na(),na(),na()]
        #print 'nongray'
        #print output.shape
        #print output[0,0,0,:]
        self.assign(out_data[0], req[0], mx.nd.array(output))

    def backward(self, req, out_grad, in_data, out_data, in_grad,aux):
        self.assign(in_grad[0], 'write', 0)

@mx.operator.register('NonGrayMask')
class NonGrayMaskProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NonGrayMaskProp, self).__init__(False)
    def list_arguments(self):
        return ['data']
    def list_output(self):
        return ['NonGrayMask']
    def infer_shape(self, in_shape):
        self.N = in_shape[0][0]
        self.C = in_shape[0][1]
        self.X = in_shape[0][2]
        self.Y = in_shape[0][3]
        return in_shape, [(self.N, 1, self.X, self.Y)]

    def create_operator(self, ctx, shapes, dtypes):
        return NonGrayMaskOperator()

