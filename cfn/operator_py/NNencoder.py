import numpy as np
import mxnet as mx
import cv2
import os
import sklearn.neighbors as nn
import time
from cfn.operator_py.additional import *

class NNEncOperator(mx.operator.CustomOp):
    def __init__(self, nnenc): 
        super(NNEncOperator, self).__init__()
        self.nnenc = nnenc
    def forward(self, is_train, req, in_data, out_data, aux):
        output = self.nnenc.encode_points_mtx_nd(in_data[0].asnumpy(), axis=1)
        self.assign(out_data[0], req[0], mx.nd.array(output))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'write', 0)

@mx.operator.register('NNEncoder')
class NNEncProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NNEncProp, self).__init__(need_top_grad=False)
        self.NN = 10.
        self.sigma = 5.
        self.ENC_DIR = '/media/Disk/ziyang/code/colorization/resources/'
        self.nnenc = NNEncode(self.NN, self.sigma, km_filepath=os.path.join(self.ENC_DIR, 'pts_in_hull.npy'))
        self.Q = self.nnenc.K

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['ab_enc']

    def infer_shape(self, in_shape):
        N = in_shape[0][0]
        X = in_shape[0][2]
        Y = in_shape[0][3]
        return in_shape, [(N, self.Q, X, Y)], []
    
    def create_operator(self, ctx, shapes, dtypes):
        return NNEncOperator(self.nnenc)
    
    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
