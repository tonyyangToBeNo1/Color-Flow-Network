import mxnet as mx
import numpy as np
import cv2
import os
from cfn.operator_py.additional import *

class PriorBoostOperator(mx.operator.CustomOp):
    def __init__(self, pc):
        super(PriorBoostOperator, self).__init__()
        self.pc = pc

    def forward(self, is_train, req, in_data, out_data, aux):
        output = self.pc.forward(in_data[0].asnumpy(), axis=1)
        #print 'prior'
        #print output[0,0,0,:]
        self.assign(out_data[0], req[0], mx.nd.array(output))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[1], 0)
        
@mx.operator.register('priorboost')
class PriorBoostProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(PriorBoostProp, self).__init__(False)
        self.ENC_DIR = '/media/Disk/ziyang/code/colorization/resources/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['priorboost']

    def infer_shape(self, in_shape):
        self.N = in_shape[0][0]
        self.X = in_shape[0][2]
        self.Y = in_shape[0][3]
        return in_shape, [(self.N, 1, self.X, self.Y)]

    def create_operator(self, ctx, shapes, dtypes):
        return PriorBoostOperator(self.pc)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

