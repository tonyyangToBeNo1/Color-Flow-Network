import mxnet as mx
import numpy as np
import cv2
import os
from cfn.operator_py.additional import *


class ClassRebalanceMultOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ClassRebalanceMultOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        #print in_data[0].asnumpy()[0,0,0,:]
        self.assign(out_data[0], 'write', in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #in_diff = in_grad[0].asnumpy()
        out_diff = out_grad[0].asnumpy()
        data = in_data[1].asnumpy()
        #print 'class rebalance'
        #print out_diff[0,0,0,:]
        #print data[0,0,0,:]
        #x = input()
        in_diff = out_diff[...] * data[...]
        #in_gradient = out_grad[0].asnumpy() * in_data[1].asnumpy()
        
        self.assign(in_grad[0], 'write', mx.nd.array(in_diff))
        self.assign(in_grad[1], 'write', 0)


@mx.operator.register('ClassRebalanceMult')
class ClassRebalanceMultProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ClassRebalanceMultProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data1', 'data2']
    
    def list_outputs(self):
        return ['CalssRebalanceMult']

    def infer_shape(self, in_shape):
        return in_shape, [(in_shape[0])]

    def create_operator(self, ctx, shapes, dtypes):
        return ClassRebalanceMultOperator()
