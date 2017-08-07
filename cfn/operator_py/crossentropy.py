import mxnet as mx
import numpy as np
import cv2
import os
import sklearn.neighbors as nn
import math

#def Softmax(in_data, out_data):
    

class CrossEntropyOperator(mx.operator.CustomOp):
    def __init__(self):
        super(CrossEntropyOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0].asnumpy()
        gt = in_data[1].asnumpy()
        count, channel, height, width = pred.shape
        loss = gt[np.where(gt > 0)] * (np.log(pred[np.where(gt > 0)]+1.0e-35) - np.log(gt[np.where(gt > 0)]) )
        Loss = (loss * (-1)).sum()
        Loss = Loss / count
        '''
        for c in range(count):
            for chan in range(channel):
                for h in range(height):
                    for w in range(width):
                        if gt[c,chan, h,w] > 0:
                            Loss -= gt[c, chan, h, w] * (math.log(pred[c,chan,h,w] + 1.0e-35) - math.log(gt[c,chan,h,w])) 
        #Loss = -loss.sum()
        '''
        self.assign(out_data[0], 'write', Loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        n = in_data[0][0]
        target = in_data[1].asnumpy()
        output_data = in_data[0].asnumpy()
        count, channel, height, width = output_data.shape
        # keep the type in nd.array to get better performance
        # Gradient is target - output_data
        diff = output_data - target
        diff = diff * (1.0 / count)
        self.assign(in_grad[0], 'write', mx.nd.array(diff))
        self.assign(in_grad[1], 'write', 0)

@mx.operator.register('CrossEntropy')
class CrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CrossEntropyProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data', 'label']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        N = in_shape[0][0]
        C = in_shape[0][1]
        H = in_shape[0][2]
        W = in_shape[0][3]
        return [(N,C,H,W), (N,C,H,W)], [(1,)]
    def create_operator(self, ctx, shapes, dtypes):
        return CrossEntropyOperator()
