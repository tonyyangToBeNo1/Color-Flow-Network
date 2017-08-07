import mxnet as mx
import numpy as np
import cv2
from skimage import color
import os
import scipy.ndimage.interpolation as sni

class CombineOperator(mx.operator.CustomOp):
    def __init__(self):
        super(CombineOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
       # the input should be a list of images
       # the output is to combine get the images together and combine them
        in_imgs = in_data[0].asnumpy()
        count = len(in_imgs)
        #concat = np.zeros()
        n,c,h,w = in_imgs.shape
        key = np.zeros((n/2,c,h,w))
        others = np.zeros((n/2, c,h,w))
        i = 0
        for c in range(0,count,2):
            key[i,:,:,:] = in_imgs[c,:,:,:]
            others[i,:,:,:] = in_imgs[c+1,:,:,:]
            i += 1

        self.assign(out_data[0], 'write',mx.nd.array(key))
        self.assign(out_data[1], 'write', mx.nd.array(others))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0],'write', 0)

@mx.operator.register('splitdata')
class CombineProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CombineProp, self).__init__(False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['key', 'others']
    def infer_shape(self, input_shape):
        N = input_shape[0][0] / 2
        C = input_shape[0][1]
        H = input_shape[0][2]
        W = input_shape[0][3]
        return input_shape, [(N,C,H,W),(N,C,H,W)], []
    def create_operator(self, ctx, shapes, dtypes):
        return CombineOperator()

class One2ThreeOperator(mx.operator.CustomOp):
    def __init__(self):
        super(One2ThreeOperator, self).__init__()
    
    def forward(self, is_train, req, in_data, out_data, aux):
        data_content = in_data[0]
        data_concat = mx.nd.concat(data_content, data_content, data_content)
        self.assign(out_data[0], 'write', data_concat)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'write', 0)

@mx.operator.register('One2Three')
class One2ThreeProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(One2ThreeProp, self).__init__(False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['One2Three']
    def infer_shape(self, input_shape):
        N = input_shape[0][0]
        C = input_shape[0][1]
        H = input_shape[0][2]
        W = input_shape[0][3]
        return input_shape, [(N,C,H,W)], []
    def create_operator(self, ctx, shapes, dtypes):
        return One2ThreeOperator()

class RGB2LABOperator(mx.operator.CustomOp):
    def __init__(self):
        super(RGB2LABOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        output = color.rgb2lab(in_data[0].asnumpy()[:,::-1,:,:].astype('uint8').transpose((2,3,0,1))).transpose((2,3,0,1))
        self.assign(out_data[0], req[0], mx.nd.array(output))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # need no to bp the gradients
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register("rgb2lab")
class RGB2LABProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(RGB2LABProp, self).__init__(False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['data_l']

    def infer_shape(self, in_shape):
        count = in_shape[0][0]
        channel = in_shape[0][1]
        height = in_shape[0][2]
        width = in_shape[0][3]
        data_l_shape = (count, 3, height, width)
        return [(count, channel, height, width)], [data_l_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return RGB2LABOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
