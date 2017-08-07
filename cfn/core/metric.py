import mxnet as mx
import numpy as np
import math
def get_color_names():
    #pred = ['softmax_conv_feat']
    #label = ['gt313']
    label = ['loss8_313_output']
    return label

class ColorMetric(mx.metric.EvalMetric):
    def __init__(self):
        #print "eval the colornet"
        super(ColorMetric, self).__init__('ColorLoss')
    
    def update(self, labels=None, preds=None):
        LOSS_key = labels[0].asnumpy()
        self.sum_metric += LOSS_key
        self.num_inst += 1

class FlowMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FlowMetric, self).__init__('FlowLoss')
    def update(self, labels=None, preds=None):
        LOSS_cur = labels[1].asnumpy()
        self.sum_metric += LOSS_cur
        self.num_inst += 1

