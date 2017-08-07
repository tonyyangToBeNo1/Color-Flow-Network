import sys
sys.path.append('symbol/')
from cfn.operator_py.rgb2lab import *
from cfn.operator_py.priorboost import *
from cfn.core import callback, metric
from cfn.core import load_model

import CFNet
import imageIter
#from utils.create_logger import create_logger
import mxnet as mx
import logging
import cv2

os.environ['PYTHONUNBUFFERED']='1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='1'
os.environ['MXNET_ENABLE_GPU_P2P']='1'


def summary(symbol, shape=None):
    internals = symbol.get_internals()
    _, out_shapes,_ = internals.infer_shape(data=shape)
    #print internals.list_outputs
    #print out_shapes
    name = internals.list_outputs()
    layer = open('layer.txt', 'w')
    for i in range(len(name)):
        print name[i], out_shapes[i]
        layer.write(name[i] + ' ' + str(out_shapes[i]) + '\n')

def main():
    Data_root = "ILSVRC2015/Data/VID/train/"
    source_root = "ILSVRC2015/ImageSets/Videos/train.txt"
    batch_size = 90
    output_path = 'output/colornet/CFN_all'
    GPU = 0
    imageiter = imageIter.ImageIter(batch_size=batch_size, source_root=source_root, Data_root=Data_root)
    sym_instance = CFNet.CFNet()
    sym = sym_instance.get_train_symbol()
    #mx.visualization.plot_network(sym).view()
    summary(sym, shape=(batch_size,3,176,176))
    # load param
    #color_pretrain = './model/pretrained_model/colornet'
    color_pretrain = 'output/colornet/CFNwarp100'
    epoch = 0
    arg_params, aux_params = load_model.load_param(color_pretrain, epoch)
    fixed_param_names = ['bw_conv1_1_weight', 'bw_conv1_1_bias',
                        'conv1_2_weight','conv1_2_bias', 'conv2_1_weight', 'conv2_2_bias',
                        'conv2_2_weight', 'conv2_2_bias', 'conv3_1_weight', 'conv3_1_bias',
                        'conv3_2_weight', 'conv3_2_bias', 'conv3_3_weight', 'conv3_3_bias',
                        'conv4_1_weight', 'conv4_1_bias', 'conv4_2_weight', 'conv4_2_bias',
                        'conv4_3_weight', 'conv4_3_bias', 'conv5_1_weight', 'conv5_1_bias',
                        'conv5_2_weight', 'conv5_2_bias', 'conv5_3_weight', 'conv5_3_bias',
                        'conv6_1_weight', 'conv6_1_bias', 'conv6_2_weight', 'conv6_2_bias',
                        'conv6_3_weight', 'conv6_3_bias', 'conv7_1_weight', 'conv7_1_bias', 
                        'conv7_2_weight', 'conv7_2_bias', 'conv7_3_weight', 'conv7_3_bias',
                        'conv8_1_weight', 'conv8_2_weight', 'conv8_2_bias', 
                        'conv8_3_weight', 'conv8_3_bias', 'conv8_313_weight', 
                        'data_ab_ss_weight', 'data_ab_ss_bias',
                        'conv1_2norm_beta', 'conv1_2norm_gamma',
                        'conv2_2norm_beta', 'conv2_2norm_gamma',
                        'conv3_3norm_beta', 'conv3_3norm_gamma',
                        'conv4_3norm_beta', 'conv4_3norm_gamma',
                        'conv5_3norm_beta', 'conv5_3norm_gamma',
                        'conv6_3norm_beta', 'conv6_3norm_gamma',
                        'conv7_3norm_beta', 'conv7_3norm_gamma']
    #fixed_param_names = ['data_ab_ss', 'data_ab_ss_cur']
    sym_instance.init_weight(arg_params, aux_params)
    # metric
    color_metric = metric.ColorMetric()
    flow_metric = metric.FlowMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    
    for child_metric in [color_metric, flow_metric]:
        eval_metrics.add(child_metric)
    
    batch_end_callback = callback.Speedometer(batch_size=1, frequent=1)
    optimizer = mx.optimizer.Adam(learning_rate = 3.16e-5, beta1 = 0.9, beta2 = 0.99, rescale_grad=1.0 / batch_size)
    data_names=['data']
    model=mx.module.Module(context=mx.gpu(GPU), symbol=sym, data_names=data_names, label_names=None)
    checkpoint = mx.callback.module_checkpoint(model, output_path, period=1, save_optimizer_states=False)
    print 'training the data'
    
    model.fit(imageiter, begin_epoch=0, num_epoch=10, \
              batch_end_callback = batch_end_callback, \
              epoch_end_callback = checkpoint, \
              arg_params=arg_params, aux_params=aux_params, \
              optimizer=optimizer, \
              kvstore='local', \
              eval_metric=eval_metrics)
    
if __name__ == '__main__':
    main()
