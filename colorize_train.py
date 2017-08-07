import os,sys
sys.path.append('symbol')
from cfn.core import callback, metric
from cfn.core import load_model
#from utils.create_logger import create_logger
import color_flow
import imageIter

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
    batch_size = 100
    GPU=1
    output_path = 'output/colornet/frame_all'
    imageiter = imageIter.ImageIter(batch_size=batch_size, source_root=source_root, Data_root=Data_root)
    sym_instance = color_flow.color_flow()
    sym = sym_instance.get_train_symbol()
    #feat_sym = sym.get_internals()
    #feat_sym = sym.get_internals('prior_boost_nongray_output')
    summary(sym, shape=(imageiter.k,3,176,176))
    # load param
    color_pretrain = './model/pretrained_model/colornet'
    #color_pretrain = 'output/colornet/colornet100'
    epoch = 0
    arg_params, aux_params = load_model.load_param(color_pretrain, epoch)
    #arg_params=None
    #aux_params=None
    sym_instance.init_weight(arg_params, aux_params)
    # metric
    color_metric = metric.ColorMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    
    for child_metric in [color_metric]:
        eval_metrics.add(child_metric)
    
    batch_end_callback = callback.Speedometer(batch_size=1, frequent=10)

    checkpoint = mx.callback.do_checkpoint('output/colornet/colornet_all', 1)
    optimizer = mx.optimizer.Adam(learning_rate = 3.16e-5, beta1 = 0.9, beta2 = 0.99, rescale_grad=1.0 / imageiter.k)
    model = mx.module.Module(context=mx.gpu(0), symbol=sym, label_names=None)
    print 'training the data'
    model.fit(imageiter, begin_epoch=0, num_epoch=3, \
              batch_end_callback = batch_end_callback, \
              epoch_end_callback = checkpoint, \
              arg_params=arg_params, aux_params=aux_params, \
              optimizer=optimizer, \
              kvstore='local', \
              eval_metric=eval_metrics)#, \
              #allow_missing=True, monitor=mon)
    
    #model.save_params('output/color-small-0000.params')
    
if __name__ == '__main__':
    main()
