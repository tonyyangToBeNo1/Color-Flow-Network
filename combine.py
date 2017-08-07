import mxnet as mx
from lib.utils.load_model import load_param
import sys
sys.path.append('CFN')
import CFNet
import numpy as np
import caffe
#caffe.set_mode_gpu()
#caffe.set_device(2)
#args, auxs = load_param('output/colornet/CFN', 1)
#args, auxs = load_param('output/CFNpretrain', 0)
sym, args, auxs = mx.model.load_checkpoint('output/colornet/CFN', 1)
arg_flow, aux_flow = load_param('model/pretrained_model/flownet', 0)
sym = CFNet.CFNet().get_train_symbol()
#prototxt = '../colorization/flownet-release/models/flownet/model_simple/deploy.tpl.prototxt'
#model = '../colorization/flownet-release/models/flownet/model_simple/flownet_official.caffemodel'
#net = caffe.Net(prototxt, model, caffe.TEST)

'''
for item in net.params.items():
    name, layer = item
    num = 0
    for p in net.params[name]:
        np.save('flownet_weights/'+str(name)+'_'+str(num), p.data)
        num += 1
'''

for k,v in args.items():
    print k
    #if 'conv' in k or 'Conv' in k:
    #     args['flow_'+k] = v

#for k, v in aux_flow.items():
#    if 'conv' in k or 'Conv' in k:
#        auxs['flow_'+k] = v

#mod = mx.module.Module(sym, data_names=['data', 'data_cur'], label_names = None)
#mod.set_params(arg_params=args, aux_params=auxs)
#mod.save_params('CFNpretrain')

