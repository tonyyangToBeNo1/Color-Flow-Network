from cfn.operator_py.rgb2lab import *
from cfn.operator_py.priorboost import *
from cfn.operator_py.crossentropy import *
from cfn.operator_py.NNencoder import *
from cfn.operator_py.nongray import *
from cfn.operator_py.classrebalance import *
import math

class color_flow():
    def __init__(self):
        self.eps = 1e-4
        self.use_global_stats = True
        self.fix_gamma = True

    def pre_train_data(self, data):
        data_l = mx.symbol.Custom(data=data, op_type='rgb2lab')
        img_l = mx.symbol.slice_axis(name='img_slice', data=data_l, axis=1, begin=0, end=1)
        img_l = img_l - 50

        img_l_ab = mx.symbol.slice_axis(name='img_l_ab_slice', data=data_l, axis=1, begin=1, end=3)
        data_ab_ss = mx.symbol.Convolution(name='data_ab_ss', data=img_l_ab, num_filter=2, kernel=(1,1), stride=(4,4), num_group=2)
        mx.symbol.BlockGrad(data_ab_ss)
        label = data_ab_ss
        gt313 = mx.symbol.Custom(data=data_ab_ss, op_type='NNEncoder')
        nongray_mask = mx.symbol.Custom(name='nongray_mask', data=label, op_type='NonGrayMask')
        #gt313 = mx.symbol.Custom(name='gt313', data=data_ab_ss, op_type='NNEncoder')
        prior_boost = mx.symbol.Custom(name='prior_boost', data=gt313, op_type='priorboost')
        #prior_boost_nongray = mx.symbol.broadcast_mul(lhs=nongray_mask, rhs=prior_boost, name='prior_boost_nongray')
        prior_boost_nongray = nongray_mask * prior_boost
        return img_l, gt313, prior_boost_nongray

    def colornet(self, data):
        conv1_1 = mx.symbol.Convolution(name='bw_conv1_1', data=data, num_filter=64, kernel=(3,3), pad=(1,1), stride=(1,1))
        relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
        conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, kernel=(3,3), pad=(1,1), stride=(2,2))
        relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
        conv1_2norm = mx.symbol.BatchNorm(name='conv1_2norm', data=relu1_2 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv2_1 = mx.symbol.Convolution(name='conv2_1', data=conv1_2norm, num_filter=128, pad=(1,1), kernel=(3,3))
        relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
        conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1,1), kernel=(3,3), stride=(2,2))
        relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
        conv2_2norm = mx.symbol.BatchNorm(name='conv2_2norm', data=relu2_2, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv3_1 = mx.symbol.Convolution(name='conv3_1',data=conv2_2norm,num_filter=256,pad=(1,1),kernel=(3,3), stride=(1,1))
        relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
        conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1,1), kernel=(3,3))
        relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
        conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1,1), kernel=(3,3), stride=(2,2))
        relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
        conv3_3norm = mx.symbol.BatchNorm(name='conv3_3norm', data=relu3_3, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv4_1 = mx.symbol.Convolution(name='conv4_1',data=conv3_3norm,num_filter=512,pad=(1,1),kernel=(3,3), stride=(1,1))
        relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
        conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512,pad=(1,1),kernel=(3,3), stride=(1,1))
        relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2, act_type='relu')
        conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2, num_filter=512,pad=(1,1),kernel=(3,3), stride=(1,1))
        relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3, act_type='relu')
        conv4_3norm = mx.symbol.BatchNorm(name='conv4_3norm', data=relu4_3, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv5_1 = mx.symbol.Convolution(name='conv5_1',data=conv4_3norm,num_filter=512,pad=(2,2),kernel=(3,3), stride=(1,1), dilate=(2,2))
        relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1, act_type='relu')
        conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1, num_filter=512,pad=(2,2),kernel=(3,3), stride=(1,1), dilate=(2,2))
        relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2, act_type='relu')
        conv5_3 = mx.symbol.Convolution(name='conv5_3', data=relu5_2, num_filter=512,pad=(2,2),kernel=(3,3), stride=(1,1), dilate=(2,2))
        relu5_3 = mx.symbol.Activation(name='relu5_3', data=conv5_3, act_type='relu')
        conv5_3norm = mx.symbol.BatchNorm(name='conv5_3norm', data=relu5_3, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv6_1 = mx.symbol.Convolution(name='conv6_1',data=conv5_3norm,num_filter=512,pad=(2,2),kernel=(3,3), dilate=(2,2))
        relu6_1 = mx.symbol.Activation(name='relu6_1', data=conv6_1, act_type='relu')
        conv6_2 = mx.symbol.Convolution(name='conv6_2', data=relu6_1, num_filter=512,pad=(2,2),kernel=(3,3), dilate=(2,2))
        relu6_2 = mx.symbol.Activation(name='relu6_2', data=conv6_2, act_type='relu')
        conv6_3 = mx.symbol.Convolution(name='conv6_3', data=relu6_2, num_filter=512,pad=(2,2),kernel=(3,3), dilate=(2,2))
        relu6_3 = mx.symbol.Activation(name='relu6_3', data=conv6_3, act_type='relu')
        conv6_3norm = mx.symbol.BatchNorm(name='conv6_3norm', data=relu6_3, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv7_1 = mx.symbol.Convolution(name='conv7_1',data=conv6_3norm,num_filter=512, pad=(1,1),kernel=(3,3), dilate=(1,1))
        relu7_1 = mx.symbol.Activation(name='relu7_1', data=conv7_1, act_type='relu')
        conv7_2 = mx.symbol.Convolution(name='conv7_2', data=relu7_1, num_filter=512, pad=(1,1),kernel=(3,3), dilate=(1,1))
        relu7_2 = mx.symbol.Activation(name='relu7_2', data=conv7_2, act_type='relu')
        conv7_3 = mx.symbol.Convolution(name='conv7_3', data=relu7_2, num_filter=512, pad=(1,1),kernel=(3,3), dilate=(1,1))
        relu7_3 = mx.symbol.Activation(name='relu7_3', data=conv7_3, act_type='relu')
        conv7_3norm = mx.symbol.BatchNorm(name='conv7_3norm', data=relu7_3, use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        conv8_1 = mx.symbol.Deconvolution(name='conv8_1', data=conv7_3norm, num_filter=256, kernel=(4,4), pad=(1,1), stride=(2,2), dilate=(1,1))
        #bias = mx.nd.ones(shape=(1,256,56,56)) * 0.01
        #conv8_1 = conv8_1 + bias
        relu8_1 = mx.symbol.Activation(name='relu8_1', data=conv8_1, act_type='relu')
        conv8_2 = mx.symbol.Convolution(name='conv8_2', data=relu8_1, num_filter=256,pad=(1,1),kernel=(3,3), dilate=(1,1))
        relu8_2 = mx.symbol.Activation(name='relu8_2', data=conv8_2, act_type='relu')
        conv8_3 = mx.symbol.Convolution(name='conv8_3', data=relu8_2, num_filter=256,pad=(1,1),kernel=(3,3), dilate=(1,1))
        relu8_3 = mx.symbol.Activation(name='relu8_3', data=conv8_3, act_type='relu')
        return relu8_3
        #return relu8_3, conv8_1, conv7_3norm
        
        #conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=relu8_3, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        #return conv_feat_313

    def get_resnet(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1 , act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pad=(1,1), kernel=(3,3), stride=(2,2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma) 
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1 , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a , act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b , act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1,scale2a_branch2c] )
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a , act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a , act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b , act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu,scale2b_branch2c] )
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b , act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a , act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b , act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu,scale2c_branch2c] )
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c , act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a , act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b , act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1,scale3a_branch2c] )
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a , act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a , act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b , act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu,scale3b1_branch2c] )
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1 , act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a , act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b , act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu,scale3b2_branch2c] )
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2 , act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a , act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b , act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu,scale3b3_branch2c] )
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3 , act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a , act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b , act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1,scale4a_branch2c] )
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a , act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a , act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b , act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu,scale4b1_branch2c] )
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1 , act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a , act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b , act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu,scale4b2_branch2c] )
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2 , act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a , act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b , act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu,scale4b3_branch2c] )
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3 , act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a , act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b , act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu,scale4b4_branch2c] )
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4 , act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a , act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b , act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu,scale4b5_branch2c] )
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5 , act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a , act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b , act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu,scale4b6_branch2c] )
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6 , act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a , act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b , act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu,scale4b7_branch2c] )
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7 , act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a , act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b , act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu,scale4b8_branch2c] )
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8 , act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a , act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b , act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu,scale4b9_branch2c] )
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9 , act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a , act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b , act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu,scale4b10_branch2c] )
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10 , act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a , act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b , act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu,scale4b11_branch2c] )
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11 , act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a , act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b , act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu,scale4b12_branch2c] )
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12 , act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a , act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b , act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu,scale4b13_branch2c] )
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13 , act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a , act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b , act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu,scale4b14_branch2c] )
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14 , act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a , act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b , act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu,scale4b15_branch2c] )
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15 , act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a , act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b , act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu,scale4b16_branch2c] )
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16 , act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a , act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b , act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu,scale4b17_branch2c] )
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17 , act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a , act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b , act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu,scale4b18_branch2c] )
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18 , act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a , act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b , act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu,scale4b19_branch2c] )
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19 , act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a , act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b , act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu,scale4b20_branch2c] )
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20 , act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a , act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b , act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu,scale4b21_branch2c] )
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21 , act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a , act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b , act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu,scale4b22_branch2c] )
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22 , act_type='relu')
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a , act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b , act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1,scale5a_branch2c] )
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a , act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a , act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b , act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu,scale5b_branch2c] )
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b , act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a , act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu , num_filter=512, pad=(2,2), dilate=(2,2), kernel=(3,3), stride=(1,1), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b , act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=self.fix_gamma)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu,scale5c_branch2c] )
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c , act_type='relu')

        

        return res5c_relu

    def get_flownet(self, img_cur, img_ref):
        data = mx.symbol.Concat(img_cur / 255.0, img_ref / 255.0, dim=1)
        resize_data = mx.symbol.Pooling(name='resize_data', data=data , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        flow_conv1 = mx.symbol.Convolution(name='flow_conv1', data=resize_data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
        ReLU1 = mx.symbol.LeakyReLU(name='ReLU1', data=flow_conv1 , act_type='leaky', slope=0.1)
        conv2 = mx.symbol.Convolution(name='conv2', data=ReLU1 , num_filter=128, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=True)
        ReLU2 = mx.symbol.LeakyReLU(name='ReLU2', data=conv2 , act_type='leaky', slope=0.1)
        conv3 = mx.symbol.Convolution(name='conv3', data=ReLU2 , num_filter=256, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=True)
        ReLU3 = mx.symbol.LeakyReLU(name='ReLU3', data=conv3 , act_type='leaky', slope=0.1)
        conv3_1 = mx.symbol.Convolution(name='conv3_1', data=ReLU3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        ReLU4 = mx.symbol.LeakyReLU(name='ReLU4', data=conv3_1 , act_type='leaky', slope=0.1)
        conv4 = mx.symbol.Convolution(name='conv4', data=ReLU4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=True)
        ReLU5 = mx.symbol.LeakyReLU(name='ReLU5', data=conv4 , act_type='leaky', slope=0.1)
        conv4_1 = mx.symbol.Convolution(name='conv4_1', data=ReLU5 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        ReLU6 = mx.symbol.LeakyReLU(name='ReLU6', data=conv4_1 , act_type='leaky', slope=0.1)
        conv5 = mx.symbol.Convolution(name='conv5', data=ReLU6 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=True)
        ReLU7 = mx.symbol.LeakyReLU(name='ReLU7', data=conv5 , act_type='leaky', slope=0.1)
        conv5_1 = mx.symbol.Convolution(name='conv5_1', data=ReLU7 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        ReLU8 = mx.symbol.LeakyReLU(name='ReLU8', data=conv5_1 , act_type='leaky', slope=0.1)
        conv6 = mx.symbol.Convolution(name='conv6', data=ReLU8 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=True)
        ReLU9 = mx.symbol.LeakyReLU(name='ReLU9', data=conv6 , act_type='leaky', slope=0.1)
        conv6_1 = mx.symbol.Convolution(name='conv6_1', data=ReLU9 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        ReLU10 = mx.symbol.LeakyReLU(name='ReLU10', data=conv6_1 , act_type='leaky', slope=0.1)
        Convolution1 = mx.symbol.Convolution(name='Convolution1', data=ReLU10 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        deconv5 = mx.symbol.Deconvolution(name='deconv5', data=ReLU10 , num_filter=512, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_deconv5 = mx.symbol.Crop(name='crop_deconv5', *[deconv5,ReLU8] , offset=(1,1))
        ReLU11 = mx.symbol.LeakyReLU(name='ReLU11', data=crop_deconv5 , act_type='leaky', slope=0.1)
        upsample_flow6to5 = mx.symbol.Deconvolution(name='upsample_flow6to5', data=Convolution1 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='crop_upsampled_flow6_to_5', *[upsample_flow6to5,ReLU8] , offset=(1,1))
        Concat2 = mx.symbol.Concat(name='Concat2', *[ReLU8,ReLU11,crop_upsampled_flow6_to_5] )
        Convolution2 = mx.symbol.Convolution(name='Convolution2', data=Concat2 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        deconv4 = mx.symbol.Deconvolution(name='deconv4', data=Concat2 , num_filter=256, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_deconv4 = mx.symbol.Crop(name='crop_deconv4', *[deconv4,ReLU6] , offset=(1,1))
        ReLU12 = mx.symbol.LeakyReLU(name='ReLU12', data=crop_deconv4 , act_type='leaky', slope=0.1)
        upsample_flow5to4 = mx.symbol.Deconvolution(name='upsample_flow5to4', data=Convolution2 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='crop_upsampled_flow5_to_4', *[upsample_flow5to4,ReLU6] , offset=(1,1))
        Concat3 = mx.symbol.Concat(name='Concat3', *[ReLU6,ReLU12,crop_upsampled_flow5_to_4] )
        Convolution3 = mx.symbol.Convolution(name='Convolution3', data=Concat3 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        deconv3 = mx.symbol.Deconvolution(name='deconv3', data=Concat3 , num_filter=128, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_deconv3 = mx.symbol.Crop(name='crop_deconv3', *[deconv3,ReLU4] , offset=(1,1))
        ReLU13 = mx.symbol.LeakyReLU(name='ReLU13', data=crop_deconv3 , act_type='leaky', slope=0.1)
        upsample_flow4to3 = mx.symbol.Deconvolution(name='upsample_flow4to3', data=Convolution3 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_upsampled_flow4_to_3 = mx.symbol.Crop(name='crop_upsampled_flow4_to_3', *[upsample_flow4to3,ReLU4] , offset=(1,1))
        Concat4 = mx.symbol.Concat(name='Concat4', *[ReLU4,ReLU13,crop_upsampled_flow4_to_3] )
        Convolution4 = mx.symbol.Convolution(name='Convolution4', data=Concat4 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
        deconv2 = mx.symbol.Deconvolution(name='deconv2', data=Concat4 , num_filter=64, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_deconv2 = mx.symbol.Crop(name='crop_deconv2', *[deconv2,ReLU2] , offset=(1,1))
        ReLU14 = mx.symbol.LeakyReLU(name='ReLU14', data=crop_deconv2 , act_type='leaky', slope=0.1)
        upsample_flow3to2 = mx.symbol.Deconvolution(name='upsample_flow3to2', data=Convolution4 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=True)
        crop_upsampled_flow3_to_2 = mx.symbol.Crop(name='crop_upsampled_flow3_to_2', *[upsample_flow3to2,ReLU2] , offset=(1,1))
        Concat5 = mx.symbol.Concat(name='Concat5', *[ReLU2,ReLU14,crop_upsampled_flow3_to_2] )
        Concat5 = mx.symbol.Pooling(name='resize_concat5', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        Convolution5 = mx.symbol.Convolution(name='Convolution5', data=Concat5 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)

        Convolution5_scale_bias = mx.symbol.Variable(name='Convolution5_scale_bias', lr_mult=0.0)
        Convolution5_scale = mx.symbol.Convolution(name='Convolution5_scale', data=Concat5 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1),
                                                   bias=Convolution5_scale_bias, no_bias=True)
        return Convolution5 * 2.5, Convolution5_scale

    def get_train_symbol(self):
        data = mx.symbol.Variable(name="data")
        img_l, gt313, prior_boost_nongray = self.pre_train_data(data)

        conv_feat = self.colornet(img_l)
        #conv_feat = self.get_resnet(img_l)
        conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        conv_feat_313_boost = mx.symbol.Custom(name='conv_feat_313_boost', data1=conv_feat_313, data2=prior_boost_nongray, op_type='ClassRebalanceMult')
        Softmax = mx.symbol.SoftmaxActivation(data=conv_feat_313_boost, name='Softmax', mode='channel')

        loss8_313 = mx.sym.MakeLoss(data=mx.symbol.Custom(name='loss8_313', data=Softmax, label=gt313, op_type='CrossEntropy'), grad_scale=1.0)
        mx.sym.BlockGrad(loss8_313)
        return loss8_313 
    
    def get_test_symbol(self):
        data = mx.symbol.Variable(name='data')
        # only get the data and need not the ground truth
        #img_l, _, _ = self.pre_train_data(data)
        #conv_feat, conv1_2norm, conv1_2 = self.colornet(data)
        conv_feat = self.colornet(data)
        conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=1, no_bias=True)
        scale = conv_feat_313 * 2.606
        softmax = mx.symbol.SoftmaxActivation(data=scale, name='Softmax', mode='channel')
        class_ab = mx.symbol.Convolution(name='class_ab', data=softmax, num_filter=2, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        #group = mx.symbol.Group([mx.symbol.BlockGrad(conv_feat), mx.symbol.BlockGrad(class_ab), mx.symbol.BlockGrad(softmax), mx.symbol.BlockGrad(conv1_2norm), mx.symbol.BlockGrad(conv1_2)])
        group = mx.symbol.Group([mx.symbol.BlockGrad(conv_feat) , mx.symbol.BlockGrad(class_ab)])
        self.symbol = group
        return group

        #return class_ab

    
    def init_weight(self, arg_params, aux_params):
        arg_params['data_ab_ss_weight'] = mx.nd.ones(shape=(2,1,1,1))
        arg_params['data_ab_ss_bias'] = mx.nd.zeros(shape=(2,))
        pts_in_hull = np.load('/media/Disk/ziyang/code/colorization/resources/pts_in_hull.npy')
        pts_in_hull = pts_in_hull.transpose((1,0))
        res = np.ones((2,313,1,1))
        res[:,:,0,0] = pts_in_hull
        arg_params['class_ab_weight'] = mx.nd.array(res)
