from cfn.operator_py.rgb2lab import *
from cfn.operator_py.priorboost import *
from cfn.operator_py.crossentropy import *
from cfn.operator_py.NNencoder import *
from cfn.operator_py.nongray import *
from cfn.operator_py.classrebalance import *
import math

class CFNet():
    def __init__(self):
        self.eps = 1e-4
        self.use_global_stats = False
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

    def colornet(self, img_key):
        conv1_1 = mx.symbol.Convolution(name='bw_conv1_1', data=img_key, num_filter=64, kernel=(3,3), pad=(1,1), stride=(1,1))
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

    def flownet(self, img_cur, img_key):
        # img_cur and img_key will be plus 50 and then rescale to (0,1)
        data_cur = img_cur + 50
        data_key = img_key + 50

        combine_cur = mx.symbol.Concat(data_cur, data_cur, data_cur, dim = 1)
        combine_key = mx.symbol.Concat(data_key, data_key, data_key, dim = 1)

        flow_data = mx.symbol.Concat(combine_cur / 255.0, combine_key / 255.0, dim = 1)
        #resize_data = mx.symbol.Pooling(name='resize_data', data=data, pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        flow_conv1 = mx.symbol.Convolution(name='flow_conv1', data=flow_data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
        ReLU1 = mx.symbol.LeakyReLU(name='ReLU1', data=flow_conv1 , act_type='leaky', slope=0.1)
        flow_conv2 = mx.symbol.Convolution(name='flow_conv2', data=ReLU1 , num_filter=128, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU2 = mx.symbol.LeakyReLU(name='ReLU2', data=flow_conv2 , act_type='leaky', slope=0.1)
        flow_conv3 = mx.symbol.Convolution(name='flow_conv3', data=ReLU2 , num_filter=256, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU3 = mx.symbol.LeakyReLU(name='ReLU3', data=flow_conv3 , act_type='leaky', slope=0.1)
        flow_conv3_1 = mx.symbol.Convolution(name='flow_conv3_1', data=ReLU3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU4 = mx.symbol.LeakyReLU(name='ReLU4', data=flow_conv3_1 , act_type='leaky', slope=0.1)
        flow_conv4 = mx.symbol.Convolution(name='flow_conv4', data=ReLU4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU5 = mx.symbol.LeakyReLU(name='ReLU5', data=flow_conv4 , act_type='leaky', slope=0.1)
        flow_conv4_1 = mx.symbol.Convolution(name='flow_conv4_1', data=ReLU5 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU6 = mx.symbol.LeakyReLU(name='ReLU6', data=flow_conv4_1 , act_type='leaky', slope=0.1)
        flow_conv5 = mx.symbol.Convolution(name='flow_conv5', data=ReLU6 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU7 = mx.symbol.LeakyReLU(name='ReLU7', data=flow_conv5 , act_type='leaky', slope=0.1)
        flow_conv5_1 = mx.symbol.Convolution(name='flow_conv5_1', data=ReLU7 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU8 = mx.symbol.LeakyReLU(name='ReLU8', data=flow_conv5_1 , act_type='leaky', slope=0.1)
        flow_conv6 = mx.symbol.Convolution(name='flow_conv6', data=ReLU8 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU9 = mx.symbol.LeakyReLU(name='ReLU9', data=flow_conv6 , act_type='leaky', slope=0.1)
        flow_conv6_1 = mx.symbol.Convolution(name='flow_conv6_1', data=ReLU9 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU10 = mx.symbol.LeakyReLU(name='ReLU10', data=flow_conv6_1 , act_type='leaky', slope=0.1)
        flow_Convolution1 = mx.symbol.Convolution(name='flow_Convolution1', data=ReLU10 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv5 = mx.symbol.Deconvolution(name='deconv5', data=ReLU10 , num_filter=512, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv5 = mx.symbol.Crop(name='crop_deconv5', *[deconv5,ReLU8] , offset=(1,1))
        ReLU11 = mx.symbol.LeakyReLU(name='ReLU11', data=crop_deconv5 , act_type='leaky', slope=0.1)
        upsample_flow6to5 = mx.symbol.Deconvolution(name='upsample_flow6to5', data=flow_Convolution1 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='crop_upsampled_flow6_to_5', *[upsample_flow6to5,ReLU8] , offset=(1,1))
        Concat2 = mx.symbol.Concat(name='Concat2', *[ReLU8,ReLU11,crop_upsampled_flow6_to_5] )
        flow_Convolution2 = mx.symbol.Convolution(name='flow_Convolution2', data=Concat2 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv4 = mx.symbol.Deconvolution(name='deconv4', data=Concat2 , num_filter=256, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv4 = mx.symbol.Crop(name='crop_deconv4', *[deconv4,ReLU6] , offset=(1,1))
        ReLU12 = mx.symbol.LeakyReLU(name='ReLU12', data=crop_deconv4 , act_type='leaky', slope=0.1)
        upsample_flow5to4 = mx.symbol.Deconvolution(name='upsample_flow5to4', data=flow_Convolution2 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='crop_upsampled_flow5_to_4', *[upsample_flow5to4,ReLU6] , offset=(1,1))
        Concat3 = mx.symbol.Concat(name='Concat3', *[ReLU6,ReLU12,crop_upsampled_flow5_to_4] )
        flow_Convolution3 = mx.symbol.Convolution(name='flow_Convolution3', data=Concat3 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv3 = mx.symbol.Deconvolution(name='deconv3', data=Concat3 , num_filter=128, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv3 = mx.symbol.Crop(name='crop_deconv3', *[deconv3,ReLU4] , offset=(1,1))
        ReLU13 = mx.symbol.LeakyReLU(name='ReLU13', data=crop_deconv3 , act_type='leaky', slope=0.1)
        upsample_flow4to3 = mx.symbol.Deconvolution(name='upsample_flow4to3', data=flow_Convolution3 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow4_to_3 = mx.symbol.Crop(name='crop_upsampled_flow4_to_3', *[upsample_flow4to3,ReLU4] , offset=(1,1))
        Concat4 = mx.symbol.Concat(name='Concat4', *[ReLU4,ReLU13,crop_upsampled_flow4_to_3] )
        flow_Convolution4 = mx.symbol.Convolution(name='flow_Convolution4', data=Concat4 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv2 = mx.symbol.Deconvolution(name='deconv2', data=Concat4 , num_filter=64, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv2 = mx.symbol.Crop(name='crop_deconv2', *[deconv2,ReLU2] , offset=(1,1))
        ReLU14 = mx.symbol.LeakyReLU(name='ReLU14', data=crop_deconv2 , act_type='leaky', slope=0.1)
        upsample_flow3to2 = mx.symbol.Deconvolution(name='upsample_flow3to2', data=flow_Convolution4 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow3_to_2 = mx.symbol.Crop(name='crop_upsampled_flow3_to_2', *[upsample_flow3to2,ReLU2] , offset=(1,1))
        Concat5 = mx.symbol.Concat(name='Concat5', *[ReLU2,ReLU14,crop_upsampled_flow3_to_2] )
        #Concat5 = mx.symbol.Pooling(name='resize_concat5', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        flow_Convolution5 = mx.symbol.Convolution(name='flow_Convolution5', data=Concat5 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)

        Convolution5_scale_bias = mx.symbol.Variable(name='Convolution5_scale_bias', lr_mult=0.0)
        Convolution5_scale = mx.symbol.Convolution(name='Convolution5_scale', data=Concat5 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1),
                                                   bias=Convolution5_scale_bias)
        return flow_Convolution5 * 2.5, Convolution5_scale

    def get_train_symbol(self):
        data = mx.symbol.Variable(name="data")
        #data_cur = mx.symbol.Variable(name='data_cur')
        #data_cur = mx.symbol.Custom(name='data_cur', data=data, op_type='getOther')
        img_l, gt313, prior_boost_nongray = self.pre_train_data(data)
        #img_cur, gt313_cur, prior_boost_nongray_cur = self.pre_train_data(data_cur)

        # colornet is for the key frame to get its feature
        #conv_feat, flow_feat, flow_scale = self.CFN(img_cur, img_l)
        img_key, img_others = mx.symbol.Custom(name='split_data', data=img_l, op_type='splitdata')
        gt_key, gt_others = mx.symbol.Custom(name='split_gt', data=gt313, op_type='splitdata')
        pbn_key, pbn_others = mx.symbol.Custom(name='split_pgb', data=prior_boost_nongray, op_type='splitdata')

        conv_feat = self.colornet(img_key)
        flow_feat, flow_scale = self.flownet(img_others, img_key)
        
        flow_grid = mx.symbol.GridGenerator(data=flow_feat, transform_type='warp', name='flow_grid')
        warp_conv_feat = mx.symbol.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')
        warp_conv_feat = warp_conv_feat * flow_scale
        conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias =True)
        conv_feat_313_boost = mx.symbol.Custom(name='conv_feat_313_boost', data1=conv_feat_313, data2=pbn_key, op_type='ClassRebalanceMult')
        Softmax = mx.symbol.SoftmaxActivation(data=conv_feat_313_boost, name='Softmax', mode='channel')
        loss8_313 = mx.sym.MakeLoss(data=mx.symbol.Custom(name='loss8_313', data=Softmax, label=gt_key, op_type='CrossEntropy'), grad_scale=1.0)
        mx.symbol.BlockGrad(loss8_313)

        flow_conv_feat_313 = mx.symbol.Convolution(name='flow_conv_feat_313', data=warp_conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias =True)
        flow_conv_feat_313_boost = mx.symbol.Custom(name='flow_conv_feat_313_boost', data1=flow_conv_feat_313, data2=pbn_others, op_type='ClassRebalanceMult')
        Softmax_flow = mx.symbol.SoftmaxActivation(data=flow_conv_feat_313_boost, name='flow_Softmax', mode='channel')
        flow_loss8_313 = mx.sym.MakeLoss(data=mx.symbol.Custom(name='flow_loss8_313', data=Softmax_flow,label=gt_others, op_type='CrossEntropy'), grad_scale=1.0)
        #mx.symbol.BlockGrad(loss8_313)
        #mx.symbol.BlockGrad(flow_loss8_313)
        group = mx.symbol.Group([loss8_313, flow_loss8_313])
        self.symbol = group
        return group

    def get_test_symbol(self):
        data = mx.symbol.Variable(name='data')
        # only get the data and need not the ground truth
        #img_l, _, _ = self.pre_train_data(data)
        #conv_feat, conv1_2norm, conv1_2 = self.colornet(data)
        conv_feat = self.colornet(data)
        conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=1, no_bias=False)
        scale = conv_feat_313 * 2.606
        softmax = mx.symbol.SoftmaxActivation(data=scale, name='Softmax', mode='channel')
        class_ab = mx.symbol.Convolution(name='class_ab', data=softmax, num_filter=2, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=False)
        #group = mx.symbol.Group([mx.symbol.BlockGrad(conv_feat), mx.symbol.BlockGrad(class_ab), mx.symbol.BlockGrad(softmax), mx.symbol.BlockGrad(conv1_2norm), mx.symbol.BlockGrad(conv1_2)])
        group = mx.symbol.Group([mx.symbol.BlockGrad(conv_feat_313) , mx.symbol.BlockGrad(class_ab)])
        self.symbol = group
        return group

        #return class_ab

    def get_flow_test_symbol(self):
        #data=mx.symbol.Variable(name='data')
        
        #data_key = mx.symbol.slice_axis(data=data, axis=0, begin=0,end=1)
        #data_cur = mx.symbol.slice_axis(data=data, axis=0, begin=1,end=2)
        data_key = mx.symbol.Variable(name='data')
        data_cur = mx.symbol.Variable(name='data_cur')
        # test the key frames
        conv_feat = self.colornet(data_key)
        conv_feat_313 = mx.symbol.Convolution(name='conv8_313', data=conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        scale = conv_feat_313 * 2.606
        softmax = mx.symbol.SoftmaxActivation(data=scale, name='Softmax', mode='channel')
        class_ab = mx.symbol.Convolution(name='class_ab', data=softmax, num_filter=2, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        mx.symbol.BlockGrad(class_ab) 
        # test the cur frames
        #img_l_key = mx.symbol.Concat(data_key, data_key, data_key, dim = 1)
        #img_l_cur = mx.symbol.Concat(data_cur, data_cur, data_cur, dim= 1)
        # get the optic flow and the scale for temporal information
        flow_feat, flow_scale = self.flownet(data_cur, data_key)
        # warp the optic flow information with the key convolution feat
        flow_grid = mx.symbol.GridGenerator(data=flow_feat, transform_type='warp', name='flow_grid')
        warp_conv_feat = mx.symbol.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')
        warp_conv_feat = warp_conv_feat * flow_scale

        #concat_feat = mx.sym.Concat(conv_feat, warp_conv_feat, dim=0)
        flow_feat_313 = mx.symbol.Convolution(name='flow_conv8_313', data=warp_conv_feat, num_filter=313, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        flow_scale = flow_feat_313 * 2.606
        flow_softmax = mx.symbol.SoftmaxActivation(data=flow_scale, name='Softmax_flow', mode='channel')
        #concat_softmax = mx.symbol.Concat(softmax, flow_softmax, dim=0)

        flow_class_ab = mx.symbol.Convolution(name='flow_class_ab', data=flow_softmax, num_filter=2, kernel=(1,1), stride=(1,1), dilate=(1,1), no_bias=True)
        mx.symbol.BlockGrad(flow_class_ab)
        
        #return class_ab
        group = mx.symbol.Group([class_ab , flow_class_ab, mx.symbol.BlockGrad(conv_feat), mx.symbol.BlockGrad(warp_conv_feat)])
        self.symbol = group
        return group
    
    
    def init_weight(self, arg_params, aux_params):
        pts_in_hull = np.load('/media/Disk/ziyang/code/colorization/resources/pts_in_hull.npy')
        pts_in_hull = pts_in_hull.transpose((1,0))
        res = np.ones((2,313,1,1))
        res[:,:,0,0] = pts_in_hull
        arg_params['class_ab_weight'] = mx.nd.array(res)
        arg_params['flow_class_ab_weight'] = mx.nd.array(res)
        arg_params['flow_conv8_313_weight'] = arg_params['conv8_313_weight']
        return 

        arg_params['data_ab_ss_weight'] = mx.nd.ones(shape=(2,1,1,1))
        arg_params['data_ab_ss_bias'] = mx.nd.zeros(shape=(2,))
        
        arg_params['data_ab_ss_cur_weight'] = mx.nd.ones(shape=(2,1,1,1))
        arg_params['data_ab_ss_cur_bias'] = mx.nd.zeros(shape=(2,)) 
        arg_params['flow_conv_feat_313_weight'] = arg_params['conv8_313_weight']
        #arg_params['flow_conv_feat_313_bias'] = arg_params['conv8_313_bias']
        #arg_params['flow_conv8_313_weight'] = arg_params['conv8_313_weight']
        
        arg_params['Convolution5_scale_weight'] = mx.nd.zeros(shape=(256,194,1,1))
        arg_params['Convolution5_scale_bias'] = mx.nd.zeros(shape=(256,))
        # init the flownet weights
        arg_params['flow_conv1_weight'] = mx.nd.array(np.load('flownet_weights/conv1_0.npy'))
        arg_params['flow_conv1_bias'] = mx.nd.array(np.load('flownet_weights/conv1_1.npy'))
        arg_params['flow_conv2_weight'] = mx.nd.array(np.load('flownet_weights/conv2_0.npy'))
        arg_params['flow_conv2_bias'] = mx.nd.array(np.load('flownet_weights/conv2_1.npy'))
        arg_params['flow_conv3_weight'] = mx.nd.array(np.load('flownet_weights/conv3_0.npy'))
        arg_params['flow_conv3_1_bias'] = mx.nd.array(np.load('flownet_weights/conv3_1.npy'))
        arg_params['flow_conv3_bias'] = mx.nd.array(np.load('flownet_weights/conv3_1_1.npy'))
        arg_params['flow_conv3_1_weight'] = mx.nd.array(np.load('flownet_weights/conv3_1_0.npy'))
        arg_params['flow_conv4_weight'] = mx.nd.array(np.load('flownet_weights/conv4_0.npy'))
        arg_params['flow_conv4_bias'] = mx.nd.array(np.load('flownet_weights/conv4_1.npy'))
        arg_params['flow_conv4_1_bias'] = mx.nd.array(np.load('flownet_weights/conv4_1_1.npy'))
        arg_params['flow_conv4_1_weight'] = mx.nd.array(np.load('flownet_weights/conv4_1_0.npy'))
        arg_params['flow_conv5_weight'] = mx.nd.array(np.load('flownet_weights/conv5_0.npy'))
        arg_params['flow_conv5_bias'] = mx.nd.array(np.load('flownet_weights/conv5_1.npy'))
        arg_params['flow_conv5_1_bias'] = mx.nd.array(np.load('flownet_weights/conv5_1_1.npy'))
        arg_params['flow_conv5_1_weight'] = mx.nd.array(np.load('flownet_weights/conv5_1_0.npy'))
        arg_params['flow_conv6_weight'] = mx.nd.array(np.load('flownet_weights/conv6_0.npy'))
        arg_params['flow_conv6_bias'] = mx.nd.array(np.load('flownet_weights/conv6_1.npy'))
        arg_params['flow_conv6_1_bias'] = mx.nd.array(np.load('flownet_weights/conv6_1_1.npy'))
        arg_params['flow_conv6_1_weight'] = mx.nd.array(np.load('flownet_weights/conv6_1_0.npy'))
        arg_params['flow_Convolution1_weight'] = mx.nd.array(np.load('flownet_weights/Convolution1_0.npy'))
        arg_params['flow_Convolution1_bias'] = mx.nd.array(np.load('flownet_weights/Convolution1_1.npy'))
        arg_params['flow_Convolution2_weight'] = mx.nd.array(np.load('flownet_weights/Convolution2_0.npy'))
        arg_params['flow_Convolution2_bias'] = mx.nd.array(np.load('flownet_weights/Convolution2_1.npy'))
        arg_params['flow_Convolution3_weight'] = mx.nd.array(np.load('flownet_weights/Convolution3_0.npy'))
        arg_params['flow_Convolution3_bias'] = mx.nd.array(np.load('flownet_weights/Convolution3_1.npy'))
        arg_params['flow_Convolution4_weight'] = mx.nd.array(np.load('flownet_weights/Convolution4_0.npy'))
        arg_params['flow_Convolution4_bias'] = mx.nd.array(np.load('flownet_weights/Convolution4_1.npy'))
        arg_params['flow_Convolution5_weight'] = mx.nd.array(np.load('flownet_weights/Convolution5_0.npy'))
        arg_params['flow_Convolution5_bias'] = mx.nd.array(np.load('flownet_weights/Convolution5_1.npy'))
        arg_params['deconv2_weight'] = mx.nd.array(np.load('flownet_weights/deconv2_0.npy'))
        arg_params['deconv2_bias'] = mx.nd.array(np.load('flownet_weights/deconv2_1.npy'))
        
        arg_params['deconv3_weight'] = mx.nd.array(np.load('flownet_weights/deconv3_0.npy'))
        
        arg_params['deconv3_bias'] = mx.nd.array(np.load('flownet_weights/deconv3_1.npy'))
        arg_params['deconv4_weight'] = mx.nd.array(np.load('flownet_weights/deconv4_0.npy'))
        arg_params['deconv4_bias'] = mx.nd.array(np.load('flownet_weights/deconv4_1.npy'))
        arg_params['deconv5_weight'] = mx.nd.array(np.load('flownet_weights/deconv5_0.npy'))
        arg_params['deconv5_bias'] = mx.nd.array(np.load('flownet_weights/deconv5_1.npy'))
        arg_params['upsample_flow3to2_weight'] = mx.nd.array(np.load('flownet_weights/upsample_flow3to2_0.npy'))
        arg_params['upsample_flow3to2_bias'] = mx.nd.array(np.load('flownet_weights/upsample_flow3to2_1.npy'))
        arg_params['upsample_flow4to3_weight'] = mx.nd.array(np.load('flownet_weights/upsample_flow4to3_0.npy'))
        arg_params['upsample_flow4to3_bias'] = mx.nd.array(np.load('flownet_weights/upsample_flow4to3_1.npy'))
        arg_params['upsample_flow5to4_weight'] = mx.nd.array(np.load('flownet_weights/upsample_flow5to4_0.npy'))
        arg_params['upsample_flow5to4_bias'] = mx.nd.array(np.load('flownet_weights/upsample_flow5to4_1.npy'))
        arg_params['upsample_flow6to5_weight'] = mx.nd.array(np.load('flownet_weights/upsample_flow6to5_0.npy'))
        arg_params['upsample_flow6to5_bias'] = mx.nd.array(np.load('flownet_weights/upsample_flow6to5_1.npy')) 
