import os, sys
sys.path.append('symbol/')
import color_flow
import imageIter

from cfn.core import load_model
import mxnet as mx
import numpy as np
import logging
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni

def summary(symbol, shape=None):
    internals = symbol.get_internals()
    _, out_shapes, _ = internals.infer_shape(data=shape)
    name = internals.list_outputs()
    layer = open('test_layer_shape.txt', 'w')
    for i in range(len(name)):
        layer.write(name[i] + ' ' + str(out_shapes[i]) + '\n')

if __name__ == '__main__':
    val_path = os.listdir('ILSVRC2015/Data/VID/val/')
    val_path.sort()
    GPU = 0
    for j in range(len(val_path)):
        sample = val_path[j][-6:]
        print 'testing', sample
        f1 = open('ILSVRC2015/ImageSets/Videos/videos/ILSVRC2015_val_00'+sample+'.txt')
        List = f1.readlines()
        f1.close()
        #img_root = 'ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00139005/'
        img_root = 'ILSVRC2015/Data/VID/val/ILSVRC2015_val_00'+sample+'/'
        for i in range(len(List)):
            List[i] = List[i].strip()
    
        sym_instance = color_flow.color_flow()
        sym = sym_instance.get_test_symbol()
        summary(sym, (1,1,224,224))
        #prefix = './output/colornet/CFN100'
        prefix = 'model/pretrained_model/colornet'
        #epoch = 2
        epoch = 0
        #param_file = '%s-0060.params' % prefix
        #symbol_file = '%s' % prefix
        print 'loading the params'
        arg_params, aux_params = load_model.load_param(prefix, epoch)
        sym_instance.init_weight(arg_params, aux_params)
        #sym, arg_params, aux_params = mx.model.load_checkpoint(symbol_file, param_file)
        print 'init the module'
        model = mx.module.Module(context=mx.gpu(GPU), symbol=sym, label_names=None)
        model.bind(data_shapes=[('data',(1,1,224,224))], label_shapes=None,for_training=False)
        model.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False, force_init=True)
        #model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False, force_init=True)
        print 'test the data'
        H_in = 224
        W_in = 224
        H_out = 56
        W_out = 56
        #for i in range(2):
        for i in range(len(List)):
            print img_root + List[i].strip()
            img_rgb = cv2.imread(img_root + List[i])
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            #img_rgb = caffe.io.load_image(img_root+List[i])
            #print img_rgb.shape
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            height, width, channel = img_rgb.shape
            gray_img = np.zeros((height, width, channel))
            gray_img[:,:,0] = img_gray
            gray_img[:,:,1] = img_gray
            gray_img[:,:,2] = img_gray
            
            (H_orig, W_orig) = img_rgb.shape[:2]
            img_lab = color.rgb2lab(img_rgb)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray_img = np.concatenate((img_gray[:,:,np.newaxis], img_gray[:,:,np.newaxis], img_gray[:,:,np.newaxis]), axis=2)
    
            #gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
            img_lab = color.rgb2lab(gray_img)
            img_l = img_lab[:,:,0]
            #print img_l
            #x = input()
            cv2.imwrite('val007010/gray_'+List[i].strip(), img_l)
            img_rs = cv2.resize(gray_img, (H_in, W_in))
            #img_rs = caffe.io.resize_image(gray_img, (H_in, W_in))
            
            
            img_lab_rs = color.rgb2lab(img_rs)
            img_l_rs = img_lab_rs[:,:,0]
            #print img_l_rs
            #x = input()
            data = np.zeros((1,1,224,224))
            data[0,0,:,:] = img_l_rs - 50
            #data = np.zeros((1,3,224,224))
            #data[0,:,:,:] = img_rs.transpose((2,1,0))
            #print data[0,0,1,:]
            DB = mx.io.DataBatch(data=[mx.nd.array(data)], provide_data=[('data', data.shape)])
            model.forward(DB)
            #test_iter = mx.io.NDArrayIter(data, batch_size=1)
            #model.predict(test_iter)
            #print model.get_outputs()
    
            result = model.get_outputs()[1].asnumpy()
            ab_dec = result[0].transpose((1,2,0))
            
            print ab_dec.shape
            #print ab_dec
            #print ab_dec[0,:,:]
            
            ab_dec_us = sni.zoom(ab_dec, (1.*H_orig/H_out, 1.*W_orig/W_out, 1))
            img_lab_out = np.concatenate((img_l[:,:,np.newaxis], ab_dec_us), axis=2)
            
            img_rgb_out = (255 * np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8')
            if not os.path.exists('images/frame'+sample):
                os.mkdir('images/frame'+sample+'/')
            save_path = 'images/frame'+sample+'/frame_'
            print 'saving image: ', List[i].strip()
            #cv2.imwrite('139005/'+List[i].strip(), img_rgb_out)
            print img_rgb_out.shape
            plt.imsave('val007010/frame_'+List[i].strip(), img_rgb_out)
        
        os.system('rm frame_res/*.JPEG')
        os.system('convert '+save_path+'*.JPEG frame_res/frame_%05d.JPEG')
        os.system('ffmpeg -r 30 -i frame_res/frame_%05d.JPEG video/frame'+sample+'.mp4')
