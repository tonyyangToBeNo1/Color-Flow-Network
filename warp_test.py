import sys
sys.path.append('symbol')
from imageIter import *
import CFNet
from cfn.core import load_model
import mxnet as mx
import logging
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
import caffe
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
    GPU=0
    for j in range(len(val_path)):
        sample = val_path[j][-6:]
        print 'testing', sample

        f1 = open('ILSVRC2015/ImageSets/Videos/videos/ILSVRC2015_val_00'+sample+'.txt')
        List = f1.readlines()
        f1.close()
        img_root = 'ILSVRC2015/Data/VID/val/ILSVRC2015_val_00'+sample+'/'
        for idx in range(len(List)):
            List[idx] = List[idx].strip()
    
        sym_instance = CFNet.CFNet()
        sym = sym_instance.get_flow_test_symbol()
        #summary(sym, (2,1,224,224))
        prefix = './output/colornet/CFNwarp100'
        #prefix = 'model/pretrained_model/colornet'
        #epoch = 2
        epoch = 0
        #param_file = '%s-0060.params' % prefix
        #symbol_file = '%s' % prefix
        print 'loading the params'
        arg_params, aux_params = load_model.load_param(prefix, epoch, process = True)
        sym_instance.init_weight(arg_params, aux_params)
        print 'init the module'
        model = mx.module.Module(context=mx.gpu(GPU), symbol=sym, data_names=['data', 'data_cur'],label_names=None)
        model.bind(data_shapes=[('data',(1,1,224,224)), ('data_cur',(1,1,224,224))], label_shapes=None, for_training=False)
        model.init_params(arg_params=arg_params, aux_params=aux_params)
        print 'test the data'
        H_in = 224
        W_in = 224
        H_out = 56
        W_out = 56
        #for i in range(1):
        for i in range(0, len(List), 2):
            if i + 1 >= len(List):
                # the last key frame will be abandon
                break
            img_rgb = cv2.imread(img_root + List[i])
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            (H_orig, W_orig) = img_rgb.shape[:2]
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray_img = np.concatenate((img_gray[:,:,np.newaxis], img_gray[:,:,np.newaxis], img_gray[:,:,np.newaxis]), axis=2)
    
            img_lab = color.rgb2lab(gray_img)
            img_l = img_lab[:,:,0]
            img_rs = cv2.resize(gray_img, (H_in, W_in))
            img_lab_rs = color.rgb2lab(img_rs)
            img_l_rs = img_lab_rs[:,:,0]
            data=np.zeros((1,1,224,224))
            data[0,0,:,:] = img_l_rs - 50
    
    
    
            img_rgb_cur = cv2.imread(img_root + List[i + 1])
            img_rgb_cur = cv2.cvtColor(img_rgb_cur, cv2.COLOR_BGR2RGB)
            img_gray_cur = cv2.cvtColor(img_rgb_cur, cv2.COLOR_RGB2GRAY)
            gray_img_cur = np.concatenate((img_gray_cur[:,:,np.newaxis], img_gray_cur[:,:,np.newaxis],img_gray_cur[:,:,np.newaxis]), axis=2)
            img_lab_cur = color.rgb2lab(gray_img_cur)
            img_l_cur = img_lab_cur[:,:,0]
            img_rs_cur = cv2.resize(gray_img_cur, (H_in,W_in))
            img_lab_rs_cur = color.rgb2lab(img_rs_cur)
            img_l_rs_cur = img_lab_rs_cur[:,:,0]
            data_cur = np.zeros((1,1,224,224))
            data_cur[0,0,:,:] = img_l_rs_cur - 50
            
            DB = mx.io.DataBatch(data=[mx.nd.array(data), mx.nd.array(data_cur)], provide_data=[('data', data.shape),('data_cur', data_cur.shape)])
            model.forward(DB)
            result = model.get_outputs()[0].asnumpy()
            ab_dec = result[0].transpose((1,2,0))
            cur_result = model.get_outputs()[1].asnumpy()
            ab_dec_cur = cur_result[0].transpose((1,2,0))
            ab_dec_us = sni.zoom(ab_dec, (1.*H_orig/H_out, 1.*W_orig/W_out, 1))
            img_lab_out = np.concatenate((img_l[:,:,np.newaxis], ab_dec_us), axis=2)
            
            img_rgb_out = (255 * np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8')
    
            ab_dec_us_cur = sni.zoom(ab_dec_cur,(1.*H_orig/H_out, 1.*W_orig/W_out,1))
            img_lab_out_cur = np.concatenate((img_l_cur[:,:,np.newaxis],ab_dec_us_cur), axis=2)
            img_rgb_out_cur=(255*np.clip(color.lab2rgb(img_lab_out_cur),0,1)).astype('uint8')
    
            if not os.path.exists('images/val'+sample):
                os.mkdir('images/val'+sample+'/')
            save_path = 'images/val'+sample+'/warp_'
            print 'saving image: ', sample, List[i].strip()
            #cv2.imwrite('139005/'+List[i].strip(), img_rgb_out)
            plt.imsave(save_path+List[i].strip(), img_rgb_out)
            print 'saving image: ', sample, List[i+1].strip()
            plt.imsave(save_path+List[i+1].strip(), img_rgb_out_cur)
    
        if not os.path.exists('test_res'):
            os.mkdir('test_res')
        os.system('rm test_res/*.JPEG')
        os.system('convert '+save_path+'*.JPEG test_res/warp_%05d.JPEG')
        os.system('ffmpeg -r 30 -i test_res/warp_%05d.JPEG video/val'+sample+'.mp4')

