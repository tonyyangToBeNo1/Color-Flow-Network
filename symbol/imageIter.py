import mxnet as mx
import os, sys
import cv2
import numpy as np
import random 

# define a data_iter to read the images
class ImageIter(mx.io.DataIter):
    def __init__(self, batch_size = 1, source_root=None, Data_root=None, data_name="data"):
        super(ImageIter, self).__init__()
        self.batch_size = 1
        self.image_idx = 0
        # each batch we will read 10 images
        self.k = batch_size
        self.source_root = source_root
        self.Data_root = Data_root
        self.crop = 176
        self.data_name = data_name
        self.label_name = 'softmax_label'
        # in the iter, we first store all the image path in the list
        # the format will be the following
        # [idx][paths]
        # name is the video name
        # paths will be the image_path in the videos
        self.data = []
        source = open(source_root, 'r')
        videos = source.readlines()
        source.close()
        self.size = []
        for video in videos:
            video_path = video.strip().split(' ')
            # video_path[0] is ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/
            # video_path[1] is /media/Disk/ziyang/code/Deep-Feature-Flow/ILSVRC2015/ImageSets/Videos/videos/ILSVRC2015_train_00001000.txt
            img_path = open(video_path[1], 'r')
            images = img_path.readlines()
            img_path.close()
            paths = []
            self.size.append(len(images))
            for image in images:
                image = image.strip()
                paths.append(self.Data_root + video_path[0] + image)
            self.data.append(paths)
        self.label = []
        self.cur = 0
        self.idx = 0
        self.reset()


    @property
    def provide_data(self):
        return [(self.data_name, (self.k, 3, self.crop, self.crop))]

    @property
    def provide_label(self):
    # this label is not useful
        return None
        #return [('softmax_label', (self.k, 1))]

    @property
    def _getpad(self):
        return self.image_idx + self.k > self.size[self.idx]

    def _getdata(self):
        images = []
        if (self.idx > len(self.size)):
            self.idx = 0
            self.image_idx = 0

        for i in range(self.image_idx, self.image_idx + self.k):
            if i >= self.size[self.idx]:
                print 'last epoch of this idx'
                for j in range(0, self.image_idx + self.k - self.size[self.idx]):
                    img = cv2.imread(self.data[self.idx][self.size[self.idx] - 1])
                    img = img.transpose((2,1,0))
                    images.append(img[:,0:176,0:176])
                break
            if not os.path.exists(self.data[self.idx][i]):
                print 'the image Path is not exists!'
                raise
            img = cv2.imread(self.data[self.idx][i]).transpose((2,1,0))
            # we need to randomly crop the image and make it into 176 * 176
            channel, height, width = img.shape
            random_height = random.randint(0, height - self.crop)
            random_width = random.randint(0, width - self.crop)
            images.append(img[:, random_height:random_height+self.crop, random_width:random_width+self.crop])
        self.image_idx += self.k
        return mx.nd.array(images)

    def _getlabel(self):
        labels = []
        for i in range(self.k):
            labels.append(1)
        return mx.nd.array(labels)

    def reset(self):
        self.cur = 0
        self.idx = 0

    def next(self):
        if self.iter_next():
            images = self._getdata()
            #print images.asnumpy()[0][0,0,:]
            #labels = self._getlabel()
            images = [images]
            #labels = [labels]
            return mx.io.DataBatch(data=images, provide_data=self.provide_data,  pad=self._getpad)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        if (self.image_idx > self.size[self.idx]):
            self.cur += 1
            self.image_idx = 0

        return self.cur < len(self.size)
    
    def infer_shape(self):
        return [(self.batch_size * self.k, 3, self.crop, self.crop)], []
