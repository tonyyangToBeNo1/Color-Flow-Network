# Color-Flow-Network
this is a baseline for CFNs

Installation
=================
1. clone this repository
2. install Mxnet from the following code:
```
git clone --recursive https://github.com/dmlc/mxnet.git
git checkout 62ecb60
git submodule update
```
modify the config.mk to set your path and run the code:
```
make all -j8
cd python
sudo python setup.py install
```
to make it convenient you can add the your path/Mxnet/pyhton into the system path

3. install ffmpeg


demo codes
===================
there is two demos
the demo.py and the demo_batch.py
the dataset is a single videos from VID dataset. 
both the demo can colorize from the videos in the demo_data files.
the difference bewteen the demos is that demo_batch deals with one key frames and nine non-key frames
while demo deals with one key frames and one non-key frames

Run the code
===================
we have the warp_train.py and frame_train.py file to train the file
the warp_train.py is optimized by the optic flows while frame_train just finetuning from the colorization models.
the train the models,
the data should be like the following format:
```
ILSVRC2015/Data
ILSVRC2015/ImageSets
ILSVRC2015/ImageSets/Videos/train.txt
ILSVRC2015/ImageSets/Videos/trainlist/train1.txt
ILSVRC2015/ImageSets/Videos/trainlist/train2.txt
```
my trainLists is like the following:
```
your_path_to_video_root/ your_path_to_video_root_images_trainlists
```

details of the code:
====================
the network structure models are in the ./symbol
some new_added python operators are saved in ./cfn/opertor_py
some scripts are saved in ./cfn/core

To add a mxnet operators, you should create as the following forms:
```
class newOperator(mx.operator.CustomOp):
  def __init__(self):
    ...
  def forward(self, is_train, req, in_data, out_data, aux): # all the params could not be ignore
    ...
  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    ...
@mx.operator.register('newOP...')
class newProp(mx.operator.CustomOpProp):
  def __init__(self):
    ...
  def list_arguments(self):
    return ['first data input', 'second data input' 'and so on']
  def list_outputs(self):
    return ['output']
  def infer_shape(self, input_shape):
    return [input_shape], [out_shape1, output_shape2,...], []
  def create_operator(self, ctx, shapes, dtypes):
    return newOperator() # this is the entry to forward and backward
                         # you can also pass some param from prop to Op
```

the dataIter will read the images file iteratively
you can set the batchsize in the train.py and you need to set the data path too.
in the code I set it to read a whole videos and the the next one
Let's say we have a video with 144 frames and the batch_size is 50
Then we will read the last 44 frames and read the last frame for six times as pading.
The Iter will automatically ignore the padding

if you still have some problems in implement this code, feel free to contract me
[tang385@purdue.edu]
