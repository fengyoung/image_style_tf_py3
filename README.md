# image_style_tf_py3

This code is based on [fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow), and it is implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Requirements

- python 3.x
- tensorflow >= 1.0

## Installation

**1. Install TensorFlow**

See [Installing TensorFlow](https://www.tensorflow.org/install/) for instructions on how to install the release binaries or how to build from source.

**2. Clone the source of image_style_tf_py3**

```
git clone https://github.com/fengyoung/image_style_tf_py3.git <YOUR REPO PATH>
```

## How to Use Trained Model to Stylize an Image

1. Download [trained models](http://pan.baidu.com/s/1kURjpLd) first. The files with suffix ".ckpt-done" are models and  ".jpg" files are corresponding style images

2. Stylize your image by using **eval.py**
```
cd <THIS REPO>
python3 eval.py --image_file xxx.jpg --model_file <MODEL PATH>/xxxx.ckpt-done --output_file yyy.jpg
```
Then check out yyy.jpg.

## How to Train a New Model

1. To train new models, you should first download VGG16 model from [Tensorflow Slim](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) or from my [Baidu Drive](http://pan.baidu.com/s/1eRDMtsY). Extract the file vgg_16.ckpt

2. Collect a image-set contains large number of ".jpg" files or download [COCO dataset](http://pan.baidu.com/s/1c2thNGG)

3. Prepare a style image and train the style model by using **train.py**
```
cd <THIS REPO>
python3 train.py --style_image <STYLE IMAGE> --naming <NAMING> --model_path <MODEL PATH> --loss_model_file <YOUR PATH TO vgg_16.ckpt> --data_set <YOUR IMAGE_SET>
```
Then copy **\<MODEL PATH\>/\<NAME OF THE MODEL\>/fast-style-model.ckpt-done** as new style model

### Required Arguments
```
--style_image STYLE_IMAGE
  Target style image which is used for training model
--naming NAMING
  The name of this model. Determine the path to save checkpoint and events files.
--model_path MODEL_PATH
  Root path to save checkpoint and events files. The final path would be <MODEL_PATH>/<NAMING>
--loss_model_file LOSS_MODEL_FILE
  The path to the model checkpoint file
--data_set DATA_SET
  Path of image data set such as COCO data set
```

### Optional Arguments
```
-h, --help
  Show help message and exit
--content_weight CONTENT_WEIGHT
  Weight for content features loss. Default is 1.0.
--style_weight STYLE_WEIGHT
  Weight for style features loss. Default is 100.0.
--tv_weight TV_WEIGHT
  Weight for total variation loss. Default is 0.0.
--image_size IMAGE_SIZE
  Size of style image normalization. Default is 256.
--batch_size BATCH_SIZE
  Batch size of one iteration. Default is 4.
--epoch EPOCH
  Epoch times of whole data set. Default is 2.
--loss_model LOSS_MODEL
  Name of loss model. "vgg_16" as default.
--content_layers CONTENT_LAYERS
  Use these layers for content loss. They are splitted by ",".
  As default, layer "vgg_16/conv3/conv3_3" is used.
--style_layers STYLE_LAYERS
  Use these layers for style loss. They are splitted by ",".
  As default, layer "vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2", "vgg_16/conv3/conv3_3" ,"vgg_16/conv4/conv4_3" are used. 
--checkpoint_exclude_scopes CHECKPOINT_EXCLUDE_SCOPES
  We only use the convolution layers, so ignore fc layers
```
