# image_style_tf_py3

This code is based on [fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow)

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
python3 train.py --style_image <STYLE IMAGE FILE> --naming <NAME OF THE MODEL> --model_path <MODEL PATH> --loss_model_file <YOUR PATH TO vgg_16.ckpt> --data_set <PATH TO YOUR IMAGE-SET>
```
Then copy the \<MODEL PATH\>/\<NAME OF THE MODEL\>/fast-style-model.ckpt-done 



