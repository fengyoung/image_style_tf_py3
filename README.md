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
git clone https://github.com/fengyoung/image_style_tf_py3.git <your source path>
```
## How to Use Trained Model to Stylize an Image

1. Download [trained models](http://pan.baidu.com/s/1kURjpLd) first. The files with suffix ".ckpt-done" are models and  ".jpg" files are corresponding style images

2. Stylize your image by using eval.py
```
cd <this repo>
python3 eval.py --image_file xxx.jpg --model_file <your model path>/xxxx.ckpt-done --output_file yyy.jpg
```
Then check out yyy.jpg.

## How to Train a New Model

[coco dataset](http://pan.baidu.com/s/1c2thNGG)
