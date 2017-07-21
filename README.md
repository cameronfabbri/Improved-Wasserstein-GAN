# Improved Wasserstein GAN Tensorflow

Tensorflow implementation of [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf), Wasserstein GAN with Gradient Penalty.

Datasets
* [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip)

___

### How to Run
`python train.py --DATASET=celeba --DATA_DIR=/path/to/celeba/`

Other options include [SELU activations](https://arxiv.org/abs/1706.02515) and
[layer normalization](https://arxiv.org/abs/1607.06450), which the authors of
WGAN-GP suggest, as the discriminator does not use batch normalization because that
would conflict with the gradient penalty. Default is to not use these, but they can be used by,

`python train.py --DATASET=celeba --DATA_DIR=/path/to/celeba/ --NORM=1`


### Results
Here are some non cherry picked results after ~100,000 training steps with batch size 128. To create an
image like this, simply run `createPhotos.py` and point towards your checkpoint directory, like so,

`python createPhotos.py checkpoints/DATASET_celeba/SCALE_10/NORM_False/SELU_False/`

![img](http://i.imgur.com/SgXTiDs.jpg)


### Notes
- Initial trials of SELU activations did not work, the model diverged pretty quickly.
- For some reason, I was getting terrible results using `tf.layers.conv2d` as opposed
to `tf.contrib.layers.conv2d`, and I am still unsure as to why.
- The last layer of the discriminator is another convolution with stride 1, kernel size of 4,
and depth of 1. I found this to work much better than the typical fully connected layer.
- Using layer normalization seems to have more stable training, although it takes longer
for each step (~2 seconds for batch size 128 on a GTX 1080 as opposed to ~1.5 seconds without
layer norm). However, it seems to be converging faster, so it's possible that offsets the time.
