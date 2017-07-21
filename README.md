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
image like this, simply run,

`python createPhotos.py checkpoints/path/to/checkpoint_file`

![img](http://i.imgur.com/SgXTiDs.jpg)

