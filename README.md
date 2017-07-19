# Wasserstein GAN Tensorflow
Implementation of [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) in Tensorflow. Official repo for
the paper can be found [here](https://github.com/martinarjovsky/WassersteinGAN).

___

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)

Datasets
* [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip)
* Image-net (coming soon)

___

### Results
Here are some non cherry-picked generated images after ~120,000 iterations. The graphs of the losses for the Generator
and Critic can be seen below. Both were generally converging.

![img](http://i.imgur.com/E3MgznB.jpg)

Critic loss

![d](http://i.imgur.com/YEcMm0P.png)

Generator loss

![g](http://i.imgur.com/Sp9hz47.png)

### Training
Training is pretty slow due to the small learning rate and multiple updates of the critic for one
update of the generator. Preloading the data helps speed it up a bit. These were trained on a GTX-1080
for about 24 hours.

I noticed that clipping the weights of the critic to [-0.1, 0.1] like they do in the paper caused the
critic and generator loss to not really change, although image quality was increasing. I found that instead
clipping the weights to [-0.05, 0.05] worked a bit better, showing better image quality and convergence.

### Data
Standard practice is to resize the CelebA images to 96x96 and the crop a center 64x64 image. `loadceleba.py`
takes as input the directory to your images, and will resize them upon loading. To load the entire dataset
at the start instead of reading from disk each step, you will need about 200000\*64\*64\*3\*3 bytes = ~7.5
GB of RAM.

### Tensorboard
Tensorboard logs are stored in `checkpoints/celeba/logs`. I am updating Tensorboard every step as training
isn't completely stable yet. *These can get very big*, around 50GB. See around line 115 in `train.py` to
change how often logs are committed.

### How to

#### Train
**You must have a dataset ready to train.**

`python train.py --DATASET=celeba --DATA_DIR=/path/to/celeba/ --BATCH_SIZE=32`

You can do this on anything though, not just celeba. If your images are png
just go into data_ops and change the ext='jpg' variable.

#### View Results

To see a fancy picture such as the one on this page, simply run

`python createPhotos.py checkpoints/celeba/`

or wherever your model is saved.

If you see the following as your "results", then you did not provide the complete path
to your checkpoint, and this is from the model's initialized weights.

![bad](http://i.imgur.com/MJfmze1.jpg)

