# monkey around data augmentation

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
import cv2
ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
#images = np.array(
#    [ia.quokka(size=(64, 64)) for _ in range(32)],
#    dtype=np.uint8
#)

# load N by height by width by 1 images from Pickle
# images are centered and filtered already
DATA_PATH = './pickle_data/'

def data_loader():

    with open(os.path.join(DATA_PATH, "valid_data_preproc.pickle"), 'rb') as jar:
        valid_data = pickle.load(jar)

    with open(os.path.join(DATA_PATH, "train_data_preproc.pickle"),'rb') as jar:
        train_data = pickle.load(jar)

    # try on 1 image of our pickle
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)

    train_data = np.expand_dims(train_data, axis = -1)
    valid_data = np.expand_dims(valid_data, axis = -1)
    return train_data, valid_data





seq = iaa.Sequential([
    iaa.Fliplr(1), # horizontal flips

    # we centered them so forget about cropping??!?!

    #iaa.Crop(percent=(0, 0.1)), # random crops

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.

    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),

    # Strengthen or weaken the contrast in each image.

    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.

    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.

    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.

    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-360, 360),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


def aug_img(images, plot = False):
    '''
    input images:  N by height by width by 1
    output: images_aug: N by height by width by 1
    '''
    #print("input images test img", images.shape)
    # testing on example given in the Lib Doc on 1 cat
    images_aug = seq.augment_images(images)
    #print("images_aug",images_aug.shape)

    if plot:
        # returns a array of images,
        # we just choose to plot the first one
        plt.figure()
        plt.imshow(images_aug[0].reshape(100, 100))
        plt.show()

    return images_aug

def invert_img(images, plot = False):
    '''
    images: N by height by width by 1
    each inverted image: height by width
    '''
    # try to invert black and white
    ret_images = []
    for i in range(images.shape[0]):
        imagem = cv2.bitwise_not(images[i])
        print("imagem", imagem.shape)
        imagem = imagem.reshape(100, 100, 1)
        ret_images.append(imagem)
        if plot:
            # plot all of them after inverting
            # be careful not to crash the system!
            plt.figure()
            plt.imshow(imagem.reshape(100, 100))
            plt.show()
    ret_images = np.array(ret_images)
    #print("ret_images", ret_images.shape)


    return ret_images

def test_final_plots(train_data, valid_data,aug_train,
        inv_aug_train, aug_valid, inv_aug_valid):
    #choose random ones to plot to verify the results
    idx_train = np.random.random_integers(low = 0, high = 9000)
    plt.figure()
    plt.imshow(train_data[idx_train].reshape(100, 100))
    plt.show()

    plt.figure()
    plt.imshow(aug_train[idx_train].reshape(100, 100))
    plt.show()

    plt.figure()
    plt.imshow(inv_aug_train[idx_train].reshape(100, 100))
    plt.show()

    idx_val = np.random.random_integers(low = 0, high = 1000)
    plt.figure()
    plt.imshow(valid_data[idx_val].reshape(100, 100))

    plt.figure()
    plt.imshow(aug_valid[idx_val].reshape(100, 100))

    plt.figure()
    plt.imshow(inv_aug_valid[idx_val].reshape(100, 100))
    plt.show()

def save_aug_pickle(aug_train, aug_valid):
    '''
    save final results after augmenting, and inverting the pixels

    '''
    with open(os.path.join(DATA_PATH, "aug_train.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_train, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "aug_valid.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_valid, jar, protocol=pickle.HIGHEST_PROTOCOL)

def save_inv_aug_pickle():
    '''
    save final result after augmenting and inverting the B/W
    of pixels
    '''

    with open(os.path.join(DATA_PATH, "inv_aug_train.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_train, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "inv_aug_valid.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_valid, jar, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":
    train_data, valid_data = data_loader()
    print("start load data", train_data.shape)

    train_data = train_data[0:10]
    valid_data = valid_data[0:10]
    aug_train = aug_img(train_data)
    inv_aug_train = invert_img(aug_train, plot = False)

    aug_valid = aug_img(valid_data)
    inv_aug_valid = invert_img(aug_valid, plot = False)
    test_final_plots(train_data, valid_data,aug_train,
            inv_aug_train, aug_valid, inv_aug_valid)
