
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
import cv2


'''
experiment of data augmentation using imgaug library
permute each image NUM_PERM times, and append its data, and labels


NOTE: permute refers to the randomness in the image augmentation, e.g.
degrees of rotation, etc, not the order of image array itself.

program takes a while to run

'''


# define the constants
DATA_PATH = './pickle_data/'
NUM_PERM = 10

def data_loader():

    '''
    load N by height by width by 1 images from Pickle
    images are centered and filtered already
    load the 9000, 1000 split of train.npz
    '''

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



# Naive implementation of transformations

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

def loop_aug(images, numPerm= NUM_PERM, filename = "original_9000_training_labels.pickle"):
    '''
    augment each image many times
    different i for different rand seed, different
    random rotations

    everytime, append/atack all the labels wrt its orignal order

    so that the index of aug_img_arr matches the one of its orignal form

    '''
    aug_img_arr = []
    aug_labels_arr = []

    with open(os.path.join(DATA_PATH, filename ), 'rb') as jar:
        original_labels = pickle.load(jar)

    for i in range(numPerm):

        ia.seed(i)
        #append permutation of images, returned as N x 100 x 100
        aug_img_arr.append(aug_img(images))

        #stack labels accordingly
        aug_labels_arr.append(original_labels)

    aug_img_arr = np.array(aug_img_arr)

    # reshape m x N x 100 x 100 to mN x 100 x 100
    # Where m is the numPerm, number of permutations for each image
    aug_img_arr = aug_img_arr.reshape(
                    aug_img_arr.shape[0]* aug_img_arr.shape[1],
                    aug_img_arr.shape[2], aug_img_arr.shape[3], 1)
    aug_labels_arr = np.array(aug_labels_arr)
    aug_labels_arr = aug_labels_arr.reshape(
                        aug_labels_arr.shape[0] * aug_labels_arr.shape[1], )
    return aug_img_arr, aug_labels_arr

def invert_img(images, plot = False):
    '''
    images: N by height by width by 1
    each inverted image: height by width
    '''
    # try to invert black and white
    ret_images = []
    for i in range(images.shape[0]):
        imagem = cv2.bitwise_not(images[i])
        #print("imagem", imagem.shape)
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
        aug_valid):
    #choose random ones to plot to verify the results
    idx_train_original = np.random.random_integers(low = 0, high = train_data.shape[0])
    idx_train_aug =  int(idx_train_original)

    plt.figure()
    plt.imshow(train_data[idx_train_original].reshape(100, 100))
    plt.show()

    plt.figure()
    plt.imshow(aug_train[idx_train_aug].reshape(100, 100))
    plt.show()

    #plt.figure()
    #plt.imshow(inv_aug_train[idx_train].reshape(100, 100))
    #plt.show()

    idx_val_original = np.random.random_integers(low = 0, high =valid_data.shape[0] )
    idx_val_aug = int(idx_val_original)
    plt.figure()
    plt.imshow(valid_data[idx_val_original].reshape(100, 100))
    plt.show()

    plt.figure()
    plt.imshow(aug_valid[idx_val_aug].reshape(100, 100))
    plt.show()



def save_aug_pickle(aug_train_data, aug_valid_data,
        aug_train_labels, aug_valid_labels):
    '''
    save final results after augmenting the pixels
    and its corresponding labels

    There is no permutation of the indices, just appending in the
    original order
    '''

    print("aug_train_data ", aug_train_data.shape)
    print("aug_train_labels", aug_train_labels.shape)
    print("aug_valid_data ", aug_valid_data.shape)

    print("aug_valid_labels", aug_valid_labels.shape)
    with open(os.path.join(DATA_PATH, "aug_train_data.pickle"), 'wb') as jar:
        pickle.dump(aug_train_data, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "aug_valid_data.pickle"), 'wb') as jar:
        pickle.dump(aug_valid_data, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "aug_train_labels.pickle"), 'wb') as jar:
        pickle.dump(aug_train_labels, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "aug_valid_labels.pickle"), 'wb') as jar:
        pickle.dump(aug_valid_labels, jar, protocol=pickle.HIGHEST_PROTOCOL)

def save_inv_aug_pickle():
    '''
    save final result after augmenting and inverting the B/W
    of pixels
    '''

    with open(os.path.join(DATA_PATH, "inv_aug_train.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_train, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(DATA_PATH, "inv_aug_valid.pickle"), 'wb') as jar:
        pickle.dump(inv_aug_valid, jar, protocol=pickle.HIGHEST_PROTOCOL)

def test_load_labels():
    '''
    verify that the arrays are stacked to NUM_PERM copies of original Sequential
    e.g.
    array([[ 0,  4,  8, 12,  1,  5,  9, 13],
       [ 2,  6, 10, 14,  3,  7, 11, 15]])

    to array([ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15]

    NOT [ 0,  8,  1,  9,  2, 10,  3, 11, 4, 12,  5, 13,  6, 14,  7, 15]

    '''

    with open(os.path.join(DATA_PATH, "original_1000_valid_labels.pickle"), 'rb') as jar:
        original_1000_valid_labels = pickle.load(jar)

    with open(os.path.join(DATA_PATH, "aug_valid_labels.pickle"),'rb') as jar:
        aug_valid_labels = pickle.load(jar)


    len_original = len(original_1000_valid_labels)
    print("len_original", len_original)
    print("aug_valid_labels", aug_valid_labels.shape)
    cnt = 0

    for i in range(len_original):

            if original_1000_valid_labels[i] != aug_valid_labels[i]:
                cnt +=1
    print(cnt)

if __name__== "__main__":

    train_data, valid_data = data_loader()
    print("start load data", train_data.shape)

    #train_data = train_data
    #valid_data = valid_data

    # loop through different random seeds for the image augmentation
    # software library
    aug_train_data, aug_train_labels = loop_aug(train_data,
                filename = "original_9000_training_labels.pickle")

    aug_valid_data, aug_valid_labels = loop_aug(valid_data,
                filename = "original_1000_valid_labels.pickle")

    test_final_plots(train_data, valid_data,aug_train_data, aug_valid_data )

    print("aug_train_data ", aug_train_data.shape)
    print("aug_train_labels", aug_train_labels.shape)
    print("aug_valid_data ", aug_valid_data.shape)

    print("aug_valid_labels", aug_valid_labels.shape)
    save_aug_pickle(aug_train_data, aug_valid_data, aug_train_labels, aug_valid_labels)



    # test if numpy has stacked the label and the pictures the same way

    '''
    with open(os.path.join(DATA_PATH, "aug_train_data.pickle"), 'rb') as jar:
        aug_train_data = pickle.load(jar)

    with open(os.path.join(DATA_PATH, "aug_valid_data.pickle"), 'rb') as jar:
        aug_valid_data = pickle.load(jar)

    test_final_plots(train_data, valid_data,aug_train_data, aug_valid_data )

    test_load_labels()
    '''
