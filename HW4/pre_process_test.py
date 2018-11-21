# to reduce noise in the test data of 10,000
# We have done similar thing to the 9000 train data on google
# taking it offline b/c limited memory
# also we only pickle it once!!

# input: test data npy binary files
# output: pickle, centered, pre-processed test data
# output goes to current path you are at

import pandas as pd
import numpy as np
import pickle
import os
import csv
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def filter_noise(image, plot=False):
  image_train1 = np.uint8(image)
  image_train1 = image_train1.reshape(100, 100)

  if plot:
    print('Original image')
    print(image_train1.shape)
    plt.figure()
    plt.imshow(image_train1)

  # Get threshold and find contours
  ret, thresh = cv2.threshold(image_train1, 127, 255, 0)#0, 255, cv2.THRESH_BINARY)
  (img2,cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # We want to keep the largest contour
  largest = max(cnts, key = len)
  image_copy = image_train1.copy()
  cv2.drawContours(image_copy, [largest], 0, (0,255,0), 3)


  if plot:
    print('Filtered image')
    plt.figure()
    plt.imshow(image_copy)

  # Diff between filtered and unfiltered images
  diff = cv2.subtract(image_train1 , image_copy)

  if plot:
    print('Object after filtering')
    plt.figure()
    plt.imshow(diff)

  return diff




def center_image(im, plot = False):

  im = np.uint8(im)

  im = im.reshape(100, 100)

  if plot:
    print("in center")
    print(im.shape)
    plt.figure()
    plt.imshow(im)

  # Get the largest contour from the now-filtered image (this should
  # just be the object in the image now)
  im_copy = im.copy()
  ret, thresh = cv2.threshold(im_copy, 0, 127, 0)
  thresh_copy = thresh.copy()
  (img2,cnts, _) = cv2.findContours(thresh_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  largest = max(cnts, key = len)

  # Crop the image according to the largest contour
  x, y, w, h = cv2.boundingRect(largest)
  cropped = im_copy[y:y+h, x:x+w]
  if plot:
    plt.figure()
    plt.imshow(cropped)

  # Pad on all sizes, to 100x100
  desired_size = 100
  old_size = cropped.shape[:2] # old_size is in (height, width) format
  new_size = old_size
  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)

  if plot:
    print('image after:')
    plt.figure()
    plt.imshow(new_im)

  return new_im


def preprocess_image(image, plot = False):
  image = filter_noise(image, plot)
  image = center_image(image, plot)
  return image


currPath = os.getcwd() #not to be confused with DATA_PATH on google drive
'''
images_test = np.load('./all/test_images.npy', encoding='latin1')
test_df = pd.DataFrame(images_test, columns = ['Id', 'img'])
testing_vectors = np.array(test_df['img'].tolist())
testing_vectors = testing_vectors.reshape(
            testing_vectors.shape[0], 100, 100)
print("testing_vectors", testing_vectors.shape)

test_data_preproc = []
for datapoint in testing_vectors:
  test_data_preproc.append(preprocess_image(datapoint, plot = False))
'''
# stack into 3 channels
with open(os.path.join(currPath, "test_data_preproc.pickle"), 'rb') as jar:
    test_data_preproc = pickle.load(jar)
test_data_preproc = np.array(test_data_preproc)
vgg_test_data = np.stack((test_data_preproc,test_data_preproc,test_data_preproc), axis = -1)


# Not stacked version
#with open(os.path.join(currPath, "test_data_preproc.pickle"), 'wb') as jar:
#    pickle.dump(test_data_preproc, jar, protocol=pickle.HIGHEST_PROTOCOL)

# stacked 3 channels for VGG
with open(os.path.join(currPath, "vgg_test_data_preproc.pickle"), 'wb') as jar:
    pickle.dump(vgg_test_data, jar, protocol=pickle.HIGHEST_PROTOCOL)
