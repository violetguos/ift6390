# Purpose of the scripts
taking some dump processing offline due to google's memory limit

the scripts are not perfect in any shape or form, apologies in advance

they are more like *patches* for what we neglected in the original data processing+ simple net notebook (`kaggle_hw4.ipynb`), or the parts where google complained about memory usage

## data_aug.py
experiment of data augmentation using imgaug library
permute each image NUM_PERM times, and append its data, and labels


NOTE: permute refers to the randomness in the image augmentation, e.g.
degrees of rotation, etc, not the order of image array itself.

program takes a while to run

## labelPickle.py

b/c we did not pickle the labels, we are doing it here offline
and uploading to google drive

then for each new model we try,
just load a pickle instead of re-reading the big NPY file

FORMAT: these pickle are of shape (N,), Category_nums, not onehot or strings
See the one hot encoded labels under oneHotLabel.py


## pre_process_test.py
to reduce noise in the test data of 10,000

We have done similar thing to the 9000 train data on google

taking it offline b/c limited memory
also we only pickle it once!!

 input: test data npy binary files

 output: pickle, centered, pre-processed test data

 output goes to current path you are at

## extract_results.py
NOTE: this is reusable each time we submit

Build csv submission file
from the pickle of test results from Google
build a CSV file locally
don't wanna use too much big brother's memory
Pickle contains one-hot predictions

one hot -> index -> string of corresponding index in the category names

file will go to current path you are at, may not be the best practice



## oneHotLabel.py

After we obtain the labels in pickles, we load the pickle,
and convert them into One hot

Then pickle the one hot into a separate pickle

You can add more labels to convert one hot and to pickle in main,

The convert target function works for any input dim
