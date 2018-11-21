import numpy as np
import pickle
import os
import pandas as pd

'''
opening comment:
b/c we did not pickle the labels, we are doing it here offline
and uploading to google drive

then for each new model we try,
just load a pickle instead of re-reading the big NPY file

FORMAT: these pickle are of shape (N,), Category_nums, not onehot or strings
See the one hot encoded labels under oneHotLabel.py
'''
# TODO: PICKLE the one hot encoded labels!!


PICKLE_DATA_PATH = './pickle_data/'
RAW_DATA_PATH = './raw_data/'

def extract_labels():
    '''
    SAME as the proc_simplenet notebook
    just changed the path name for local computer
    returns the corresponding labels of the original dataset
    '''

    images_train = np.load(os.path.join(RAW_DATA_PATH, 'train_images.npy'),
                    encoding='latin1')
    train_labels = pd.read_csv(os.path.join(RAW_DATA_PATH,'train_labels.csv'))

    label_list = train_labels.Category.unique().tolist()
    label_dict = {label:i  for i,label in enumerate(label_list)}
    train_labels['Category_num'] = train_labels.Category.apply(lambda x: label_dict[x])



    # naive 9000 to 1000 split, no random

    training_labels = train_labels['Category_num'][:9000].values
    validation_labels = train_labels['Category_num'][9000:].values


    return training_labels, validation_labels


def original_label_pickle(training_labels, validation_labels):
    '''
    Not one hot encoded
    '''
    with open(os.path.join(PICKLE_DATA_PATH, "original_9000_training_labels.pickle"), 'wb') as jar:
        pickle.dump(training_labels, jar, protocol=pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(PICKLE_DATA_PATH, "original_1000_valid_labels.pickle"), 'wb') as jar:
        pickle.dump(validation_labels, jar, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":
    # the two lines get the num category of each label

    training_labels, validation_labels = extract_labels()

    # only pickle once, commented out
    # not one-hot encoded
    #original_label_pickle(training_labels, validation_labels)
