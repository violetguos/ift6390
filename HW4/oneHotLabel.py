import numpy as np
import os
import pickle

'''
Opening comment:
After we obtain the labels in pickles, we load the pickle,
and convert them into One hot
Then pickle the one hot into a separate pickle

You can add more labels to convert one hot and to pickle in main,
The convert target function works for any input dim
'''


PICKLE_DATA_PATH = './pickle_data/'


def convertTarget(targetValues):
    numClasses = np.max(targetValues) + 1
    return np.eye(numClasses)[targetValues]


if __name__== "__main__":
    # original_label_pickle

    with open(os.path.join(PICKLE_DATA_PATH, "original_9000_training_labels.pickle"), 'rb') as jar:
        training_labels = pickle.load(jar)


    with open(os.path.join(PICKLE_DATA_PATH, "original_1000_valid_labels.pickle"), 'rb') as jar:
        validation_labels = pickle.load(jar)


    # convert to one hot
    training_labels_one_hot = convertTarget(training_labels)
    validation_labels_one_hot = convertTarget(validation_labels)

    # verify dimensions
    print("training_labels", training_labels.shape)
    print("training_labels_one_hot", training_labels_one_hot.shape)
    print("validation_labels", validation_labels.shape)
    print("validation_labels_one_hot", validation_labels_one_hot.shape)

    # store in a jar
    with open(os.path.join(PICKLE_DATA_PATH, "training_labels_one_hot.pickle"), 'wb') as jar:
        pickle.dump(training_labels_one_hot, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(PICKLE_DATA_PATH, "validation_labels_one_hot.pickle"), 'wb') as jar:
        pickle.dump(validation_labels_one_hot, jar, protocol=pickle.HIGHEST_PROTOCOL)


    ########## finished one hot convert for the original data set######


    # augmented label_pickle

    with open(os.path.join(PICKLE_DATA_PATH, "aug_train_labels.pickle"), 'rb') as jar:
        aug_train_labels = pickle.load(jar)


    with open(os.path.join(PICKLE_DATA_PATH, "aug_valid_labels.pickle"), 'rb') as jar:
        aug_valid_labels = pickle.load(jar)

    # convert to one hot
    aug_train_labels_one_hot = convertTarget(aug_train_labels)
    aug_valid_labels_one_hot = convertTarget(aug_valid_labels)

    # verify dimensions
    print("aug_train_labels", aug_train_labels.shape)
    print("aug_train_labels_one_hot", aug_train_labels_one_hot.shape)
    print("aug_valid_labels", aug_valid_labels.shape)
    print("aug_valid_labels_one_hot", aug_valid_labels_one_hot.shape)

    # store in a jar
    with open(os.path.join(PICKLE_DATA_PATH, "aug_train_labels_one_hot.pickle"), 'wb') as jar:
        pickle.dump(aug_train_labels_one_hot, jar, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(PICKLE_DATA_PATH, "aug_valid_labels_one_hot.pickle"), 'wb') as jar:
        pickle.dump(aug_valid_labels_one_hot, jar, protocol=pickle.HIGHEST_PROTOCOL)
