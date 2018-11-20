# Get the pickle of test results from Google
# build a CSV file locally
# don't wanna use too much big brother's memory
# TODO: make it into a function with argparser and shell script!


import pandas as pd
import numpy as np
import pickle
import os
import csv

currPath = os.getcwd()
with open(os.path.join(currPath, 'test_predicted_cross_val.pickle'), 'rb') as jar:
  y_test = pickle.load(jar)


# get the one - to one mapping for labels to one hot
table_labels = pd.read_csv('./all/train_labels.csv')
label_list = table_labels.Category.unique().tolist()


# from one hot encoding back to the category name for test pickle
y_classes = [label_list[np.argmax(y, axis=None, out=None)] for y in y_test]

# build csv
with open('test_predicted.csv', 'w') as csvfile:
    # defined by the sample csv
    fieldnames = ['Id', 'Category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(len(y_classes)):
        writer.writerow({'Id':i, 'Category':y_classes[i]})
