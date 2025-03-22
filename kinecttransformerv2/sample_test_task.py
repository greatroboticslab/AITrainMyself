import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from kinect_learning import *
import time
np.set_printoptions(suppress=True)
DATA_DIR = 'data'

def create_datasets(x, y, valid_size=0.2, test_size=0.2):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # Shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = np.take(x, indices, axis=0)
    y = np.take(y, indices)

    # Split into train, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=valid_size + test_size)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=test_size / (valid_size + test_size))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

data_file_name = 'left-right.csv'
noise  = False
file_path = os.path.join(DATA_DIR, data_file_name)
data_collection = joints_collection(data_file_name.rstrip('.csv'))
data = load_data_multiple_dimension(file_path, data_collection, noise)
(train_x, train_y), (val_x, val_y), (test_x, test_y) = create_datasets(data['positions'], data['labels'])

print(len(data['positions']))

print(data_collection)
print(train_x[0])
print(type(train_y))
print(len(train_y))

with open(file_path) as json_file:
    datafile = json.loads(json_file.read())

features = datafile[1]['jointPositions']['jointPositionDict']
print(features)
Xi=[]
for joint in data_collection:
   #print(joint)
   xj = list(features[joint].values())
   Xi.append(xj)

print(Xi)
print(data['positions'][1])