from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt

import random
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.quantization
import math
from os.path import join
#from kinect_learning import joints_collection, load_data # * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)
import time  # Import time for measuring inference speed

def joints_collection(posture):
    switcher = {
        "left-right" : ['KneeRight', 'KneeLeft', 'AnkleRight', 'AnkleLeft', 'FootRight', 'FootLeft'],
        "turning" : ['HandLeft', 'HandRight', 'WristLeft', 'WristRight', 'ElbowLeft', 'ElbowRight', 'ShoulderLeft',
                    'ShoulderRight', 'ShoulderCenter', 'HipLeft', 'HipRight', 'HipCenter', 'KneeLeft', 'KneeRight'],
        "bending" : ['Head', 'ShoulderLeft', 'ShoulderRight', 'ShoulderCenter', 'ElbowLeft', 'ElbowRight', 'WristLeft',
                   'WristRight', 'HandLeft', 'HandRight', 'Spine', 'HipLeft', 'HipRight', 'HipCenter'],
        "up-down" : ['HandLeft', 'HandRight', 'WristLeft', 'WristRight', 'ElbowLeft', 'ElbowRight', 'ShoulderLeft',
                   'ShoulderRight', 'ShoulderCenter'],
        "sit-stand" : ['HipCenter', 'HipLeft', 'HipRight', 'KneeRight', 'KneeLeft', 'WristRight', 'WristLeft',
                     'HandRight', 'HandLeft', 'ElbowRight', 'ElbowLeft'],
        "all" : ['HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                      'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HipLeft', 'KneeLeft',
                      'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']
    }
    return switcher.get(posture)


def load_data(file_name, collection, noise):
	with open(file_name) as json_file:
		data = json.loads(json_file.read())
	X = []
	y = []
	count = 0
	for datum in data:
		yi = datum['label']
        ## Add 30 noise data or not
		if yi == 2:
			if noise == True:
				count += 1
				if count > 30:
					continue
				else:
					yi = random.randint(0,1)
			else:
				continue

		y.append(yi)
		Xi = []
		features = datum['jointPositions']['jointPositionDict']
		for joint in collection:
			xj = list(features[joint].values())
			Xi = Xi + xj
		X = X + [Xi]
	return {'positions' : X, 'labels' : y}


def load_data_multiple_dimension(file_name, collection, noise):
	with open(file_name) as json_file:
		data = json.loads(json_file.read())
	X = []
	y = []
	count = 0
	length = len(data)
	for datum in data:
		yi = datum['label']
        ## Add 30 noise data or not
		if yi == 2:
			if noise == True:
				count += 1
				if count > 2000:
					continue
				else:
					yi = random.randint(0,1)
			else:
				continue

		y.append(yi)
		Xi = []
		features = datum['jointPositions']['jointPositionDict']
		i = 0
		for joint in collection:
			xj = list(features[joint].values())
			#if noise == True:
			#	xj[0] += random.randint(0, 1)
			#	xj[1] += random.randint(0, 1)
			#	xj[2] += random.randint(0, 1)
			Xi.append(xj)
		X.append(Xi)
	return {'positions' : X, 'labels' : y}


## Build path to file.
DATA_DIR = 'data'
FILE_NAME = 'bending.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)

left_right_col = joints_collection('left-right')
sit_stand_col = joints_collection('sit-stand')
turning_col = joints_collection('turning')
bending_col = joints_collection('bending')
up_down_col = joints_collection('up-down')
all_col = joints_collection('all')

seed = 1
np.random.seed(seed)

COLLECTION = bending_col
print("Printing scores of small collection...")
print("Collection includes", COLLECTION)
print("Printing scores of small collection with noise data...")
noise = False
X, y = load_data_multiple_dimension(FILE_PATH, COLLECTION, noise)['positions'], load_data_multiple_dimension(FILE_PATH, COLLECTION, noise)['labels']



def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()


# Inference speed measurement function
def measure_inference_speed(model, data_loader):
    model.eval()  # Set model to evaluation mode
    start_time = time.time()  # Start time

    # Perform inference across entire dataset
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation for faster inference
        for x_val, y_val in data_loader:
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

    end_time = time.time()  # End time
    accuracy = correct / total
    inference_time = end_time - start_time  # Total inference time
    return accuracy, inference_time

class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2
    return scheduler

def create_datasets(X, y, test_size=0.2):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int16)

    divsion = X.shape[0] % 10
    actual_length = X.shape[0] - divsion
    X = X[0:actual_length, :]
    y = y[0:actual_length]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)

    return train_ds, valid_ds

def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

if __name__ == "__main__":
    print('Preparing datasets')
    trn_ds, val_ds = create_datasets(X, y)
    bs = 200
    print('Creating data loaders with batch size: {}'.format(bs))
    trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=0)

    input_dim = 3
    hidden_dim = 256
    layer_dim = 5
    output_dim = 2
    lr = 0.0005
    n_epochs = 10
    iterations_per_epoch = len(trn_dl)
    best_acc = 0
    patience, trials = 100000, 0

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))


    '''
    # show the changes that were made
    print('Here is the floating point version of this module:')
    print(model)
    print('')
    print('and now the quantized version:')
    print(quantized_lstm)'''

    print('Start model training')
    for epoch in range(1, n_epochs + 1):
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            opt.step()
            sched.step()
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()

        # Evaluate model without quantization
        print("Evaluating non-quantized model...")
        acc_non_quantized, time_non_quantized = measure_inference_speed(model, val_dl)
        print(f"Non-Quantized Model - Epoch {epoch}: Accuracy: {acc_non_quantized:.5f}, Inference Time: {time_non_quantized:.5f} seconds")

        # Quantize model and evaluate again
        print("Evaluating quantized model...")
        quantized_lstm = torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        acc_quantized, time_quantized = measure_inference_speed(quantized_lstm, val_dl)
        print(f"Quantized Model - Epoch {epoch}: Accuracy: {acc_quantized:.5f}, Inference Time: {time_quantized:.5f} seconds")

        #print("Epoch {}: Accuracy: {:.5f}".format(epoch, acc))

        if acc_non_quantized > best_acc:
            trials = 0
            best_acc = acc_non_quantized
            torch.save(model.state_dict(), 'best.pth')
            print(f"Epoch {epoch} best model saved with accuracy: {best_acc:.2f}")
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping on epoch {epoch}")
                break
