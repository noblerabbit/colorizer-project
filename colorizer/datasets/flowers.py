# Flowers dataset.

import os
import h5py
import numpy as np
import time
from psutil import virtual_memory

from sklearn.model_selection import train_test_split

DATA_FILENAME = "../hdf5-image-dataset-builder/data.hdf5"

class FlowersDataset():
    def __init__(self, **dataset_args):
        if not os.path.exists(DATA_FILENAME):
            print ("Please provide path to datafile")
        self.input_shape = (256, 256, 1)
        self.output_shape = (256, 256, 2)
        self.train_val_split = 0.2
            
        if "train_val_split" in dataset_args:
            self.train_val_split = dataset_args["train_val_split"]

    def load_data(self):
        with h5py.File(DATA_FILENAME) as f:
            self.X = [file for file in f.keys() if file.startswith("X")]
            self.Y = [file for file in f.keys() if file.startswith("Y")]
            
            #form X and Y pick  random Xtest, Ytest, X_train, Y_train filenames
            
            #get input and output dims of data
            width_x, height_x = f[self.X[0]][:].shape
            width_y, heigth_y, depth_y = f[self.Y[0]][:].shape
        
        # create train val split
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
        self.X, self.Y, test_size=self.train_val_split, random_state=42)
        print(len(self.X_train), len(self.Y_train))

        
        #define numpy arrays
        self.Xdata_train = np.zeros((len(self.X_train), width_x, height_x, 1))
        self.Ydata_train = np.zeros((len(self.Y_train), width_y, heigth_y, depth_y))
        self.Xdata_val = np.zeros((len(self.X_val), width_y, height_x, 1))
        self.Ydata_val =np.zeros((len(self.Y_val), width_x, heigth_y, depth_y))
        
        print("[INFO] Total of Samples: {}, ".format(len(self.X)))
        print('[INFO] Number of Training samples: {}'.format(len(self.X_train)))
        print("[INFO] Number of Validation samples: {}".format(len(self.Y_train)))
        
        print("[INFO] Size of Xdata_train = {} MB".format(round(self.Xdata_train.nbytes/1000000,2)))
        print("[INFO] Size of Ydata_train = {} MB".format(round(self.Ydata_train.nbytes/1000000, 2)))
        
        #loading the data to memory
        print("[INFO] Total size of the memory (RAM): {} MB".format(round(virtual_memory().total/1000000),2))
        print("[INFO] Loading data to memory...")
        
        start = time.time()
        with h5py.File(DATA_FILENAME) as f:
            # load training data
            for i in range(len(self.X_train)):
                self.Xdata_train[i,:,:,0] = f[self.X_train[i]]
                self.Ydata_train[i,:,:,:] = f[self.Y_train[i]]
            # load validation data
            for i in range(len(self.X_val)):
                self.Xdata_val[i,:,:,0] = f[self.X_val[i]]
                self.Ydata_val[i,:,:,:] = f[self.Y_val[i]]
    
        print("[INFO] Data Loaded in {} seconds.".format(round(time.time() - start)))
    
    def __repr__(self):
        return (
            f'Flowers Dataset\n'
            f'Num of images: {len(self.X)}\n'
            f'Input shape: {self.input_shape}\n'
            f'Output shape: {self.output_shape}\n'
            f'Train samples : {len(self.X_train)} samples\n'
            f'Validation samples: {len(self.X_val)} samples\n'
        )


if __name__ == '__main__':
    data = FlowersDataset()
    data.load_data()
    print(data)

    # TODO RESHUFFLE DATASET
    # TODO GENERATE TEST AND TRAIN DATASETS
    # input can be uint8 and it will consume much less memory! dtype='uint8' inside np.zeros