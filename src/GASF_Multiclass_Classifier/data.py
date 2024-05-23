#!/bin/python

import numpy as np
import h5py
import torch
import torch.nn.functional as F

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pyts.image import GramianAngularField
from tqdm import tqdm

import random

seed = 55
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def read_data(
        file_name,
        ifo,
        key,
        slice = [0, 1000],
):
    with h5py.File(file_name, 'r') as f:
        data = f[key][slice[0]:slice[1]]
    return data

def gasf_conversion(
        data,
        image_size = 194,
        sample_range = (-1, 1),
        method = 'summation',
):
    GASF = GramianAngularField(
        image_size=image_size,
        sample_range=sample_range,
        method=method,
    )
    return GASF.transform(data)

def main():
    ifos = ["H1", "L1"]
    # Loading datafiles and only load a slice of the total data.
    numSamples = 3
    bbhs = dict.fromkeys(ifos)
    bgs = dict.fromkeys(ifos)
    glitches = dict.fromkeys(ifos)
    for ifo in ifos:
        file_name = '/home/chiajui.chou/GW-anomaly-detection/data/dataset_inj/bbh_dataset_p1.hdf5'
        bbhs[ifo] = read_data(
            file_name=file_name,
            ifo=ifo,
            key=ifo,
            slice=[0, numSamples]
        )
        file_name = f'/home/chiajui.chou/GW-anomaly-detection/data/dataset_noise/{ifo}_bg_dataset_p1.hdf5'
        bgs[ifo] = read_data(
            file_name=file_name,
            ifo=ifo,
            key='background_noise',
            slice=[0, numSamples]
        )
        file_name = f'/home/chiajui.chou/GW-anomaly-detection/data/dataset_noise/{ifo}_glitch_dataset_p1.hdf5'
        glitches[ifo] = read_data(
            file_name=file_name,
            ifo=ifo,
            key='glitch',
            slice=[0, numSamples]
        )

    # Stacking the arrays for the H1 and L1 detectors
    bbh_signals = np.dstack(tuple([bbhs[ifo] for ifo in ifos]))
    bg_signals = np.dstack(tuple([bgs[ifo] for ifo in ifos]))
    glitch_signals = np.dstack(tuple([glitches[ifo] for ifo in ifos]))
    x_train = np.concatenate((glitch_signals, bbh_signals, bg_signals))
    print(x_train.shape)

    # Multiclassifier labels
    anomaly_class = {
        'Glitch': [1, 0, 0],
        'BBH': [0, 1, 0],
        'Background': [0, 0, 1],
    }
    glitch_train_ids = np.full((glitch_signals.shape[0], 3), anomaly_class['Glitch'])
    bbh_train_ids = np.full((bbh_signals.shape[0], 3), anomaly_class['BBH'])
    bg_train_ids = np.full((bg_signals.shape[0], 3), anomaly_class['Background'])
    y_train_ids = np.concatenate((glitch_train_ids, bbh_train_ids, bg_train_ids), axis=0)

    # GASF Conversion
    # Generate the idx_train, idx_val to randomly sample from the training set and validation set.
    idx_train = np.random.randint(0, len(x_train), size=numSamples)

    y_train = y_train_ids[idx_train]
    # Convert training data to images.
    img_x_train_decs = dict.fromkeys(ifos)
    for i, ifo in enumerate(ifos):
        img_x_train_decs[ifo] = gasf_conversion(x_train[idx_train,:,i])
        
    print(img_x_train_decs['H1'].shape)
    print(img_x_train_decs['L1'].shape)
    img_x_train = np.stack([img for img in img_x_train_decs.values()], axis=1)
    print(img_x_train.shape)

    with h5py.File('/home/chiajui.chou/GASF/gwgasf/src/gasf_data/gasf_img_train_data.hdf5', 'w') as f:
        f.create_dataset(
            'x_train',
            shape=img_x_train.shape,
            dtype=img_x_train.dtype,
            data=img_x_train,
        )
        f.create_dataset(
            'y_train',
            shape=y_train.shape,
            dtype=y_train.dtype,
            data=y_train,
        )

if __name__ == "__main__":
    main()