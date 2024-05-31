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

class ts_data():
    def __init__(self, file_name):
        self.file_name = file_name
        with h5py.File(self.file_name, 'r') as f:
            self.keys = list(f.keys())

    def get_keys(self):
        return self.keys

    def get_data(self, key):
        with h5py.File(self.file_name, 'r') as f:
            data = f[key][:]
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
    bbh_files = [
        '/home/chiajui.chou/GW-anomaly-detection/data/dataset_inj/bbh_dataset_p1.hdf5',
    ]
    bg_files = dict.fromkeys(ifos)
    glitch_files = dict.fromkeys(ifos)
    for ifo in ifos:
        bg_files[ifo] = [
            f'/home/chiajui.chou/GW-anomaly-detection/data/dataset_noise/{ifo}_bg_dataset_p1.hdf5'
        ]
        glitch_files[ifo] = [
            f'/home/chiajui.chou/GW-anomaly-detection/data/dataset_noise/{ifo}_glitch_dataset_p1.hdf5',
        ]

    # Loading datafiles and only load a slice of the total data.
    bbhs = dict.fromkeys(ifos)
    bgs = dict.fromkeys(ifos)
    glitches = dict.fromkeys(ifos)
    bbh_data = ts_data(bbh_files[0])
    for ifo in ifos:
        bbhs[ifo] = bbh_data.get_data(key=ifo)
        bg_data = ts_data(bg_files[ifo][0])
        bgs[ifo] = bg_data.get_data(key='background_noise')
        glitch_data = ts_data(glitch_files[ifo][0])
        glitches[ifo] = glitch_data.get_data(key='glitch')

        # print(bbhs[ifo].shape)
        # print(bgs[ifo].shape)
        # print(glitches[ifo].shape)

    # Stacking the arrays for the H1 and L1 detectors
    num_bbh = 3000
    num_bg = 3000
    num_glitch = 3000
    bbh_signals = np.dstack(tuple([bbhs[ifo][:num_bbh] for ifo in ifos]))
    bg_signals = np.dstack(tuple([bgs[ifo][:num_bg] for ifo in ifos]))
    glitch_signals = np.dstack(tuple([glitches[ifo][:num_glitch] for ifo in ifos]))
    signals = {
        'glitch': glitch_signals,
        'bbh': bbh_signals,
        'bg': bg_signals,
    }
    for key in signals.keys():
        print(signals[key].shape)

    # Multiclassifier labels
    anomaly_class = {
        'glitch': [1, 0, 0],
        'bbh': [0, 1, 0],
        'bg': [0, 0, 1],
    }

    # GASF Conversion
    for key in signals.keys():
        print(key)
        ids = np.full((signals[key].shape[0], 3), anomaly_class[key])
        img_decs = dict.fromkeys(ifos)
        for i, ifo in enumerate(ifos):
            img_decs[ifo] = gasf_conversion(signals[key][:,:,i])

        imgs = np.stack([img for img in img_decs.values()], axis=1)
        print(imgs.shape)

        with h5py.File(f'/home/chiajui.chou/GASF/gwgasf/src/gasf_data/gasf_img_{key}_data.hdf5', 'w') as f:
            img_data = imgs
            f.create_dataset(
                key,
                shape=img_data.shape,
                dtype=img_data.dtype,
                data=img_data,
            )
            label_data = ids
            f.create_dataset(
                f'{key}_label',
                shape=label_data.shape,
                dtype=label_data.dtype,
                data=label_data,
            )
        

if __name__ == "__main__":
    main()