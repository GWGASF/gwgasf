# src/libs/data/data_utils.py

import numpy as np
import h5py
from pyts.image import GramianAngularField
import torch
import numpy as np
import random
import glob
import os

class ts_data:
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

def load_all_data(config):
    """Load all datasets (BBH, background, glitch) using data_path."""
    data_path = config['paths']['data_path'] + 'raw_data/'
    ifos = config['options']['ifos']
    apply_snr_filter = config['options']['apply_snr_filter']
    snr_threshold = config['options']['snr_threshold']
    file_paths = get_file_paths(data_path)
    
    def load_files(files, key):
        data_list = []
        for file in files:
            data_list.append(ts_data(file).get_data(key))
        return np.concatenate(data_list)

    # Load datasets
    bbhs = {ifo: load_files(file_paths['bbh'], ifo) for ifo in ifos}
    bgs = {ifo: load_files(file_paths[f'{ifo}_bg'], 'background_noise') for ifo in ifos}
    glitches = {ifo: load_files(file_paths[f'{ifo}_glitch'], 'glitch') for ifo in ifos}

    data = {'bbh': bbhs, 'bg': bgs, 'glitch': glitches}

    if apply_snr_filter:
        glitch_info = {ifo: load_files(file_paths[f'{ifo}_glitch'], 'glitch_info') for ifo in ifos}
        glitch_snr = {ifo: glitch_info[ifo]['snr'] for ifo in ifos}
        snr_glitches = find_high_snr_glitches(data, glitch_snr, snr_threshold)
        data = {'bbh': bbhs, 'bg': bgs, 'glitch': snr_glitches}

    return data

# Finds all files with these prefixes
def get_file_paths(data_path):
    """Return all file paths needed."""
    file_patterns = [
        'bbh_dataset_p*.hdf5',
        'H1_bg_dataset_p*.hdf5',
        'L1_bg_dataset_p*.hdf5',
        'H1_glitch_dataset_p*.hdf5',
        'L1_glitch_dataset_p*.hdf5'
    ]

    paths = {}
    for pattern in file_patterns:
        matched_files = glob.glob(os.path.join(data_path, pattern))
        if matched_files:
            if pattern.startswith('bbh'):
                key = pattern.split('_')[0]
            else:
                key = '_'.join(pattern.split('_')[:2])
            paths[key] = matched_files[::-1]  # Reverse the list of matched files

    return paths

def find_high_snr_glitches(data, glitch_snr, snr_threshold):
    """Find high SNR glitches."""
    snr_data = {}
    for ifo in data['glitch']:
        indices = np.where(glitch_snr[ifo] >= snr_threshold)
        snr_data[ifo] = data['glitch'][ifo][indices]
    return snr_data

def stack_arrays(data, config):
    """Stack arrays for the H1 and L1 detectors."""
    ifos = config['options']['ifos']
    num_samples = {
        'bbh': min(min([data['bbh'][ifo].shape[0] for ifo in ifos]), config['options']['num_bbh']),
        'bg': min(min([data['bg'][ifo].shape[0] for ifo in ifos]), config['options']['num_bg']),
        'glitch': min(min([data['glitch'][ifo].shape[0] for ifo in ifos]), config['options']['num_glitch'])
    }

    signals = {
        key: np.dstack([data[key][ifo][:num_samples[key]] for ifo in ifos])
        for key in num_samples
    }

    return signals

def convert_and_label_data(data, config):
    """Label the data and convert it to GASF format."""
    ifos = config['options']['ifos']
    anomaly_class = {'glitch': [1, 0, 0], 'bbh': [0, 1, 0], 'bg': [0, 0, 1]}
    GASF = GramianAngularField(image_size=194, sample_range=(-1, 1), method="summation")

    gasf_data = {}
    labels = {}
    
    for key in data.keys():
        ids = np.full((data[key].shape[0], 3), anomaly_class[key])
        img_decs = dict.fromkeys(ifos)
        for i, ifo in enumerate(ifos):
            img_decs[ifo] = GASF.transform(data[key][:, :, i])
        gasf_images = np.stack([img for img in img_decs.values()], axis=1)
        gasf_data[key] = gasf_images
        labels[key] = ids
    
    return gasf_data, labels

def save_gasf_to_hdf5(data, labels, config):
    """Save GASF data and labels to HDF5 file."""
    file_path = config['paths']['data_path'] + 'gasf_data/gasf_data.hdf5'
    with h5py.File(file_path, 'w') as f:
        for key in data:
            f.create_dataset(key, data=data[key])
            f.create_dataset(f'{key}_label', data=labels[key])

def load_gasf_from_hdf5(config):
    """Load GASF data and labels from HDF5 file."""
    file_path = config['paths']['data_path'] + 'gasf_data/gasf_data.hdf5'
    num_bbh = config['options']['num_bbh']
    num_bg = config['options']['num_bg']
    num_glitch = config['options']['num_glitch']

    with h5py.File(file_path, 'r') as f:
        data = {
            'bbh': np.array(f['bbh'][:num_bbh]),
            'bg': np.array(f['bg'][:num_bg]),
            'glitch': np.array(f['glitch'][:num_glitch])
        }
        labels = {
            'bbh': np.array(f['bbh_label'][:num_bbh]),
            'bg': np.array(f['bg_label'][:num_bg]),
            'glitch': np.array(f['glitch_label'][:num_glitch])
        }
    return data, labels

def split_dataset(data, labels, config):
    """Split data into training, testing, and validation sets using ratios from config."""
    def split(data, train_ratio, test_ratio, val_ratio):
        total_samples = len(data)
        num_train = int(train_ratio * total_samples)
        num_test = int(test_ratio * total_samples)
        num_val = total_samples - num_train - num_test

        return data[:num_train], data[num_train:num_train + num_test], data[num_train + num_test:]

    train_ratio = config['ratios']['train']
    test_ratio = config['ratios']['test']
    val_ratio = config['ratios']['validation']

    # Split each type of data
    split_data = {key: split(data[key], train_ratio, test_ratio, val_ratio) for key in data}
    split_labels = {key: split(labels[key], train_ratio, test_ratio, val_ratio) for key in labels}

    # Concatenate the splits to form the combined structure
    strains = {
        'training': np.concatenate([split_data['glitch'][0], split_data['bbh'][0], split_data['bg'][0]]),
        'testing': np.concatenate([split_data['glitch'][1], split_data['bbh'][1], split_data['bg'][1]]),
        'validation': np.concatenate([split_data['glitch'][2], split_data['bbh'][2], split_data['bg'][2]])
    }

    targets = {
        'training': np.concatenate([split_labels['glitch'][0], split_labels['bbh'][0], split_labels['bg'][0]]),
        'testing': np.concatenate([split_labels['glitch'][1], split_labels['bbh'][1], split_labels['bg'][1]]),
        'validation': np.concatenate([split_labels['glitch'][2], split_labels['bbh'][2], split_labels['bg'][2]])
    }

    return strains, targets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False