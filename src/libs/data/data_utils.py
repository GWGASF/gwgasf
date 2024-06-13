# libs/data/data_utils.py

import numpy as np
import h5py
from pyts.image import GramianAngularField

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

def get_file_paths(data_path, ifos):
    """Return all file paths needed."""
    paths = {
        'bbh': data_path + 'bbh_dataset_p1.hdf5',
        'H1_bg': data_path + 'H1_bg_dataset_p1.hdf5',
        'L1_bg': data_path + 'L1_bg_dataset_p1.hdf5',
        'H1_glitch': data_path + 'H1_glitch_dataset_p1.hdf5',
        'L1_glitch': data_path + 'L1_glitch_dataset_p1.hdf5'
    }
    return paths

def load_data(file_path):
    """Load data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = {key: np.array(f[key]) for key in f.keys()}
    return data

def load_all_data(data_path, ifos, apply_snr_filter, snr_threshold):
    """Load all datasets (BBH, background, glitch) using data_path."""
    file_paths = get_file_paths(data_path, ifos)
    
    # Load datasets
    bbhs = {ifo: ts_data(file_paths['bbh']).get_data(ifo) for ifo in ifos}
    bgs = {ifo: ts_data(file_paths[f'{ifo}_bg']).get_data('background_noise') for ifo in ifos}
    glitches = {ifo: ts_data(file_paths[f'{ifo}_glitch']).get_data('glitch') for ifo in ifos}

    data = {'bbh': bbhs, 'bg': bgs, 'glitch': glitches}

    if apply_snr_filter:
        data = find_high_snr_glitches(data, snr_threshold)

    return data

def find_high_snr_glitches(data, snr_threshold):
    """Find high SNR glitches."""
    for ifo in data['glitch']:
        glitch_snr = data['glitch'][ifo]['glitch_info']
        indices = np.where(glitch_snr['snr'] >= snr_threshold)
        data['glitch'][ifo] = data['glitch'][ifo]['glitch'][indices]
    return data

def stack_arrays(data, ifos):
    """Stack arrays for the H1 and L1 detectors."""
    bbh_signals = np.dstack(tuple([data['bbh'][ifo] for ifo in ifos]))
    bg_signals = np.dstack(tuple([data['bg'][ifo] for ifo in ifos]))
    glitch_signals = np.dstack(tuple([data['glitch'][ifo] for ifo in ifos]))
    
    signals = {'glitch': glitch_signals, 'bbh': bbh_signals, 'bg': bg_signals}
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

def save_gasf_to_hdf5(data, labels, file_path):
    """Save GASF data and labels to HDF5 file."""
    with h5py.File(file_path, 'w') as f:
        for key in data:
            f.create_dataset(key, data=data[key])
            f.create_dataset(f'{key}_label', data=labels[key])

def load_gasf_from_hdf5(file_path, num_bbh, num_bg, num_glitch):
    """Load GASF data and labels from HDF5 file."""
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
