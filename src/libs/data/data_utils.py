# src/libs/data/data_utils.py

from libs.utils.s3_helper import create_s3_filesystem
import numpy as np
import h5py
from pyts.image import GramianAngularField
import torch
import numpy as np
import random
import glob
import os
import tempfile
import logging
import re

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
    fs = create_s3_filesystem()
    inj_data_path = config['paths']['data_path_inj']  # Injected dataset path
    noise_data_path = config['paths']['data_path_noise']  # Noise dataset path
        
    ifos = config['options']['ifos']
    apply_snr_filter = config['options']['apply_snr_filter']
    snr_threshold = config['options']['snr_threshold']
    
    # Load files from S3 paths
    inj_file_paths = get_file_paths(inj_data_path, fs)
    noise_file_paths = get_file_paths(noise_data_path, fs)
    def load_files(files, key):
        data_list = []
        for file in files:
            with fs.open(file, 'rb') as f:
                data_list.append(ts_data(f).get_data(key))
        return np.concatenate(data_list)

    # Load datasets
    bbhs = {ifo: load_files(inj_file_paths['bbh'], ifo) for ifo in ifos}
    bgs = {ifo: load_files(noise_file_paths[f'{ifo}_bg'], 'background_noise') for ifo in ifos}
    glitches = {ifo: load_files(noise_file_paths[f'{ifo}_glitch'], 'glitch') for ifo in ifos}

    data = {'bbh': bbhs, 'bg': bgs, 'glitch': glitches}

    if apply_snr_filter:
        glitch_info = {ifo: load_files(noise_file_paths[f'{ifo}_glitch'], 'glitch_info') for ifo in ifos}
        glitch_snr = {ifo: glitch_info[ifo]['snr'] for ifo in ifos}
        snr_glitches = find_high_snr_glitches(data, glitch_snr, snr_threshold)
        data = {'bbh': bbhs, 'bg': bgs, 'glitch': snr_glitches}

    return data

# Finds all files with these prefixes
def get_file_paths(data_path, fs):
    """Return all file paths needed using s3fs, sorted numerically."""
    file_patterns = [
        'bbh_dataset_p*.hdf5',
        'H1_bg_dataset_p*.hdf5',
        'L1_bg_dataset_p*.hdf5',
        'H1_glitch_dataset_p*.hdf5',
        'L1_glitch_dataset_p*.hdf5'
    ]

    paths = {}

    # Function to extract the numerical part from the filename
    def extract_number_from_path(path):
        match = re.search(r'p(\d+)', path)  # Look for the number after 'p' in the filename
        return int(match.group(1)) if match else float('inf')  # Use infinity if no match is found

    for pattern in file_patterns:
        matched_files = fs.glob(os.path.join(data_path, pattern))
        if matched_files:
            # Sort the matched files based on the extracted number
            matched_files.sort(key=extract_number_from_path)
            
            if pattern.startswith('bbh'):
                key = pattern.split('_')[0]
            else:
                key = '_'.join(pattern.split('_')[:2])
            paths[key] = matched_files
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
    """Save GASF data and labels to HDF5 file in S3."""
    fs = create_s3_filesystem()
    file_path_s3 = config['paths']['data_path'] + 'gasf_data.hdf5'

    # Write to a temporary file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp_file:
        temp_file_path = tmp_file.name
        logging.info(f"Temporary file created: {temp_file_path}")
        with h5py.File(temp_file_path, 'w') as hf:
            for key in data:
                hf.create_dataset(key, data=data[key])
                hf.create_dataset(f'{key}_label', data=labels[key])

    # Attempt to upload the temporary file to S3 using fs.put
    try:
        logging.info(f"Uploading {temp_file_path} to S3 at {file_path_s3}")
        fs.put(temp_file_path, file_path_s3)  # Use fs.put for direct file copy to S3
        logging.info(f"Successfully uploaded {temp_file_path} to {file_path_s3}")
    except Exception as e:
        logging.error(f"Failed to upload {temp_file_path} to S3: {e}")

    # Clean up the local temporary file
    try:
        os.remove(temp_file_path)
        logging.info(f"Temporary file {temp_file_path} deleted.")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {temp_file_path}: {e}")

def load_gasf_from_hdf5(config):
    """Load GASF data and labels from HDF5 file in S3."""
    file_path = config['paths']['data_path'] + 'gasf_data.hdf5'
    fs = create_s3_filesystem()
    
    with fs.open(file_path, 'rb') as f:
        with h5py.File(f, 'r') as hf:
            num_bbh = config['options']['num_bbh']
            num_bg = config['options']['num_bg']
            num_glitch = config['options']['num_glitch']
            
            data = {
                'bbh': np.array(hf['bbh'][:num_bbh]),
                'bg': np.array(hf['bg'][:num_bg]),
                'glitch': np.array(hf['glitch'][:num_glitch])
            }
            labels = {
                'bbh': np.array(hf['bbh_label'][:num_bbh]),
                'bg': np.array(hf['bg_label'][:num_bg]),
                'glitch': np.array(hf['glitch_label'][:num_glitch])
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