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

from libs.data.s3_utils import S3_session


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
    """Load all datasets (BBH, background, glitch) using data path."""
    logging.info("Loading raw data...")
    fs = create_s3_filesystem(config)
    data_path = config['paths']['data_path']  # raw dataset path
    apply_snr_filter = config['options']['apply_snr_filter']
    snr_threshold = config['options']['snr_threshold']
    ifos = ["H1", "L1"]

    file_paths = get_file_paths(data_path, fs)

   # Load datasets
    bbhs = {ifo: load_files(file_paths['bbh'], ifo, config) for ifo in ifos} if 'bbh' in file_paths else None
    bgs = {ifo: load_files(file_paths['bg'], ifo, config) for ifo in ifos} if 'bg' in file_paths else None
    glitches = {ifo: load_files(file_paths['glitch'], ifo, config) for ifo in ifos} if 'glitch' in file_paths else None

    data = {'bbh': bbhs, 'bg': bgs, 'glitch': glitches}

    if apply_snr_filter and 'glitch' in file_paths:
        glitch_info = {ifo: load_files(file_paths['glitch'], f'{ifo}_glitch_info', config) for ifo in ifos}
        glitch_snr = {ifo: glitch_info[ifo]['snr'] for ifo in ifos}
        snr_glitches = find_high_snr_glitches(data, glitch_snr, snr_threshold)
        data['glitch'] = snr_glitches  # Replace glitches with high-SNR ones

    return data

# Finds all files with these prefixes
def get_file_paths(data_path, fs):
    """Return all file paths needed using s3fs, sorted numerically."""
    file_patterns = [
        'bbh_dataset_p*.hdf5',
        'H1_bg_dataset_p*.hdf5',
        'L1_bg_dataset_p*.hdf5',
        'H1_glitch_dataset_p*.hdf5',
        'L1_glitch_dataset_p*.hdf5',
        'bg_data_ids_*-*.hdf5', 
        'glitch_data_ids_*-*.hdf5',  
        'injection_data_type_bbh_ids_*-*.hdf5'
    ]
## NOTE: ADD FILE PATHS OF NEW DATA
    paths = {}

    for pattern in file_patterns:
        matched_files = fs.glob(os.path.join(data_path, pattern))
        if matched_files:
            matched_files.sort(key=extract_start_id)  # Sort using START_ID

            # Assign category keys for old and new formats
            if pattern.startswith('bbh'):
                key = 'bbh'
            elif pattern.startswith('H1_bg') or pattern.startswith('L1_bg'):
                key = pattern.split('_')[0] + '_bg'
            elif pattern.startswith('H1_glitch') or pattern.startswith('L1_glitch'):
                key = pattern.split('_')[0] + '_glitch'
            elif pattern.startswith('bg_data_ids'):
                key = 'bg'
            elif pattern.startswith('glitch_data_ids'):
                key = 'glitch'
            elif pattern.startswith('injection_data_type_bbh'):
                key = 'bbh'  # Extract BBH injection files

            paths[key] = matched_files

    return paths

def find_high_snr_glitches(data, glitch_snr, snr_threshold):
    """Find high SNR glitches."""
    snr_data = {}
    for ifo in data['glitch']:
        indices = np.where(glitch_snr[ifo] >= snr_threshold)
        snr_data[ifo] = data['glitch'][ifo][indices]
    return snr_data

def load_files(files, ifo, config):
    """Load HDF5 files from S3 and extract data for a given interferometer (H1 or L1)."""
    fs = create_s3_filesystem(config)
    data_list = []

    for file in files:
        with fs.open(file, 'rb') as f:
            data = ts_data(f).get_data(ifo)  # Load data for the specific IFO
            if data is not None:
                data_list.append(data)

    return np.concatenate(data_list) if data_list else None

def extract_start_id(path):
    """Extract START_ID from filename for sorting."""
    match = re.search(r'ids_(\d+)-\d+', path)
    return int(match.group(1)) if match else float('inf')

# # Function to extract the numerical part from the filename
# def extract_number_from_path(path):
#     match = re.search(r'p(\d+)', path)  # Look for the number after 'p' in the filename
#     return int(match.group(1)) if match else float('inf')  # Use infinity if no match is found

def stack_arrays(data):
    """Stack arrays for the H1 and L1 detectors with varying sample sizes."""
    logging.info("Stacking arrays...")
    ifos = ["H1", "L1"]

    # Determine the number of samples for each signal type by finding the minimum across detectors
    num_samples = {
        'bbh': min([data['bbh'][ifo].shape[0] for ifo in ifos]),
        'bg': min([data['bg'][ifo].shape[0] for ifo in ifos]),
        'glitch': min([data['glitch'][ifo].shape[0] for ifo in ifos])
    }

    # Stack the samples across detectors for each signal type
    signals = {
        'bbh': np.dstack([data['bbh'][ifo][:num_samples['bbh']] for ifo in ifos]),
        'bg': np.dstack([data['bg'][ifo][:num_samples['bg']] for ifo in ifos]),
        'glitch': np.dstack([data['glitch'][ifo][:num_samples['glitch']] for ifo in ifos])
    }

    return signals

def convert_and_label_data(data):
    """Label the data and convert it to GASF format."""
    logging.info("Converting to GASF...")
    ifos = ["H1", "L1"]
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
    # fs = create_s3_filesystem(config)
    file_path_s3 = config['paths']['data_path_gasf'] + 'gasf_data.hdf5'

    # Write to a temporary file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp_file:
        tmp_file.name = 'gasf_data.hdf5'  # Rename for clarity
        temp_file_path = tmp_file.name
        logging.info(f"Created temporary file for GASF data: {temp_file_path}")
        with h5py.File(temp_file_path, 'w') as hf:
            for key in data:
                hf.create_dataset(key, data=data[key])
                hf.create_dataset(f'{key}_label', data=labels[key])
    
    # Attempt to upload the temporary file to S3 using fs.put
    try:
        # logging.info(f"Uploading GASF data to S3 at {file_path_s3}")
        s3 = S3_session(config['s3'])
        s3.upload(
            file_name=temp_file_path,
            upload_dir = config['paths']['data_path_gasf'].replace(f"s3://{s3.bucket}/", "").rstrip("/")
        )
        # logging.info(f"GASF data saved to {config['paths']['data_path_gasf']}")

        # fs.put(temp_file_path, file_path_s3)  # Use fs.put for direct file copy to S3
        # logging.info(f"Successfully uploaded GASF data to {file_path_s3}")
    except Exception as e:
        logging.error(f"Failed to upload GASF data to S3: {e}")

    # Clean up the local temporary file
    try:
        os.remove(temp_file_path)
        logging.info(f"Deleted temporary file for GASF data: {temp_file_path}")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {temp_file_path}: {e}")

def load_gasf_from_hdf5(config):
    """Load GASF data and labels from HDF5 file in S3."""
    logging.info("Loading GASF data...")
    file_path = config['paths']['data_path_gasf'] + 'gasf_data.hdf5'
    fs = create_s3_filesystem(config)
    
    with fs.open(file_path, 'rb') as f:
        with h5py.File(f, 'r') as hf:
            
            # Check if the user wants to select a set number of samples
            if config['options']['select_samples']:
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

            else:  # Load all samples if select_samples is false
                data = {
                    'bbh': np.array(hf['bbh'][:]),
                    'bg': np.array(hf['bg'][:]),
                    'glitch': np.array(hf['glitch'][:])
                }
                labels = {
                    'bbh': np.array(hf['bbh_label'][:]),
                    'bg': np.array(hf['bg_label'][:]),
                    'glitch': np.array(hf['glitch_label'][:])
                }

    return data, labels

def split_dataset(data, labels, config):
    """Split data into training, testing, and validation sets using ratios from config, with optional shuffling."""
    logging.info("Splitting dataset...")
    # Set the seed for reproducibility
    seed = config['hyperparameters']['seed']
    set_seed(seed)

    def split(data, train_ratio, test_ratio, val_ratio):
        total_samples = len(data)
        num_train = round(train_ratio * total_samples)
        num_test = round(test_ratio * total_samples)
        num_val = total_samples - num_train - num_test  # Adjust to ensure the total adds up

        return data[:num_train], data[num_train:num_train + num_test], data[num_train + num_test:]

    train_ratio = config['ratios']['train']
    test_ratio = config['ratios']['test']
    val_ratio = config['ratios']['validation']

    # Check whether to shuffle the data based on config
    if config['options']['shuffle_data']:
        for key in data:
            indices = np.arange(len(data[key]))
            np.random.shuffle(indices)
            data[key] = np.array(data[key])[indices]
            labels[key] = np.array(labels[key])[indices]

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
    """Set the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False