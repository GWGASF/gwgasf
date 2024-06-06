import numpy as np
import torch
from pyts.image import GramianAngularField
from gasf.utils import h5_thang
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, target, ifos=3, kernel_size=4096):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(target)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

def create_dataloaders(strains, targets):
    training_data = DataLoader(
        MyDataset(strains[0], targets[0]), 
        batch_size=32, 
        shuffle=True
    )

    testing_data = DataLoader(
        MyDataset(strains[2], targets[2]), 
        batch_size=32, 
        shuffle=True
    )

    validation_data = DataLoader(
        MyDataset(strains[1], targets[1]), 
        batch_size=32, 
        shuffle=True
    )

    return training_data, testing_data, validation_data

def prepare_data():
    # Initializes numpy and pytorch random seeds for reproducibility 
    seed = 55
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # Preparing the data

    ### Load dataset ###

    numSamples = 3000 

    bbh_signals_filename = '/home/dfredin/gwgasf/data/5_2_2024/bbh_dataset_p1.hdf5'
    bbh_info = h5_thang(bbh_signals_filename)
    H1_bbh = bbh_info.h5_data()['H1'][0:numSamples]
    L1_bbh = bbh_info.h5_data()['L1'][0:numSamples]

    H1_bg_filename = '/home/dfredin/gwgasf/data/5_2_2024/H1_bg_dataset_p1.hdf5'
    H1_bg_info = h5_thang(H1_bg_filename)
    H1_bg = H1_bg_info.h5_data()['background_noise'][0:numSamples]

    L1_bg_filename = '/home/dfredin/gwgasf/data/5_2_2024/L1_bg_dataset_p1.hdf5'
    L1_bg_info = h5_thang(L1_bg_filename)
    L1_bg = L1_bg_info.h5_data()['background_noise'][0:numSamples]

    H1_glitch_filename = '/home/dfredin/gwgasf/data/5_2_2024/H1_glitch_dataset_p1.hdf5'
    H1_glitch_info = h5_thang(H1_glitch_filename)
    H1_glitch = H1_glitch_info.h5_data()['glitch'][0:numSamples]

    L1_glitch_filename = '/home/dfredin/gwgasf/data/5_2_2024/L1_glitch_dataset_p1.hdf5'
    L1_glitch_info = h5_thang(L1_glitch_filename)
    L1_glitch = L1_glitch_info.h5_data()['glitch'][0:numSamples]

    variables = [H1_bbh, L1_bbh, H1_bg, H1_glitch, L1_bg, L1_glitch]
    names = ['H1 BBH', 'L1 BBH', 'H1 BG', 'H1 Glitch', 'L1 BG', 'L1 Glitch']

    for name, var in zip(names, variables):
        print(f"{name} Shape: {var.shape}")

    ### HIGH SNR GLITCH ###

    H1_glitch_snr = H1_glitch_info.h5_data()['glitch_info']
    L1_glitch_snr = L1_glitch_info.h5_data()['glitch_info']

    h1_indices = np.where(H1_glitch_snr['snr'] >= 15)
    H1_glitch_snr_high = H1_glitch_info.h5_data()['glitch'][h1_indices]

    l1_indices = np.where(L1_glitch_snr['snr'] >= 15)
    L1_glitch_snr_high = L1_glitch_info.h5_data()['glitch'][l1_indices]

    H1_glitch_snr_high = H1_glitch_snr_high[0:len(L1_glitch_snr_high)]
    L1_glitch_snr_high = L1_glitch_snr_high[0:len(L1_glitch_snr_high)]

    glitch_highsnr_signals = np.dstack((H1_glitch_snr_high, L1_glitch_snr_high))
    glitch_signals = glitch_highsnr_signals

    del L1_glitch_snr_high, H1_glitch_snr_high, H1_glitch_snr, L1_glitch_snr, glitch_highsnr_signals

    ### STACK ARRAYS ###

    bbh_signals = np.dstack((H1_bbh, L1_bbh))
    bg_signals = np.dstack((H1_bg, L1_bg))
    # glitch_signals = np.dstack((H1_glitch, L1_glitch))

    del H1_bbh, L1_bbh, H1_bg, L1_bg, H1_glitch, L1_glitch

    x_train = np.concatenate((glitch_signals, bbh_signals, bg_signals))

    anomaly_class = {
        'Glitch': [1, 0, 0],
        'Signal': [0, 1, 0],
        'Background': [0, 0, 1]
    }

    glitch_train_ids = np.full((glitch_signals.shape[0], 3), anomaly_class['Glitch'])
    bbh_train_ids = np.full((bbh_signals.shape[0], 3), anomaly_class['Signal'])
    bg_train_ids = np.full((bg_signals.shape[0], 3), anomaly_class['Background'])

    y_train_ids = np.concatenate((glitch_train_ids, bbh_train_ids, bg_train_ids), axis=0)
    del bbh_signals, bg_signals, glitch_signals
    del glitch_train_ids, bbh_train_ids, bg_train_ids

    n_train = 2500
    idx_train = np.random.randint(0, len(x_train), size=n_train)
    y_train = y_train_ids[idx_train]

    ### GASF CONVERSION ###

    GASF = GramianAngularField(image_size=194, sample_range=(-1,1), method="summation")
    img_x_train_dec1 = GASF.transform(x_train[idx_train,:,0])
    img_x_train_dec2 = GASF.transform(x_train[idx_train,:,1])

    img_x_train = np.stack((img_x_train_dec1, img_x_train_dec2), axis=1)
    del img_x_train_dec1, img_x_train_dec2

    ### Split into training and validation datasets ###

    total_samples = img_x_train.shape[0]
    num_train = int(0.8 * total_samples)
    num_test = int(0.15 * total_samples)
    num_val = total_samples - num_train - num_test

    x_train_data = img_x_train[:num_train]
    x_test_data = img_x_train[num_train:num_train + num_test]
    x_val_data = img_x_train[num_train + num_test:]

    y_train_data = y_train[:num_train]
    y_test_data = y_train[num_train:num_train + num_test]
    y_val_data = y_train[num_train + num_test:]

    strains = [x_train_data, x_test_data, x_val_data]
    targets = [y_train_data, y_test_data, y_val_data]
    labels = ['Training', 'Testing', 'Validation']

    array_shapes = [arr.shape for arr in strains]
    for i, shape in enumerate(array_shapes):
        print(f"{labels[i]} Data Shape: {shape}")

    array_shapes = [arr.shape for arr in targets]
    for i, shape in enumerate(array_shapes):
        print(f"{labels[i]} Targets Shape: {shape}")

    training_data, testing_data, validation_data = create_dataloaders(strains, targets)

    return training_data, testing_data, validation_data, device, x_train, y_train, idx_train, img_x_train

if __name__ == "__main__":
    prepare_data()
