# main.py

import torch
from libs.utils.argument_parser import parse_arguments
from libs.data.data_utils import load_all_data, stack_arrays, convert_and_label_data, save_gasf_to_hdf5, load_gasf_from_hdf5, split_dataset
from libs.data.create_dataloaders import create_dataloaders
from libs.train.train_model import train_model
from libs.train.model_utils import save_best_model, load_best_model, save_checkpoint
from libs.architecture.cnn_model import CNNModel
from libs.utils.analysis_utils import calculate_metrics, calculate_confusion_matrix
from libs.utils.plot_utils import plot_confusion_matrix, save_confusion_matrix, plot_gasf, save_plot

def main():
    # Load arguments from the TOML file
    config = parse_arguments('/home/dfredin/gwgasf/src/arguments.toml')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    ifos = config['options']['ifos']
    if config['options']['create_new_gasf']:
        data = load_all_data(config['paths']['data_path'] + 'raw_data/', ifos, config['options']['apply_snr_filter'], config['options']['snr_threshold'])
        data = stack_arrays(data, ifos)
        gasf_data, labels = convert_and_label_data(data, config)
        save_gasf_to_hdf5(gasf_data, labels, config['paths']['data_path'] + 'gasf_data/gasf_data.hdf5')
    else:
        num_bbh = config['options']['num_bbh']
        num_bg = config['options']['num_bg']
        num_glitch = config['options']['num_glitch']
        gasf_data, labels = load_gasf_from_hdf5(config['paths']['data_path'] + 'gasf_data/gasf_data.hdf5', num_bbh, num_bg, num_glitch)

    strains, targets = split_dataset(gasf_data, labels, config)
    
    training_data, validation_data, testing_data = create_dataloaders(strains, targets, config['hyperparameters']['batch_size'])

    # Model Training
    train_model(config, device, training_data, validation_data)

    # Evaluation and Analysis
    best_model = CNNModel().to(device)
    load_best_model(config['paths']['models_path'], best_model)
    
    # Plot confusion matrices for training, validation, and testing datasets
    for dataset, name in zip([training_data, validation_data, testing_data], ['Training', 'Validation', 'Test']):
        conf_matrix = calculate_confusion_matrix(best_model, dataset, device)
        plot_confusion_matrix(conf_matrix, name, config['paths']['results_path'])

    # Visualization
    # fig = plot_gasf(gasf_data['bbh'], "GASF Data")  # Example for bbh
    # save_plot(fig, config['paths']['results_path'] + 'gasf_plot.png')
    
    # fig = plot_time_series(data['bbh'], "Time Series Data")  # Example for bbh
    # save_plot(fig, config['paths']['results_path'] + 'time_series_plot.png')

if __name__ == "__main__":
    main()
