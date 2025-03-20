# src/main.py

import torch
import logging
import argparse
from libs.utils.argument_parser import parse_arguments
from libs.data.data_utils import load_all_data, stack_arrays, convert_and_label_data, save_gasf_to_hdf5, load_gasf_from_hdf5, split_dataset
from libs.data.create_dataloaders import create_dataloaders
from libs.train.train_model import train_model
from libs.train.model_utils import load_best_model
from libs.utils.analysis_utils import calculate_confusion_matrix
from libs.utils.plot_utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO)

def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    if config['options']['create_new_gasf']:
        data = load_all_data(config)
        data = stack_arrays(data)
        gasf_data, labels = convert_and_label_data(data)
        save_gasf_to_hdf5(gasf_data, labels, config)
    else:
        gasf_data, labels = load_gasf_from_hdf5(config)
        strains, targets = split_dataset(gasf_data, labels, config)
        training_data, validation_data, testing_data = create_dataloaders(strains, targets, config)
        
        if config['options']['train_model']:
            # Model Training
            logging.info("Training model...")
            train_model(config, device, training_data, validation_data)

        else:
            logging.info("Training skipped. Loading the best model for evaluation...")
            # Evaluation and Analysis
            # Plot confusion matrices for training, validation, and testing datasets
            for dataset, name in zip([training_data, validation_data, testing_data], ['Training', 'Validation', 'Test']):
                conf_matrix = calculate_confusion_matrix(load_best_model(config, device), dataset, config, name)
                plot_confusion_matrix(conf_matrix, name, config)

            # Visualization
            # fig = plot_gasf(gasf_data['bbh'], "GASF Data")  # Example for bbh
            # save_plot(fig, config['paths']['results_path'] + 'gasf_plot.png')
            
            # fig = plot_time_series(data['bbh'], "Time Series Data")  # Example for bbh
            # save_plot(fig, config['paths']['results_path'] + 'time_series_plot.png')

if __name__ == "__main__":
    # Load arguments from the TOML file
    config = parse_arguments('gasf/src/arguments.yaml')
    parser = argparse.ArgumentParser()

    parser.add_argument('--newgasf', action='store_true', default = config['options']['create_new_gasf'])
    parser.add_argument('--train', action='store_true', default = config['options']['train_model'])

    parser.add_argument('--nbbh', type=int, default = config['options']['num_bbh'])
    parser.add_argument('--nbg', type=int, default = config['options']['num_bg'])
    parser.add_argument('--nglitch', type=int, default = config['options']['num_glitch'])

    parser.add_argument('--epoch', type=int, default = config['hyperparameters']['epochs'])
    parser.add_argument('--batch', type=int, default = config['hyperparameters']['batch_size'])

    args = parser.parse_args()
    config['options']['create_new_gasf'] = args.newgasf
    config['options']['train_model'] = args.train

    config['options']['num_bbh'] = args.nbbh
    config['options']['num_bg'] = args.nbg
    config['options']['num_glitch'] = args.nglitch

    config['hyperparameters']['epochs'] = args.epoch
    config['hyperparameters']['batch_size'] = args.batch    

    main(config)
