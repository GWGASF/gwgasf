import torch
import logging
from libs.utils.argument_parser import parse_arguments
from libs.data.data_utils import load_all_data, stack_arrays, convert_and_label_data, save_gasf_to_hdf5, load_gasf_from_hdf5, split_dataset
from libs.data.create_dataloaders import create_dataloaders
from libs.train.train_model import train_model
from libs.train.model_utils import load_best_model
from libs.utils.analysis_utils import calculate_confusion_matrix
from libs.utils.plot_utils import plot_confusion_matrix, plot_gasf

logging.basicConfig(level=logging.INFO)


def main():
    # Load arguments from the TOML file
    config = parse_arguments('src/arguments.toml')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    if config['options']['create_new_gasf']:
        data = load_all_data(config)
        data = stack_arrays(data, config)
        gasf_data, labels = convert_and_label_data(data, config)
        save_gasf_to_hdf5(gasf_data, labels, config)
    else:
        gasf_data, labels = load_gasf_from_hdf5(config)

    strains, targets = split_dataset(gasf_data, labels, config)
    
    training_data, validation_data, testing_data = create_dataloaders(strains, targets, config)

    # Model Training
    train_model(config, device, training_data, validation_data)

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
    main()
