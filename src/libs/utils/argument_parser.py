# src/libs/utils/argument_parser.py

import toml
import yaml

# def parse_arguments(file_path):
#     """Parse arguments from a TOML file."""
#     with open(file_path, 'r') as f:
#         config = toml.load(f)
#     return config

def parse_arguments(file_path):
    """Parse arguments from a TOML file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config