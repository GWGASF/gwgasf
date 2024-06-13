# src/libs/utils/argument_parser.py

import toml

def parse_arguments(file_path):
    """Parse arguments from a TOML file."""
    with open(file_path, 'r') as f:
        config = toml.load(f)
    return config
