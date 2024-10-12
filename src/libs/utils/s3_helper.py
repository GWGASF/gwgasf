# src/libs/utils/s3_helper.py

import configparser
import os
import s3fs

def load_s3_credentials():
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    host_base = os.getenv('AWS_S3_ENDPOINT')

    if not access_key or not secret_key or not host_base:
        raise ValueError("Missing S3 credentials from environment variables")

    return access_key, secret_key, host_base

# def load_s3_credentials():
#     """Load credentials from the .s3cfg file."""
#     config = configparser.ConfigParser()
#     config.read(os.path.expanduser('~/.s3cfg'))  # Read from ~/.s3cfg
#     access_key = config.get('default', 'access_key')
#     secret_key = config.get('default', 'secret_key')
#     host_base = config.get('default', 'host_base')
#     return access_key, secret_key, host_base

def create_s3_filesystem():
    """Create an S3 filesystem object using credentials from .s3cfg."""
    access_key, secret_key, host_base = load_s3_credentials()
    return s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={'endpoint_url': f'{host_base}'}
    )