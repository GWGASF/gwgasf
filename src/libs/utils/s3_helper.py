# src/libs/utils/s3_helper.py
import s3fs

def create_s3_filesystem(config):
    """Create an S3 filesystem object using credentials from the config."""
    s3_config = config.get('s3', {})
    access_key = s3_config.get('access_key')
    secret_key = s3_config.get('secret_key')
    host_base  = s3_config.get('host_base')
    region     = s3_config.get('region', 'us-east-1')  # default if not provided

    if not access_key or not secret_key or not host_base:
        raise ValueError("Missing S3 credentials from arguments.yaml")

    return s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={
            'endpoint_url': host_base,
            'region_name': region,
        },
        config_kwargs={
            'signature_version': 's3v4',
            's3': {
                'addressing_style': 'path',
                'payload_signing_enabled': True,  # <-- crucial for multipart SHA256 match
            },
        },
        # Optional, but can help on some gateways:
        # use_listings_cache=False
    )
