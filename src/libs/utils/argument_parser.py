# src/libs/utils/argument_parser.py

import toml
import yaml

# def parse_arguments(file_path):
#     """Parse arguments from a TOML file."""
#     with open(file_path, 'r') as f:
#         config = toml.load(f)
#     return config

def parse_arguments(file_path):
    """Parse arguments from a YAML file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# def load_config(file_path):
#     with open(file_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # Override with env vars if set
#     def override(path, env_var, cast_fn=str):
#         keys = path.split('.')
#         if env_var in os.environ:
#             val = cast_fn(os.environ[env_var])
#             d = config
#             for k in keys[:-1]:
#                 d = d.setdefault(k, {})
#             d[keys[-1]] = val

#     override("paths.data_path", "DATA_PATH_INJ")
#     override("paths.data_path_gasf", "DATA_PATH_GASF")
#     override("paths.models_path", "MODELS_PATH")
#     override("paths.results_path", "RESULTS_PATH")

#     override("options.create_new_gasf", "CREATE_NEW_GASF", lambda x: x.lower() == 'true')
#     override("options.apply_snr_filter", "APPLY_SNR_FILTER", lambda x: x.lower() == 'true')
#     override("options.snr_threshold", "SNR_THRESHOLD", float)
#     override("options.shuffle_data", "SHUFFLE_DATA", lambda x: x.lower() == 'true')
#     override("options.select_samples", "SELECT_SAMPLES", lambda x: x.lower() == 'true')
#     override("options.train_model", "TRAIN_MODEL", lambda x: x.lower() == 'true')

#     override("options.num_bbh", "NUM_BBH", int)
#     override("options.num_bg", "NUM_BG", int)
#     override("options.num_glitch", "NUM_GLITCH", int)

#     override("hyperparameters.learning_rate", "LEARNING_RATE", float)
#     override("hyperparameters.epochs", "EPOCHS", int)
#     override("hyperparameters.L2_reg", "L2_REG", float)
#     override("hyperparameters.batch_size", "BATCH_SIZE", int)
#     override("hyperparameters.seed", "SEED", int)

#     override("ratios.train", "TRAIN_RATIO", float)
#     override("ratios.test", "TEST_RATIO", float)
#     override("ratios.validation", "VALIDATION_RATIO", float)

#     override("s3_credentials.access_key", "AWS_ACCESS_KEY_ID")
#     override("s3_credentials.secret_key", "AWS_SECRET_ACCESS_KEY")
#     override("s3_credentials.host_base", "AWS_S3_ENDPOINT")

#     return config