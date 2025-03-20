#!/bin/bash

# Path to the TOML file
# TOML_FILE="src/arguments.toml"
TOML_FILE="src/arguments.yaml"


# Update paths section
if [ -n "$DATA_PATH_INJ" ]; then
    sed -i "s|^data_path_inj = .*|data_path_inj = \"$DATA_PATH_INJ\"|" "$TOML_FILE"
fi

if [ -n "$DATA_PATH_NOISE" ]; then
    sed -i "s|^data_path_noise = .*|data_path_noise = \"$DATA_PATH_NOISE\"|" "$TOML_FILE"
fi

if [ -n "$DATA_PATH_GASF" ]; then
    sed -i "s|^data_path_gasf = .*|data_path_gasf = \"$DATA_PATH_GASF\"|" "$TOML_FILE"
fi

if [ -n "$MODELS_PATH" ]; then
    sed -i "s|^models_path = .*|models_path = \"$MODELS_PATH\"|" "$TOML_FILE"
fi

if [ -n "$RESULTS_PATH" ]; then
    sed -i "s|^results_path = .*|results_path = \"$RESULTS_PATH\"|" "$TOML_FILE"
fi

# Update options section
if [ -n "$CREATE_NEW_GASF" ]; then
    sed -i "s/^create_new_gasf = .*/create_new_gasf = $CREATE_NEW_GASF/" "$TOML_FILE"
fi

if [ -n "$APPLY_SNR_FILTER" ]; then
    sed -i "s/^apply_snr_filter = .*/apply_snr_filter = $APPLY_SNR_FILTER/" "$TOML_FILE"
fi

if [ -n "$SNR_THRESHOLD" ]; then
    sed -i "s/^snr_threshold = .*/snr_threshold = $SNR_THRESHOLD/" "$TOML_FILE"
fi

if [ -n "$SHUFFLE_DATA" ]; then
    sed -i "s/^shuffle_data = .*/shuffle_data = $SHUFFLE_DATA/" "$TOML_FILE"
fi

if [ -n "$SELECT_SAMPLES" ]; then
    sed -i "s/^select_samples = .*/select_samples = $SELECT_SAMPLES/" "$TOML_FILE"
fi

if [ -n "$TRAIN_MODEL" ]; then
    sed -i "s/^train_model = .*/train_model = $TRAIN_MODEL/" "$TOML_FILE"
fi

if [ -n "$NUM_BBH" ]; then
    sed -i "s/^num_bbh = .*/num_bbh = $NUM_BBH/" "$TOML_FILE"
fi

if [ -n "$NUM_BG" ]; then
    sed -i "s/^num_bg = .*/num_bg = $NUM_BG/" "$TOML_FILE"
fi

if [ -n "$NUM_GLITCH" ]; then
    sed -i "s/^num_glitch = .*/num_glitch = $NUM_GLITCH/" "$TOML_FILE"
fi

# Update hyperparameters section
if [ -n "$LEARNING_RATE" ]; then
    sed -i "s/^learning_rate = .*/learning_rate = $LEARNING_RATE/" "$TOML_FILE"
fi

if [ -n "$EPOCHS" ]; then
    sed -i "s/^epochs = .*/epochs = $EPOCHS/" "$TOML_FILE"
fi

if [ -n "$L2_REG" ]; then
    sed -i "s/^L2_reg = .*/L2_reg = $L2_REG/" "$TOML_FILE"
fi

if [ -n "$BATCH_SIZE" ]; then
    sed -i "s/^batch_size = .*/batch_size = $BATCH_SIZE/" "$TOML_FILE"
fi

if [ -n "$SEED" ]; then
    sed -i "s/^seed = .*/seed = $SEED/" "$TOML_FILE"
fi

# Update ratios section
if [ -n "$TRAIN_RATIO" ]; then
    sed -i "s/^train = .*/train = $TRAIN_RATIO/" "$TOML_FILE"
fi

if [ -n "$TEST_RATIO" ]; then
    sed -i "s/^test = .*/test = $TEST_RATIO/" "$TOML_FILE"
fi

if [ -n "$VALIDATION_RATIO" ]; then
    sed -i "s/^validation = .*/validation = $VALIDATION_RATIO/" "$TOML_FILE"
fi

# Print the TOML file
cat "$TOML_FILE"

# Run the Python script
python3 src/main.py
