# Gravitational Wave Multiclass Classifier with GASF


This project harnesses the power of Gramian Angular Summation Fields (GASF) in combination with advanced machine learning techniques for the purpose of classifying gravitational wave signals. By transforming time series data into a format that's suitable for image-based classification, the project leverages convolutional neural networks to perform multiclass classification of these signals. The project aims to provide a robust, scalable solution for the automatic categorization of complex gravitational waveforms, thereby aiding scientific communities in their quest to unravel the mysteries of the cosmos.

## Getting Started

### Prerequisites
Copy paste the following command to create the environment to run GWGASF Multiclass Classifier in src/main.py.
```
conda create -n GASF39 python==3.9.18 -y
conda activate GASF39
pip install poetry==1.7.1

cd ../src
poetry install
```


## Retrieving/Loading Data

Since data is loaded from large h5py files, the input data for the model is stored on the [GASF shared Google Drive](https://drive.google.com/drive/folders/12jjEFBU81Y8PB7VUrPHPcZViBhm86obJ?usp=drive_link).
