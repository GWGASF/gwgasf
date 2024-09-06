FROM ubuntu:22.04

RUN apt -y update \
    && apt -y upgrade \
    && apt -y install python3 \
    && apt -y install python3-pip 

RUN pip install pyts==0.13.0 \
    && pip install numpy==1.24.1 \
    && pip install matplotlib==3.7.1 \
    && pip install h5py==3.7.0 \
    && pip install tqdm==4.65.0 \
    && pip install toml==0.10.2 \
    && pip install s3cmd==2.2.0

RUN pip install torch==2.1.2 \
    && pip install torchsummary==1.5.1 