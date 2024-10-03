# syntax=docker/dockerfile:1.2

FROM ubuntu:22.04

# Install dependencies and Python 3.9
RUN apt -y update \
    && apt -y upgrade \
    && apt -y install python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install pyts==0.13.0 \
    && pip install numpy==1.24.1 \
    && pip install matplotlib==3.7.1 \
    && pip install h5py==3.7.0 \
    && pip install tqdm==4.65.0 \
    && pip install toml==0.10.2 \
    && pip install s3cmd==2.2.0 \
    && pip install s3fs \
    && pip install torch==2.1.2 \
    && pip install torchsummary==1.5.1

# Set the working directory
WORKDIR /app/gwgasf

# Add the GitLab SSH host key to known_hosts to bypass host key verification prompt
RUN mkdir -p ~/.ssh \
    && ssh-keyscan -p 30622 gitlab-ssh.nrp-nautilus.io >> ~/.ssh/known_hosts

# Clone the GitLab repository into the current working directory using SSH
# This allows users to clone the repo using their own SSH key
RUN --mount=type=ssh git clone ssh://git@gitlab-ssh.nrp-nautilus.io:30622/gwgasf/gwgasf.git .

# Cleanup to reduce image size (remove git after cloning)
RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Command to run the script when the container starts
CMD ["python3", "src/main.py"]
