# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# Be Aware! For the Jenkins CI/CD pipeline, 
# input args are defined inside the JenkinsConstants.groovy, not here!

ARG tag=2.19.0-gpu

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM tensorflow/tensorflow:${tag}

LABEL maintainer='Ignacio Heredia (CSIC), Wout Decrop (VLIZ)'
LABEL version='0.1.0'
# Add container's metadata to appear along the models metadata
ENV CONTAINER_MAINTAINER "Wout Decrop <wout.decrop@vliz.be>"

# Identify the species level of Plankton for 95 classes. Working on OSCAR


ARG branch=main

ARG tag   # need to correctly parse $tag variable

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs) (it is already installed in tensorflow images)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        libgl1\
        psmisc \
        curl \
        libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Update python packages
# [!] Remember: DEEP API V2 only works with python>=3.6
RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip "setuptools<60.0.0" wheel

# TODO: remove setuptools version requirement when [1] is fixed
# [1]: https://github.com/pypa/setuptools/issues/3301

# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Install rclone (needed if syncing with NextCloud for training; otherwise remove)
RUN curl -O https://downloads.rclone.org/rclone-current-linux-amd64.deb && \
    dpkg -i rclone-current-linux-amd64.deb && \
    apt install -f && \
    mkdir /srv/.rclone/ && \
    touch /srv/.rclone/rclone.conf && \
    rm rclone-current-linux-amd64.deb && \
    rm -rf /var/lib/apt/lists/*

ENV RCLONE_CONFIG=/srv/.rclone/rclone.conf

# Disable FLAAT authentication by default
ENV DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER yes

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash

# Install user app
RUN git clone -b $branch --depth 1 https://github.com/ai4os-hub/phyto-plankton-classification && \
    cd  phyto-plankton-classification && \
    pip install --ignore-installed blinker -e .  && \
    cd ..
   # pip uninstall -y numpy && \
  # pip install numpy~=1.24

# Set environment variables
ENV MODEL_TAR=Phytoplankton_EfficientNetV2B0.tar.gz
ENV MODEL_DIR=./phyto-plankton-classification/models
ENV MODEL_URL=https://zenodo.org/records/15269453/files/${MODEL_TAR}?download=1

# Create models directory, download and extract model, then delete the archive
RUN mkdir -p ${MODEL_DIR} && \
    curl -L "${MODEL_URL}" -o ${MODEL_DIR}/${MODEL_TAR} && \
    tar -xzf ${MODEL_DIR}/${MODEL_TAR} -C ${MODEL_DIR} && \
    rm ${MODEL_DIR}/${MODEL_TAR}


# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
