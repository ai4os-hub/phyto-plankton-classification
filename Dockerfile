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

ARG tag=1.14.0-py3

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM tensorflow/tensorflow:${tag}

LABEL maintainer='Ignacio Heredia (CSIC), Wout Decrop (VLIZ)'
LABEL version='0.1.0'
# Add container's metadata to appear along the models metadata
ENV CONTAINER_MAINTAINER "Wout Decrop <wout.decrop@vliz.be>"
ENV CONTAINER_VERSION "0.1"
ENV CONTAINER_DESCRIPTION "AI4OS/DEEP as a Service Container: Phyto-Plankton Classification"
# Identify the species level of Plankton for 95 classes. Working on OSCAR

# What user branch to clone [!]
ARG branch=main
ARG tag   # need to correctly parse $tag variable

ARG DEBIAN_FRONTEND=noninteractive

# 2024: need to re-add GPG keys for Nvidia repos but only in the case of GPU images
# Note for GPU build: see https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN if [[ "$tag" =~ "-gpu" ]]; then \
    apt-key del 7fa2af80 ; \
    curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - ; \
    curl https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - ; fi


# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs) (it is already installed in tensorflow images)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        curl \
        psmisc \
        python3-setuptools \
        python3-pip \
        python3-wheel \
        libgl1 \
        libsm6 \
        libxrender1 \
        libfontconfig1 \
    && apt-get clean \
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
# * allows to run shorter command "deep-start"
# * deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Useful tool to debug extensions loading
RUN pip install --no-cache-dir entry_point_inspector && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash

# Install user app
RUN git clone -b $branch --depth 1 https://github.com/ai4os-hub/phyto-plankton-classification && \
    cd  phyto-plankton-classification && \
    pip3 install --no-cache-dir -e . && \
    cd ..

# https://share.services.ai4os.eu/index.php/s/rJQPQtBReqHAPf3/download/phytoplankton_vliz.tar.gz
# https://share.services.ai4os.eu/index.php/s/dFg9cma5FwG6PZD/download/phytoplankton_vliz.tar.xz
# ENV SWIFT_CONTAINER https://share.services.ai4os.eu/index.php/s/jfS26BjQzHx3osc/download/
ENV SWIFT_CONTAINER=https://share.services.ai4os.eu/index.php/s/rJQPQtBReqHAPf3/download
ENV MODEL_TAR=phytoplankton_vliz.tar.gz
# mkdir -p ./phyto-plankton-classification/models \
# Download and extract the file
RUN curl -L ${SWIFT_CONTAINER} -o ./phyto-plankton-classification/models/${MODEL_TAR} 
RUN cd ./phyto-plankton-classification/models && \
    tar -xzf ${MODEL_TAR} && \
    rm ${MODEL_TAR}

# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
