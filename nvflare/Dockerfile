# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# [!] Note: For the Jenkins CI/CD pipeline, input args are defined inside
# the Jenkinsfile, not here!

ARG tag=latest

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM ai4oshub/phyto-plankton-classification:${tag}

LABEL maintainer='Fahimeh Alibabaei, Leonhard Duda'
LABEL version='0.0.1'

RUN  pip install -r phyto-plankton-classification/nvflare/requirements_nvflare.txt
 
