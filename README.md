Phytoplankon classifier: PI10
=========================================

**Authors:** [Wout Decrop](https://github.com/woutdecrop) *(VLIZ)*, Jonas Mortelmans *(VLIZ)*, and [Ignacio Heredia](https://github.com/IgnacioHeredia)  
*(CSIC)*

**Project:** This branch is based on the main github which is part of the [iMagine](https://www.imagine-ai.eu/) project, funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101058625.

---
## Table of Contents

1. [Installing this module](#installing-this-module)  
   1. [Local installation (not recommended)](#local-installation-not-recommended)  
2. [Prepare for training](#1-prepare-for-training)  
   1. [Data preprocessing](#1-data-preprocessing)  
      1. [Prepare the images](#11-prepare-the-images)  
      2. [Prepare the data splits (optional)](#12-prepare-the-data-splits-optional)  
   2. [Train the network](#2-train-the-network)  
      1. [Adapting the YAML file](#21-adapting-the-yaml-file)  
      2. [Running the training](#22-running-the-training)  
3. [Predict an image classifier](#2-predict-pi10)  
   1. [Predicting methods](#predicting-methods)  
      1. [Predict with Jupyter Notebooks (Recommended)](#predict-with-jupyter-notebooks-recommended)  
      2. [Predict with Deepaas](#predict-with-deepaas)  
4. [Acknowledgements](#acknowledgements)

# Installing this module

## Local installation (not recommended)
Although a local installation is possible, we recommend an installation through docker. This is less likely to breake support and has been tested with latest updates. We are working with python 3.6.9 which can be difficult to install. 
> **Requirements**
```bash
git clone https://github.com/lifewatch/phyto-plankton-classification
cd phyto-plankton-classification
pip install -e .
```

# 1. Prepare for training

You can train your own audio classifier with your custom dataset. For that you have to:

## 1. Data preprocessing

The first step to train you image classifier if to have the data correctly set up. 

### 1.1 Prepare the images

The model needs to be able to access the images. So you have to place your images in the [./data/images](/data/images) folder. If you have your data somewhere else you can use that location by setting the `image_dir` parameter in the training args. 
Please use a standard image format (like `.png` or `.tif`). 

You can copy the images to [phyto-plankton-classification/data/images](/data/images) folder on your pc. 
If the images are on nextcloud, you can one of the next steps depending if you have rclone or not. 


### 1.2 Prepare the data splits (optional)

Next, you need add to the [./data/dataset_files](/data/dataset_files) directory the following files:

| *Mandatory files* | *Optional files*  | 
|:-----------------------:|:---------------------:|
|  `classes.txt`, `train.txt` |  `val.txt`, `test.txt`, `info.txt`,`aphia_ids.txt`|

The `train.txt`, `val.txt` and `test.txt` files associate an image name (or relative path) to a label number (that has
to *start at zero*).
The `classes.txt` file translates those label numbers to label names.
The `aphia_ids.txt` file translates those the classes to their corresponding aphia_ids.
Finally the `info.txt` allows you to provide information (like number of images in the database) about each class. 

You can find examples of these files at [./data/demo-dataset_files](/data/demo-dataset_files).

If you don't want to create your own datasplit, this will be done automatically for you with a 80% train, 10% validation, and 10% test split.


## 2. Train the network
Although you can train within docker too, this branch is specifically explained for training locally.

### 2.1: Adapting the yaml file
Clarify the location of the images inside the [yaml file](/etc/config.yaml) file. If not, [./data/images](/data/images) will be taken. 
Any additional parameter can also be changed here such as the type of split for training/validation/testing, batch size, etc

You can change the config file directly as shown below, or you can change it when running the api.

```bash
  images_directory:
    value: "/srv/phyto-plankton-classification/data/images"
    type: "str"
    help: >
          Base directory for images. If the path is relative, it will be appended to the package path.
```
### 2.2: Running the training
After this, you can go to `/srv/phyto-plankton-classification/planktonclas#` and run `train_runfile.py`.

```bash
cd /srv/phyto-plankton-classification/planktonclas` 
python train_runfile.py
```
The new model will be saved under [phyto-plankton-classification/models](/models)

# 2. Predict PI10

Run [VLIZ-Pi-10_processing](/PI10/VLIZ-Pi-10_processing.py)
More info here [readme.md](/PI10/readme.md)

## Acknowledgements

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692. 