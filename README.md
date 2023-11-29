DEEP Open Catalogue: Image classification
=========================================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/phyto-plankton-classification/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/phyto-plankton-classification/job/master/)


**Author:** [Ignacio Heredia & Wout Decrop](https://github.com/IgnacioHeredia) (CSIC)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has
received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool to train and evaluate an phytoplankton classifier on a custom dataset using deep neural networks.

You can find more information about it in the [DEEP Marketplace](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-phyto-plankton-classification.html).

**Table of contents**
1. [Installing this module](#installing-this-module)
    1. [Local installation](#local-installation)
    2. [Docker installation](#docker-installation)
2. [Train a new image classifier](#train-an-image-classifier)
    1. [Data preprocessing](#1-data-preprocessing)
        1. [Prepare the images](#11-prepare-the-images)
        2. [Prepare the data splits](#12-prepare-the-data-splits)
    2. [Train the classifier](#train-an-image-classifier)
3. [Test an image classifier](#test-an-image-classifier)
4. [More info](#more-info)
5. [Acknowledgements](#acknowledgments)

## Installing this module

### Local installation

> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6.9. Further package requirements are described in the
> `requirements.txt` file.
> - It is a requirement to have [Tensorflow>=2.3.3 installed](https://www.tensorflow.org/install/pip) (either in gpu 
> or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166). 
> - Run `python -c 'import cv2'` to check that you installed correctly the `opencv-python` package (sometimes
> [dependencies are missed](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo) in `pip` installations).

To start using this framework clone the repo and download the [default weights](https://api.cloud.ifca.es:8080/swift/v1/imagenet-tf/default_imagenet.tar.xz):

```bash
# First line installs OpenCV requirement
apt-get update && apt-get install -y libgl1
git clone https://github.com/deephdc/phyto-plankton-classification
cd phyto-plankton-classification
pip install -e .
curl -o ./models/default_imagenet.tar.xz https://api.cloud.ifca.es:8080/swift/v1/imagenet-tf/default_imagenet.tar.xz #create share link from nextcloud
cd models && tar -xf default_imagenet.tar.xz && rm default_imagenet.tar.xz
```
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui and look for the methods belonging to the `planktonclas` module.

### Install through Docker

#### 1.1 Install docker
Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/). 

#### 1.2 Run docker
Ensure Docker is installed and running on your system before executing the DEEP-OC Phytoplankton Classification module using Docker containers.
So open docker, if correct, you should see a small ship (docker desktop) symbol running on the bottom right of your windows screen

#### 1.3 Clone the directory
The directory is cloned so that the remote and the local directory are the same. This makes it easier to copy files inside the remote directory
```bash
git clone https://github.com/deephdc/phyto-plankton-classification
```

#### 1.4 Run the docker container inside the local folder

After docker is installed and running, you can run the ready-to-use [Docker container](https://github.com/deephdc/DEEP-OC-phyto-plankton-classification) to
run this module. To run it:

##### option 1) Run container and activate acess to nextcloud server through rclone.

If you rclone credentials (see [Tutorial](https://docs.ai4eosc.eu/en/latest/user/howto/rclone.html#configuring-rclone)) from the [NEXTCLOUD](https://data-deep.a.incd.pt/) server, you can also create a direct link to these credentials through one line of code. 

First, install it directly on your machine:
```bash
$ curl -O https://downloads.rclone.org/v1.62.2/rclone-v1.62.2-linux-amd64.deb
$ apt install ./rclone-v1.62.2-linux-amd64.deb
$ rm rclone-current-linux-amd64.deb
```

Secondly, run 'rclone config'
```bash
$ rclone config
choose "n"  for "New remote"
choose name for DEEP-Nextcloud --> rshare
choose "Type of Storage" --> Webdav
provide DEEP-Nextcloud URL for webdav access --> https://data-deep.a.incd.pt/remote.php/webdav/
choose Vendor --> Nextcloud
specify "user" --> (see `<user>` in "Configuring rclone" above).
password --> y (Yes type in my own password)
specify "password" --> (see `<password>` in "Configuring rclone" above).
bearer token --> ""
Edit advanced config? --> n (No)
Remote config --> y (Yes this is OK)
Current remotes --> q (Quit config)
```

After installing rclone and running 'rclone config'. The rclone.conf page should look like this: 
```bash
[rshare]
type = webdav
url = https://data-deep.a.incd.pt/remote.php/webdav/
vendor = nextcloud
user = ***some-username***
pass = ***some-userpassword**  --> this is equivalent to `rclone obscure <password>`
```
Copy the location to the rclone.config location (e.g., C:/Users/wout.decrop/project_environment/module_imagine/rclone.conf) and apply the line of code

```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "LOCATION/rclone.conf:/root/.config/rclone/rclone.conf" -v "$(pwd):/srv/phyto-plankton-classification" deephdc/uc-woutdecrop-deep-oc-phyto-plankton-classification:latest /bin/bash
```
For example
```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "C:/Users/wout.decrop/project_environment/module_imagine/rclone.conf:/root/.config/rclone/rclone.conf" -v "$(pwd):/srv/phyto-plankton-classification" deephdc/uc-woutdecrop-deep-oc-phyto-plankton-classification:latest /bin/bash
```

##### option 2) Run container and only have local access
All files can be locally saved but rclone needs to be configured after activation to acces nextcloud server, follow [Tutorial](https://docs.ai4eosc.eu/en/latest/user/howto/rclone.html#configuring-rclone)
```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" deephdc/uc-woutdecrop-deep-oc-phyto-plankton-classification:latest /bin/bash
```
Now open http://0.0.0.0:5000/ui and look for the methods belonging to the `planktonclas` module.


## Train an image classifier

You can train your own audio classifier with your custom dataset. For that you have to:

### 1. Data preprocessing

The first step to train you image classifier if to have the data correctly set up. 

#### 1.1 Prepare the images

Put your images in the`./data/images` folder. If you have your data somewhere else you can use that location by setting
 the `image_dir` parameter in the training args. 
Please use a standard image format (like `.png` or `.jpg`). 

#### 1.2 Prepare the data splits

First you need add to the `./data/dataset_files` directory the following files:

| *Mandatory files* | *Optional files*  | 
|:-----------------------:|:---------------------:|
|  `classes.txt`, `train.txt` |  `val.txt`, `test.txt`, `info.txt`|

The `train.txt`, `val.txt` and `test.txt` files associate an image name (or relative path) to a label number (that has
to *start at zero*).
The `classes.txt` file translates those label numbers to label names.
Finally the `info.txt` allows you to provide information (like number of images in the database) about each class. 

You can find examples of these files at  `./data/demo-dataset_files`.

### 2. Train the classifier

> **Tip**: Training is usually depend on the training args you use. Although the default ones work reasonable well,
> you can explore how to modify them with the [dataset exploration notebook](./notebooks/1.0-Dataset_exploration.ipynb).

Go to http://0.0.0.0:5000/ui and look for the ``TRAIN`` POST method. Click on 'Try it out', change whatever training args
you want and click 'Execute'. The training will be launched and you will be able to follow its status by executing the 
``TRAIN`` GET method which will also give a history of all trainings previously executed.

If the module has some sort of training monitoring configured (like Tensorboard) you will be able to follow it at 
http://0.0.0.0:6006.


## Test an image classifier

Go to http://0.0.0.0:5000/ui and look for the `PREDICT` POST method. Click on 'Try it out', change whatever test args
you want and click 'Execute'. You can **either** supply a:

* a `data` argument with a path pointing to an image.

OR
* a `url` argument with an URL pointing to an image.
  Here is an [example](https://file-examples.com/wp-content/uploads/2017/10/file_example_PNG_500kB.png) of such an url
  that you can use for testing purposes.


## More info

You can have more info on how to interact directly with the module (not through the DEEPaaS API) by examining the 
``./notebooks`` folder:

* [dataset exploration notebook](./notebooks/1.0-Dataset_exploration.ipynb):
  Visualize relevant statistics that will help you to modify the training args.

* [computing predictions notebook](./notebooks/3.0-Computing_predictions.ipynb):
  Test the classifier on a number of tasks: predict a single local image (or url), predict multiple images (or urls),
  merge the predictions of a multi-image single observation, etc.

<img src="./reports/figures/predict.png" alt="predict" width="400">

* [predictions statistics notebook](./notebooks/3.1-Prediction_statistics.ipynb):
  Make and store the predictions of the `test.txt` file (if you provided one). Once you have done that you can visualize
  the statistics of the predictions like popular metrics (accuracy, recall, precision, f1-score), the confusion matrix, etc.



Finally you can [launch a simple webpage](./planktonclas/webpage/README.md) to use the trained classifier to predict images (both local and urls) on your favorite browser.


## Acknowledgements

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692. 
