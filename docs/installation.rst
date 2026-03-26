Installation
============

Recommended setup
-----------------

The repository supports local Python use, Docker-based use, and DEEPaaS service execution. For most users, Docker is the safest option because the ML stack is heavy and the project has historically been tested around Python 3.9.

Clone the repository first:

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification

Local install
-------------

Use a dedicated virtual environment. The project metadata and current configuration live in ``pyproject.toml`` and ``etc/config.yaml``.

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

Notes:

* the package entry point is ``planktonclas``
* the DEEPaaS model entry point is ``planktonclas.api``
* training and inference depend on TensorFlow and the packages listed in ``requirements.txt``

Docker install
--------------

If you already use the published image, mount the repository into the container so that notebooks, data, and produced models stay visible on your machine.

.. code-block:: bash

   docker run -ti -p 8888:8888 -p 5000:5000 ^
     -v "${PWD}:/srv/phyto-plankton-classification" ^
     ai4oshub/phyto-plankton-classification:latest /bin/bash

Inside the container, the repository is expected at:

.. code-block:: text

   /srv/phyto-plankton-classification

Project layout
--------------

The main directories are:

* ``planktonclas/``: package source code
* ``etc/config.yaml``: runtime and training configuration
* ``notebooks/``: end-user notebooks
* ``data/images/``: input images
* ``data/dataset_files/``: train/validation/test split files and class metadata
* ``models/``: trained model outputs

Dataset files
-------------

The training pipeline expects these split and metadata files under ``data/dataset_files/``:

* required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``

The split files map image names to numeric labels starting at ``0``. ``classes.txt`` maps those labels back to class names.
