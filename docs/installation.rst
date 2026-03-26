Installation
============

Recommended setup
-----------------

The repository supports local Python use, Docker-based use, and DEEPaaS service execution. For local use, prefer a dedicated virtual environment.

Clone the repository first:

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification

Local install
-------------

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

Notes:

* the package CLI entry point is ``planktonclas``
* the DEEPaaS model entry point is ``planktonclas.api``
* training and inference require TensorFlow and the packages listed in ``requirements.txt``
* new user projects should use a project-local ``config.yaml`` created by ``planktonclas init``
* report generation uses the saved training statistics and test predictions from a completed run

Initialize a project
--------------------

.. code-block:: bash

   planktonclas init my_project

Or create a demo project:

.. code-block:: bash

   planktonclas init my_project --demo

Docker install
--------------

If you already use the published image, mount the repository into the container so that notebooks, data, and produced models stay visible on your machine.

.. code-block:: bash

   docker run -ti -p 8888:8888 -p 5000:5000 ^
     -v "${PWD}:/srv/phyto-plankton-classification" ^
     ai4oshub/phyto-plankton-classification:latest /bin/bash

Inside the container, use the same CLI workflow with ``planktonclas init``, ``planktonclas train``, and ``planktonclas api``.

Project layout
--------------

The main directories in a generated project are:

* ``config.yaml``: runtime and training configuration
* ``data/images/``: input images
* ``data/dataset_files/``: train/validation/test split files and class metadata
* ``models/``: trained model outputs

Dataset files
-------------

The training pipeline expects these split and metadata files under ``data/dataset_files/``:

* required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``

The split files map image names to numeric labels starting at ``0``. ``classes.txt`` maps those labels back to class names.
