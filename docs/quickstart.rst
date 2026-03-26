Quickstart
==========

Minimal CLI workflow
--------------------

1. Install the package in a virtual environment.
2. Create a project directory with ``planktonclas init``.
3. Edit the generated ``config.yaml``.
4. Add your images under ``data/images/`` and split files under ``data/dataset_files/``.
5. Run training or start the API.
6. Generate a report with plots in the run's ``results/`` folder.

Create a new project
--------------------

.. code-block:: bash

   planktonclas init my_project

For a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

Validate the configuration
--------------------------

.. code-block:: bash

   planktonclas validate-config --config ./my_project/config.yaml

Run training
------------

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

Outputs are written into a timestamped directory under ``my_project/models/``.

Generate a report
-----------------

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

This writes evaluation images and metric files under ``my_project/models/<timestamp>/results/``.

Start the API
-------------

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open one of:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api``

You can also start DEEPaaS directly after cloning and installing the repository:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Then use:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

List trained models
-------------------

.. code-block:: bash

   planktonclas list-models --config ./my_project/config.yaml

Open notebooks
--------------

.. code-block:: bash

   planktonclas notebooks

Dataset files
-------------

The only mandatory input is the image directory.

If ``data/dataset_files/`` is empty, training can create the split files automatically from the image-folder structure.

If you provide your own dataset metadata files, the expected files under ``data/dataset_files/`` are:

* custom-split required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``
