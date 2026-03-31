Installation
============

Choose your installation mode
-----------------------------

There are four common ways to use ``planktonclas``:

* install as a package for normal CLI usage
* use Docker for a containerized runtime
* install from the repository for local development
* use AI4OS / OSCAR for hosted deployment

Package install
---------------

.. code-block:: bash

   pip install planktonclas

This is the best option for users who just want the CLI workflow without cloning the repository.

Then initialize a project:

.. code-block:: bash

   planktonclas init my_project

Or create a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

Optional helpers:

.. code-block:: bash

   planktonclas pretrained my_project
   planktonclas notebooks my_project

Docker install
--------------

.. code-block:: bash

   docker run -ti -p 8888:8888 -p 5000:5000 ^
     -v "${PWD}:/srv/phyto-plankton-classification" ^
     ai4oshub/phyto-plankton-classification:latest /bin/bash

Inside the container, use the same CLI workflow with ``planktonclas init``, ``planktonclas train``, ``planktonclas pretrained``, and ``planktonclas api``.

The container image also ships with the published pretrained model under ``models/``.

If the image or deployment provides the AI4OS helper scripts, you may also have:

.. code-block:: bash

   deep-start -j
   deep-start --deepaas

Important:

* a normal local install does not provide ``deep-start``
* for local installs, use ``planktonclas ...`` or ``deepaas-run``
* ``deep-start`` is typically available only in AI4OS/container/deployment environments that ship those helpers

Repository install
------------------

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification
   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

This is the best option for development work on the repository itself.

Direct API startup
------------------

After a repository install, you can also start DEEPaaS directly:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Project layout
--------------

After ``planktonclas init``, a project looks like this:

.. code-block:: text

   my_project/
     config.yaml
     data/
       images/
       dataset_files/
     models/

If you also copy the packaged notebooks, the project gains:

.. code-block:: text

   my_project/
     notebooks/

Required input
--------------

The only mandatory input is the image directory.

If ``data/dataset_files/`` is empty, training can generate split files automatically from the image-folder structure.

If you provide your own dataset metadata files, the expected files are:

* custom-split required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``
