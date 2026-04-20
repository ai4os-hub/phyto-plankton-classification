Installation
============

This page explains the setup choices for the full repository.

If you want the package-only installation page, use the companion repository:

* ``planktonclas``: https://github.com/woutdecrop/planktonclas

Setup choices
-------------

This repository supports four common setup paths:

* install ``planktonclas`` as a package for normal local usage
* use Docker for a containerized runtime
* install the repository locally for development
* use AI4OS / OSCAR for hosted deployment

Option A: Package install
-------------------------

.. code-block:: bash

   pip install planktonclas

For local notebook use:

.. code-block:: bash

   pip install "planktonclas[notebooks]"

This is the best option if you want the local CLI, API, or notebook workflow without cloning the whole repository.

For the package-focused explanation of this path, use:

* https://github.com/woutdecrop/planktonclas

Option B: Docker
----------------

This is the simplest repository-based workflow if you want the full project files but do not want to install all Python dependencies on your machine.

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification
   docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" ai4oshub/phyto-plankton-classification:latest /bin/bash

Inside the container, you can use the same ``planktonclas`` commands as in the local workflow.

The container image also ships with the published pretrained model under ``models/``.

Option C: Repository install for development
--------------------------------------------

Choose this only if you want to work on the repository itself.

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification
   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

After a repository install, you can also start DEEPaaS directly:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Option D: AI4OS / OSCAR
-----------------------

Use this path when you want hosted deployment or a managed remote runtime.

Useful links:

* `AI4OS / iMagine Marketplace <https://dashboard.cloud.imagine-ai.eu/marketplace/>`_
* `AI4OS docs <https://docs.ai4os.eu/en/latest/>`_
* `OSCAR manual deployment guide <https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar-manual.html>`_
* `OSCAR scripted deployment guide <https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar.html>`_
* `Marketplace notes <https://github.com/ai4os-hub/phyto-plankton-classification/blob/main/references/README_marketplace.md>`_

Project structure
-----------------

After ``planktonclas init``, a project looks like this:

.. code-block:: text

   my_project/
     config.yaml
     data/
       images/
       dataset_files/
     models/
     notebooks/

Next step
---------

After installation or setup, continue with :doc:`quickstart` to choose your path.
