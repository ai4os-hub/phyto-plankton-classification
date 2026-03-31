Quickstart
==========

Choose your path
----------------

Most users should choose one of these workflows:

1. local training
2. local API
3. notebooks

These are alternative entry points. You do not need to use all of them.

Minimal project setup
---------------------

.. code-block:: bash

   planktonclas init my_project
   planktonclas validate-config --config ./my_project/config.yaml

For a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

To download the published pretrained model into the project:

.. code-block:: bash

   planktonclas pretrained my_project

Local training
--------------

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

Outputs are written into a timestamped directory under ``my_project/models/``.

For a quick smoke test on the demo project:

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml --quick

When test evaluation is enabled, training also writes a compact metrics JSON next to the saved prediction JSON in ``my_project/models/<timestamp>/predictions/``. That file includes top-k accuracy plus precision, recall, and F1 summaries.

Generate a report
-----------------

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

This writes evaluation images and metric files under ``my_project/models/<timestamp>/results/``.

If you leave out ``--timestamp``, the CLI suggests the newest run automatically, shows the available timestamps, and lets you choose another one by number.
If you leave out ``--mode``, the CLI suggests ``quick`` automatically. Quick mode creates the core figures only, while full mode also generates the threshold-based plots in the ``results/`` subfolders.

Local API
---------

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

You can also start DEEPaaS directly after a repository install:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Notebook workflow
-----------------

For local notebook use:

.. code-block:: bash

   pip install "planktonclas[notebooks]"

.. code-block:: bash

   planktonclas notebooks my_project

This copies the packaged notebooks into ``my_project/notebooks/``.

Useful commands
---------------

.. code-block:: bash

   planktonclas list-models --config ./my_project/config.yaml
   planktonclas pretrained my_project

Dataset notes
-------------

The only mandatory input is the image directory.

If ``data/dataset_files/`` is empty, training can create split files automatically.

If you provide your own metadata files, the expected files are:

* custom-split required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``
