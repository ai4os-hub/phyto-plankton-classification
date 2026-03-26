Quickstart
==========

Minimal training workflow
-------------------------

1. Put your images under ``data/images/`` or set ``general.images_directory`` in ``etc/config.yaml``.
2. Add dataset split files under ``data/dataset_files/``.
3. Adjust training parameters in ``etc/config.yaml``.
4. Run training either from Python, from the DEEPaaS API, or from the notebooks.

Minimal prediction workflow
---------------------------

1. Train at least one model, or place an existing trained timestamp under ``models/``.
2. Start the DEEPaaS service or load the package in Python.
3. Provide either a single image or a ZIP archive of images.

Start the API
-------------

.. code-block:: bash

   deepaas-run --listen-ip 0.0.0.0

Then open one of:

* ``http://0.0.0.0:5000/ui``
* ``http://0.0.0.0:5000/api#/``

Run training directly
---------------------

.. code-block:: bash

   cd planktonclas
   python train_runfile.py

Outputs are written into a timestamped directory under ``models/``. Each run stores checkpoints, logs, copied dataset split files, configuration, and prediction artifacts.

Start Jupyter
-------------

If you are inside the prepared container environment:

.. code-block:: bash

   deep-start -j

Open the provided URL and use the notebooks in ``notebooks/``.
