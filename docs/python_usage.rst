Python Usage
============

Overview
--------

You can use the package directly from Python without going through the DEEPaaS web UI. The most relevant modules are:

* ``planktonclas.config``: load and validate configuration
* ``planktonclas.paths``: resolve data, model, log, and output directories
* ``planktonclas.train_runfile``: run training
* ``planktonclas.api``: load trained models and run prediction logic
* ``planktonclas.test_utils``: prediction helpers used by inference

Load configuration
------------------

.. code-block:: python

   from planktonclas import config

   conf = config.get_conf_dict()
   print(conf["general"]["images_directory"])
   print(conf["training"]["epochs"])

The default configuration comes from ``etc/config.yaml``.

Run training from Python
------------------------

.. code-block:: python

   from datetime import datetime
   from planktonclas import config
   from planktonclas.train_runfile import train_fn

   conf = config.get_conf_dict()
   conf["general"]["images_directory"] = "./data/images"
   conf["training"]["epochs"] = 5

   timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
   train_fn(TIMESTAMP=timestamp, CONF=conf)

This creates a new timestamped output directory under ``models/``.

Inspect output paths
--------------------

.. code-block:: python

   from planktonclas import paths

   print(paths.get_models_dir())
   print(paths.get_checkpoints_dir())
   print(paths.get_logs_dir())

Be aware that ``paths.timestamp`` controls which timestamped run directory is currently addressed.

Load a trained model for inference
----------------------------------

.. code-block:: python

   from planktonclas.api import load_inference_model

   load_inference_model(timestamp="2026-03-26_120000", ckpt_name="best_model.keras")

If you omit arguments, the API selects the latest available timestamp and a preferred checkpoint.

Run prediction from Python
--------------------------

The direct inference helper used by the API is ``planktonclas.test_utils.predict``. A typical flow is:

.. code-block:: python

   from planktonclas import config
   from planktonclas import api
   from planktonclas import test_utils

   api.load_inference_model()
   conf = config.conf_dict

   labels, probabilities = test_utils.predict(
       model=api.model,
       X=["/absolute/path/to/image.png"],
       conf=conf,
       top_K=5,
       filemode="local",
       merge=False,
       use_multiprocessing=False,
   )

Notes:

* ``X`` should be a string or a list of file paths
* inference uses crop-based prediction and averages crops per image
* ``merge=True`` merges multiple images into one observation-level prediction

Practical caution
-----------------

The package modules assume the project directory structure exists and that configuration, splits, and trained models follow the repository conventions. For scripted use, keep the same folder layout as the repository unless you are deliberately overriding paths in the configuration.
