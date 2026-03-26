Reference
=========

Package entry points
--------------------

``planktonclas.api``
   DEEPaaS-facing API layer. Handles metadata, schema generation, training dispatch, model loading, file validation, and prediction formatting.

``planktonclas.train_runfile``
   Direct training runner. Creates output directories, builds generators, trains the TensorFlow model, stores metrics, saves checkpoints, and optionally evaluates a test split.

``planktonclas.config``
   Loads the packaged default config template or a user-provided project ``config.yaml``, validates values, and exposes the flattened configuration dictionary used across the package.

``planktonclas.paths``
   Central path resolver for images, models, checkpoints, logs, stats, and predictions.

``planktonclas.report_utils``
   Generates evaluation plots and summary files in the timestamped ``results/`` directory.

``planktonclas.test_utils``
   Inference helpers for crop-based prediction and top-k accuracy computation.

``planktonclas.visualization``
   Visualization and explainability utilities, including saliency-related helpers used by the notebooks.

Configuration map
-----------------

The runtime configuration is grouped in the active ``config.yaml`` under:

* ``general``
* ``model``
* ``dataset``
* ``training``
* ``monitor``
* ``augmentation``
* ``testing``

Important conventions
---------------------

* images are read from ``general.images_directory``
* dataset split files are expected under ``data/dataset_files/``
* outputs are organized by training timestamp under ``models/<timestamp>/``
* inference defaults to the latest available trained timestamp

Source files
------------

For the implementation details, start with these files in the repository:

* ``planktonclas/api.py``
* ``planktonclas/train_runfile.py``
* ``planktonclas/config.py``
* ``planktonclas/paths.py``
* ``planktonclas/test_utils.py``
