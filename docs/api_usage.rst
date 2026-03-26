API Usage
=========

Overview
--------

The DEEPaaS entry point is defined in ``pyproject.toml``:

.. code-block:: text

   [project.entry-points."deepaas.v2.model"]
   planktonclas = "planktonclas.api"

The CLI command:

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

sets the project config and starts DEEPaaS for the ``planktonclas`` model.

Start the service
-----------------

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/api``
* ``http://127.0.0.1:5000/ui``

Use ``127.0.0.1`` in the browser. ``0.0.0.0`` is only the bind address.

Direct DEEPaaS startup
----------------------

If you cloned the repository and installed it locally, you can also start the API without the CLI wrapper:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Main API functions
------------------

The main public API functions are:

* ``get_metadata()``: returns package metadata
* ``get_train_args()``: returns the training schema exposed by DEEPaaS
* ``train(**args)``: launches a training run
* ``get_predict_args()``: returns the prediction schema exposed by DEEPaaS
* ``predict(**args)``: runs inference on an uploaded image or ZIP archive

Train endpoint
--------------

The training schema is generated from the active config template and can be overridden by request parameters. In practice, the most important fields are:

* ``images_directory``: folder containing the input images
* ``modelname``: backbone architecture
* ``image_size``: input image size
* ``batch_size``: training batch size
* ``epochs``: number of epochs
* ``use_validation`` and ``use_test``: whether validation and test splits are used
* ``use_best_model``: whether the best validation checkpoint is saved and preferred for later inference

Typical browser workflow:

1. start ``planktonclas api --config ./my_project/config.yaml``
2. open ``/ui`` or ``/api``
3. find the ``TRAIN`` operation
4. change the parameters you need
5. execute the request

What training writes
--------------------

Each training run creates a timestamped folder under ``models/``. The important subdirectories are:

* ``ckpts/``: saved model files
* ``conf/``: saved run configuration
* ``logs/``: training log and CSV epoch metrics
* ``stats/``: serialized training statistics
* ``dataset_files/``: a copy of the dataset split files used for that run
* ``predictions/``: evaluation predictions when test evaluation is enabled

Predict endpoint
----------------

The prediction endpoint accepts one of these inputs:

* ``image``: a single uploaded image or file argument
* ``zip``: a ZIP archive containing one or more images, including nested folders

Typical browser flow:

1. start the API
2. open ``/ui`` or ``/api#/``
3. find the ``PREDICT`` ``POST`` method
4. click ``Try it out``
5. provide either ``image`` or ``zip``
6. click ``Execute``

Prediction response
-------------------

The response payload contains:

* ``filenames``: original input file names
* ``pred_lab``: top predicted class names for each input
* ``pred_prob``: matching probabilities
* ``aphia_ids``: Aphia identifiers when they are available in the dataset metadata

Checkpoint selection
--------------------

The API loads one timestamp and one checkpoint for inference:

* if no timestamp is provided, it uses the latest timestamp under ``models/``
* if no checkpoint is provided, it prefers ``.keras`` checkpoints over older formats
* when training stored ``use_best_model = true`` and ``best_model.keras`` exists, inference prefers that checkpoint

Operational notes
-----------------

* ZIP prediction extracts the archive to a temporary directory and scans recursively for image files
* prediction writes a JSON artifact to the configured predictions directory
* training validates the ``images_directory`` path before starting
* if there are no models yet, the API can still be used for training, but not for inference
