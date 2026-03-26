Notebooks
=========

Overview
--------

The repository includes notebooks for:

* dataset exploration
* image transformation
* augmentation
* model training
* prediction
* prediction statistics
* saliency and explainability

They are the best choice when you want an interactive workflow.

Notebook list
-------------

``1.0-Dataset_exploration.ipynb``
   Explore class balance, dataset composition, and general dataset statistics.

``1.1-Image_transformation.ipynb``
   Inspect and adapt preprocessing so a new dataset matches the expected training input format.

``1.2-Image_augmentation.ipynb``
   Experiment with augmentation strategies.

``2.0-Model_training.ipynb``
   Run model training interactively.

``3.0-Computing_predictions.ipynb``
   Predict one image or many images and inspect raw outputs.

``3.1-Prediction_statistics.ipynb``
   Evaluate predictions on a labeled split and inspect metrics and confusion-style summaries.

``3.2-Saliency_maps.ipynb``
   Visualize explainability outputs.

Finding the notebooks
---------------------

Print the packaged notebook directory with:

.. code-block:: bash

   planktonclas notebooks

If you are already running Jupyter locally, open that directory and work from there.

If you are inside an AI4OS deployment or a container image that ships the helper commands, you may also have:

.. code-block:: bash

   deep-start -j

That command is deployment-specific. It is not part of the local ``planktonclas`` CLI.

Recommended order
-----------------

1. dataset exploration
2. transformations and augmentation
3. model training
4. predictions
5. prediction statistics
6. saliency maps
