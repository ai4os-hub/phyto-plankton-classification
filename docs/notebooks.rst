Notebooks
=========

Overview
--------

The repository includes notebooks for data exploration, preprocessing, training, evaluation, and explainability. They live in ``notebooks/`` and are the most convenient entry point for interactive use.

Notebook guide
--------------

``1.0-Dataset_exploration.ipynb``
   Explore class balance, dataset composition, and other useful dataset statistics before training.

``1.1-Image_transformation.ipynb``
   Inspect and adapt image preprocessing so a new dataset matches the expected training input format.

``1.2-Image_augmentation.ipynb``
   Experiment with augmentation strategies to expand or stress-test the training data.

``2.0-Model_training.ipynb``
   Run model training interactively while tuning configuration parameters.

``3.0-Computing_predictions.ipynb``
   Predict one image or many images and inspect raw model outputs.

``3.1-Prediction_statistics.ipynb``
   Evaluate predictions on a labeled test split and inspect accuracy-oriented metrics and confusion-style summaries.

``3.2-Saliency_maps.ipynb``
   Visualize explainability outputs such as saliency-style attribution maps.

When to use notebooks
---------------------

Use the notebooks when you want:

* iterative experimentation
* quick visual inspection of images and augmentations
* interactive training and debugging
* prediction analysis without going through the web API

Working inside Docker
---------------------

Inside the prepared environment, start Jupyter with:

.. code-block:: bash

   deep-start -j

Then open the generated URL in a browser and navigate to ``notebooks/``.

Recommended workflow
--------------------

1. start with dataset exploration
2. verify transformations and augmentations
3. run model training
4. compute predictions
5. inspect prediction statistics
6. use saliency maps when you need explainability
