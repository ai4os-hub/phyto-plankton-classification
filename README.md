AI4OS/DEEP Open Catalogue: Phytoplankton Classification
=======================================================
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/phyto-plankton-classification/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/phyto-plankton-classification/job/main/)

**Authors:** [Ignacio Heredia & Wout Decrop](https://github.com/IgnacioHeredia) (CSIC & VLIZ)

This package trains and serves phytoplankton image classifiers. It supports:

* local training from a project config file
* DEEPaaS API serving
* notebook-based exploration and evaluation
* reusable timestamped model outputs under a single `models/` directory

**Related publication:**  
[*Automated image classification workflow for phytoplankton monitoring*](https://doi.org/10.3389/fmars.2025.1699781)

Quick start
-----------

Create a fresh environment and install the package:

```bash
git clone https://github.com/ai4os-hub/phyto-plankton-classification
cd phyto-plankton-classification
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

Initialize a project directory with a user-editable `config.yaml`:

```bash
planktonclas init my_project
```

Or initialize a runnable demo project:

```bash
planktonclas init my_project --demo
```

Validate the generated config:

```bash
planktonclas validate-config --config .\my_project\config.yaml
```

Train a model:

```bash
planktonclas train --config .\my_project\config.yaml
```

Start the DEEPaaS API:

```bash
planktonclas api --config .\my_project\config.yaml
```

Then open:

* `http://127.0.0.1:5000/api`
* `http://127.0.0.1:5000/ui`

Command overview
----------------

The package installs a `planktonclas` CLI with these commands:

* `planktonclas init [DIR]`: create a local project with `config.yaml`, `data/`, and `models/`
* `planktonclas init [DIR] --demo`: create the same structure and copy demo images and demo split files
* `planktonclas validate-config --config PATH`: validate the config file and print resolved paths
* `planktonclas train --config PATH`: run training from a config file
* `planktonclas report --config PATH [--timestamp TS]`: generate evaluation plots and metric files into `models/<timestamp>/results/`
* `planktonclas api --config PATH`: launch the DEEPaaS API with that config
* `planktonclas list-models --config PATH`: list timestamped models in the configured project
* `planktonclas notebooks`: print the package notebooks directory

Project layout
--------------

After `planktonclas init`, the project directory looks like this:

```text
my_project/
  config.yaml
  data/
    images/
    dataset_files/
  models/
```

The expected dataset metadata files under `data/dataset_files/` are:

* required: `classes.txt`, `train.txt`
* optional: `val.txt`, `test.txt`, `info.txt`, `aphia_ids.txt`

The split files map image paths to integer labels starting at `0`.

Configuration
-------------

The package now uses a project-local `config.yaml` as the main user configuration file. The default template is shipped inside the package and copied into your project by `planktonclas init`.

Important fields:

* `general.base_directory`: base directory for project outputs
* `general.images_directory`: where training images are read from
* `model.modelname`: model backbone
* `training.epochs`: number of training epochs
* `training.batch_size`: training batch size
* `training.use_validation`: whether to use validation split
* `training.use_test`: whether to run test evaluation after training
* `testing.timestamp`: model timestamp to use for inference
* `testing.ckpt_name`: checkpoint to use for inference

Outputs
-------

Each training run creates a timestamped folder under `models/` with:

* `ckpts/`: checkpoints such as `best_model.keras`
* `conf/`: saved run configuration
* `logs/`: training log and epoch CSV
* `stats/`: serialized training metrics
* `dataset_files/`: copied split files used for the run
* `predictions/`: test-set prediction artifacts when enabled

To generate user-facing performance images after training:

```bash
planktonclas report --config .\my_project\config.yaml
```

This writes plots such as:

* `training_metrics.png`
* `confusion_matrix_counts.png`
* `confusion_matrix_normalized.png`
* `topk_accuracy.png`
* `per_class_metrics.png`
* `class_support.png`
* `classification_report.csv`
* `summary.json`

Python and notebooks
--------------------

You can still work directly with the package modules or the notebooks:

* `planktonclas.train_runfile` for training from Python
* `planktonclas.api` for DEEPaaS-facing prediction and training hooks
* `notebooks/` for exploration, prediction, and explainability workflows

Run:

```bash
planktonclas notebooks
```

to print the notebooks directory.

Docker
------

If you prefer Docker:

```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" ai4oshub/phyto-plankton-classification:latest /bin/bash
```

Inside the container, use the same CLI workflow.

Notes
-----

* `0.0.0.0` is a bind address; open `127.0.0.1` in your browser.
* Training and inference require TensorFlow to be installed in the active environment.
* The old `etc/config.yaml` workflow is now legacy. New usage should go through `planktonclas init` and the generated project-local `config.yaml`.

Acknowledgements
----------------

If you use this project, please consider citing:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781

and:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692.
