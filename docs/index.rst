Phyto Plankton Classification
=============================

This documentation belongs to the full ``phyto-plankton-classification`` repository.

It is the general project-level home for:

* local CLI workflows
* local DEEPaaS API workflows
* notebook workflows
* Docker usage
* AI4OS and OSCAR deployment
* project-level models, assets, and integration material

If you want package-focused installation, command explanations, and reusable package documentation, use the companion repository instead:

* ``planktonclas`` package repo: https://github.com/woutdecrop/planktonclas

Home
----

This repository supports five main approaches:

1. package / local CLI usage through ``planktonclas``
2. local API usage through DEEPaaS
3. notebook usage
4. Docker usage
5. AI4OS / OSCAR deployment

The important thing for new users is:

* you do not have to use every workflow
* these are alternative ways to use the same project and package
* package-only details live in ``planktonclas``

How To Read These Docs
----------------------

Read the docs in this order:

1. :doc:`installation` to decide how you want to set up or launch the project
2. :doc:`quickstart` to choose your path
3. one of the numbered workflow pages below
4. :doc:`reference` for project structure, outputs, and conventions

Use ``planktonclas`` for:

* package installation
* package command explanations
* command-line workflow details
* package-level API and notebook documentation

Workflow Pages
--------------

* :doc:`python_usage` is Option 1 and explains the package / local CLI path at a high level, then points to the fuller ``planktonclas`` docs
* :doc:`api_usage` is Option 2 and explains the local API path in this repository
* :doc:`notebooks` is Option 3 and explains the notebook path in this repository
* Docker and AI4OS / OSCAR are described in :doc:`installation` and :doc:`quickstart` as Options 4 and 5

Citation
--------

If you use this project, please consider citing:

* Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   api_usage
   python_usage
   notebooks
   reference
