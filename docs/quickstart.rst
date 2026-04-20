Quickstart
==========

This page is the practical chooser for the main ways you can use this repository.

If you want the package-only quickstart and command explanations, use:

* ``planktonclas``: https://github.com/woutdecrop/planktonclas

Choose your path
----------------

Most users should choose one of these workflows:

1. Package / Local CLI
2. Local API
3. Notebooks
4. Docker
5. AI4OS / OSCAR

Option 1: Package / Local CLI
-----------------------------

Typical local workflow:

.. code-block:: bash

   planktonclas init my_project
   planktonclas validate-config --config ./my_project/config.yaml
   planktonclas train --config ./my_project/config.yaml
   planktonclas report --config ./my_project/config.yaml

For package-level details and command meaning:

* https://github.com/woutdecrop/planktonclas

Next page in this docs set:

* :doc:`python_usage`

Option 2: Local API
-------------------

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

For package-focused API details:

* https://github.com/woutdecrop/planktonclas

Next page in this docs set:

* :doc:`api_usage`

Option 3: Notebooks
-------------------

.. code-block:: bash

   pip install "planktonclas[notebooks]"
   planktonclas notebooks my_project

This copies the packaged notebooks into ``my_project/notebooks/``.

For package-focused notebook details:

* https://github.com/woutdecrop/planktonclas

Next page in this docs set:

* :doc:`notebooks`

Option 4: Docker
----------------

.. code-block:: bash

   git clone https://github.com/ai4os-hub/phyto-plankton-classification
   cd phyto-plankton-classification
   docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" ai4oshub/phyto-plankton-classification:latest /bin/bash

Inside the container, you can use the same ``planktonclas`` workflow.

Option 5: AI4OS / OSCAR
-----------------------

Use this path when you want hosted execution or deployment.

Useful links:

* `AI4OS / iMagine Marketplace <https://dashboard.cloud.imagine-ai.eu/marketplace/>`_
* `AI4OS docs <https://docs.ai4os.eu/en/latest/>`_
* `OSCAR manual deployment guide <https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar-manual.html>`_
* `OSCAR scripted deployment guide <https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar.html>`_

What comes next
---------------

After your chosen setup:

* use :doc:`python_usage` for Option 1 context
* use :doc:`api_usage` for Option 2 details
* use :doc:`notebooks` for Option 3 details
* use :doc:`reference` for project structure, outputs, and conventions
