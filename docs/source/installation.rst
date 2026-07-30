🐳 Installation
===============

.. _installation:

To use **disclOSE**, you first need to install it using the following procedure:

From Git
--------

You can install **disclOSE** from `git <https://git-scm.com/>`_ if you want to have an editable version of it. This can be usefull for contributing to **OSEkit**, or to be able to easily update to the latest version.

The package will be installed with `uv <https://docs.astral.sh/uv/>`_, which is a Python package and project manager. Please refer to the `uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_ for installing uv.

To download the repository, simply clone it from git: ::

    git clone https://github.com/Project-OSmOSE/disclOSE.git

Then, you can pull the latest update: ::

    git pull origin main

You can now install the package using uv from the cloned repository: ::

    uv sync

This will create a virtual environment within your cloned repository and install all required dependencies for using and contributing to **disclOSE**.
