Installation
=====

.. _installation:

DeePKS-kit
------------

DeePKS-kit is a pure python library so it can be installed following the standard `git clone` then `pip install` procedure. Note that the two main requirements `pytorch` and `ABACUS` will not be installed automatically so you will need to install them manually in advance. Below is a more detailed instruction that includes installing the required libraries in the environment.

We use `conda` here as an example. So first you may need to install `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

To reduce the possibility of library conflicts, we suggest create a new environment (named `deepks`) with basic dependencies installed (optional):

.. code-block:: console

  conda create -n deepks numpy scipy h5py ruamel.yaml paramiko
  conda activate deepks

Now you are in the new environment called `deepks`.
Next, install `PyTorch <https://pytorch.org/get-started/locally/>`_

.. code-block:: console

  # assuming a GPU with cudatoolkit 10.2 support
  conda install pytorch cudatoolkit=10.2 -c pytorch
  

Once the environment has been setup properly, using `pip` to install DeePKS-kit:

.. code-block:: console

  $ pip install git+https://github.com/deepmodeling/deepks-kit@abacus



ABACUS with DeePKS enabled
------------

To run DeePKS-kit in connection with ABACUS, users first need to install ABACUS with DeePKS enabled. 
Details of such installation guide can be found at `installation with DeePKS <https://abacus.deepmodeling.com/en/latest/advanced/install.html#build-with-deepks>`_. 



DPDispatcher (optional)
----------------

While DeePKS-kit has its built-in job dispacther, users are welcome to use DPDispatcher for automatic job submission. 
The usage of these two types of dispatchers is given in xxx. DPDispacther can simply be installed via 

.. code-block:: console

  $ pip install dpdispatcher

More details about DPDispacther can be found via `DPDispatcher's documentation <https://docs.deepmodeling.com/projects/dpdispatcher/en/latest/>`_. 
