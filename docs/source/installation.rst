Installation
=====

.. _installation:

DeePKS-kit
------------

DeePKS-kit is a pure python library so it can be installed following the standard `git clone` then `pip install` procedure. Note that the main requirements `pytorch` will not be installed automatically so you will need to install them manually in advance. Below is a more detailed instruction that includes installing the required libraries in the environment.

We use `conda` here as an example. So first you may need to install `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`.

To reduce the possibility of library conflicts, we suggest create a new environment (named `deepks`) with basic dependencies installed (optional):

.. code-block:: console

  conda create -n deepks numpy scipy h5py ruamel.yaml paramiko
  conda activate deepks

Now you are in the new environment called `deepks`.
Next, install `PyTorch <https://pytorch.org/get-started/locally/>` 

.. code-block:: console

  # assuming a GPU with cudatoolkit 10.2 support
  conda install pytorch cudatoolkit=10.2 -c pytorch
  

Once the environment has been setup properly, using pip to install DeePKS-kit:

.. code-block:: console

  $ pip install git+https://github.com/deepmodeling/deepks-kit/



ABACUS with DeePKS enabled
------------

To run DeePKS-kit in connection with ABACUS, the user first needs to install ABACUS with DeePKS enabled. 
Details of such installation guide can be found at `installation with DeePKS <https://github.com/deepmodeling/abacus-develop/blob/develop/docs/install.md#installation-with-deepks>`. 



DPDispatcher (optional)
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

