Important outputs explanation
=============================

During the training process, a bunch of outputs will be generated. First, ``ABACUS`` folder will be generated under each training/testing group (``group.xx`` under ``systems``), which further includes *N=nframes* subfolders, ``0``, ``1``, ..., ``${nframes}``. For example, for ``water_single_lda2pbe_abacus``, ``ABACUS`` in folder ``systems/group.00`` contains 300 subfolders, while ``ABACUS`` in folder ``systems/group.03`` contains 100 subfolders. Each subfolder contains the input and output file of the ABACUS SCF job of corresponding frame at current iteration, and will be overwritten on the next iteration.

For each iteration, error statistics and training outputs are generated in ``iter.xx`` folder. For example, the file structure of ``iter.init`` basically looks like:

If ``niter`` is larger than 0, then ``iter.00``, ``iter.01``, ..., will be generated at corresponding iteration. These folders share similar file structures as ``iter.init`` does. Important output files during the training processes are explained as below.

.. _log.data:

log.data
----------

path: ``iter/iter.xx/00.scf/log.data``

This file contains error statistics as well as SCF convergence ratio of each iteration. For example, for ``water_single_lda2pbe_abacus``, ``log.data`` of the init iteration (located at ``iter/iter.init/00.scf``) looks like

.. code-block:: plaintext

  Training:
    Convergence:
      900 / 900 =          1.00000
    Energy:
      ME:          -0.09730528149450003
      MAE:         0.09730528149450003
      MARE:        0.00030881151639484673
  Testing:
    Convergence:
      100 / 100 =          1.00000
    Energy:
      ME:          -0.09730505954754445
      MAE:         0.09730505954754445
      MARE:        0.0003349933606729039

where ME = mean error, MAE = mean absolute error, MARE = mean relative absolute error. MARE is calculated via removing any constant energy shift between the target and base energy. Note that only energy error is included here since only energy label is trained in the init iteration.

In this example, force label is triggered on after the init iteration by setting ``extra_label`` to be ``true`` and ``force_factor`` to be 1 in :ref:`params.yaml`. And ``log.data`` in ``iter.00/00.scf`` therefore has the force error statistics:

.. code-block:: plaintext

  Training:
    Convergence:
      899 / 900 =          0.99889
    Energy:
      ME:          1.707869318132222e-05
      MAE:         3.188871711078968e-05
      MARE:        3.054509587845316e-05
    Force:
      MAE:         0.00030976685248761896
  Testing:
    Convergence:
      100 / 100 =          1.00000
    Energy:
      ME:          1.8457155353139854e-05
      MAE:         3.5420404788446546e-05
      MARE:        3.3798956665677724e-05
    Force:
      MAE:         0.0003271656570860149

To judge whether the DeePKS model has converged, users may compare error statistics in ``log.data`` between current and former iterations, if the errors almost remain the same, the model can be considered as converged. 

.. _log.train:

log.train
------------

path: ``iter/iter.xx/01.train/log.train``

This file records the learning curve of the training process at each iteration. It should be noted that for iterations *after* the initial one, *train error (trn err)* recorded in this file corresponds to the **total error** of the training set, i.e., energy error plus the error from extra labels, while *test error (tst err)* corresponds to only the **energy error** of the testing set. For init training, both the train error and the test error correspond to the energy error since no extra label is included. 

For a successful training process, users would expect a remarkable decrease in both the train and the test error, especially during the first one or two iterations. As the iterative training goes on, the decrease in errors will gradually become subtle. 

RECORD
--------

path: ``iter/RECORD``

This file records every step taken in the iterative training process and is **crucial** when resubmitting the job. Each row of this ``RECORD`` file corresponds to a unique step, and details are given as follows:

- ``(X 0 0)``: at iteration number ``X`` (``X=0`` corresponds to ``iter.init``; ``X=1`` corresponds to ``iter.00``; ``X=2`` corresponds to ``iter.01``; etc), pre process of SCF, generate ABACUS work directory and input files in each group of systems
- ``(X 0 1)``: run SCF calculations in ABACUS 
- ``(X 0 2)``: concatenate and check the SCF result and print convergence and accuracy in :ref:`log.data` in ``iter.xx/00.scf``.
- ``(X 0)``: current SCF job done; prepare for training 
- ``(X 1 0)``: train a new model using the old one (if any) as starting point 
- ``(X 1 1)``: current training done; learning curve is recorded in :ref:`log.train` in ``iter.xx/01.train``
- ``(X 1)``: test the model on all data to see the pure fitting error in ``log.test`` in iter.xx/01.train
- ``(X)``: current iteration done 

For example, if we want to restart the training process for iter.00, then the corresponding ``RECORD`` file should look like

.. code-block:: plaintext

  0 0 0
  0 0 1
  0 0 2
  0 0
  0 1 0
  0 1 1
  0 1
  0
  1 0 0
  1 0 1
  1 0 2
  1 0
  
.. Note::
  
  To re-run the whole procedure, make sure that all ``iter.xx`` folder, ``share`` folder and ``RECORD`` file are deleted!
    

