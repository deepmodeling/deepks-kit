Important outputs explanation
=============================

During the training process, a bunch of outputs will be generated. First, ``ABACUS`` folder will be generated under each training/testing group (``group.xx`` under ``systems``), which further includes *N=nframes* subfolders, ``0``, ``1``, ..., ``${nframes}``. For example, for ``water_single_lda2pbe_abacus``, ``ABACUS`` in folder ``systems/group.00`` contains 300 subfolders, while ``ABACUS`` in folder ``systems/group.03`` contains 100 subfolders. Each subfolder contains the input and output file of the ABACUS SCF job of corresponding frame at current iteration, and will be overwritten on the next iteration.

For each iteration, error statistics and training outputs are generated in ``iter.xx`` folder. For example, the file structure of ``iter.init`` basically looks like:

If ``niter`` is larger than 0, then ``iter.00``, ``iter.01``, ..., will be generated at corresponding iteration. These folders share similar file structures as ``iter.init`` does. Important output files during the training processes are explained as below.

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

log.train
------------


RECORD
--------
