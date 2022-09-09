Important outputs explanation
=============================

During the training process, a bunch of outputs will be generated. First, ``ABACUS`` folder will be generated under each training/testing group (``group.xx`` under ``systems``), which further includes *N=nframes* subfolders, ``0``, ``1``, ..., ``${nframes}``. For example, for ``water_single_lda2pbe_abacus``, ``ABACUS`` in folder ``systems/group.00`` contains 300 subfolders, while ``ABACUS`` in folder ``systems/group.03`` contains 100 subfolders. Each subfolder contains the input and output file of the ABACUS SCF job of corresponding frame at current iteration, and will be overwritten on the next iteration.

Error statistics and training outputs are generated in ``iter`` folder. For example, if ``niter`` is set to 1, i.e., performing one additional iteration after the init iteration, the file structure of ``iter`` will basically look like:

Important output files are explained as below.

log.data
----------

path: ``iter/iter.xx/00.scf/log.data``

This file contains error statistics as well as SCF convergence ratio of each iteration. For example, for ``water_single_lda2pbe_abacus``, ``log.data`` of the init iteration (located at ``iter/iter.init/00.scf``) looks like

where ME = mean error, MAE = mean absolute error, MARE = mean relative absolute error. MARE is calculated via removing any constant energy shift between the target and base energy. Note that only energy error is included here since only energy label is trained in the init iteration.

In this example, force label is triggered on after the init iteration by setting ``extra_label`` to be ``true`` and ``force_factor`` to be 1 in :ref:`params.yaml`. 



log.train
------------

RECORD
--------
