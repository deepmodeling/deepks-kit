Getting Started
================

To give it a shot on a DeePKS-ABACUS sample run, users may try the 
single water example provided `here <https://github.com/ouqi0711/deepks-kit/tree/abacus/examples/water_single_lda2pbe_abacus>`_.

In this example, 1000 structures of the single water molecules with corresponding PBE property labels (including energy and force) have been prepared in advance. Four subfolders, i.e., ``group.00-03`` can be found under the folder ``systems``. ``group.00-group.02`` contain 300 frames each and can be applied as training sets, while ``group.03`` contains 100 frames and can be applied as testing set.
More details about the file structures and preparation are introduced at :ref:`label-preperation`.

This sample job can either be run on a local machine or on Bohrium. Users may modify the input files to make it run on various environment following the instruction in :ref:`inputs-preperation`. 
To run this job on a local machine, simply issue:

.. code-block:: bash

  cd deepks-kit/examples/water_single_lda2pbe_abacus/iter
  bash run.sh

To run this job on Bohrium (which uses DPDispacther for job submission and data gathering), simply issue:

.. code-block:: bash

  cd deepks-kit/examples/water_single_lda2pbe_abacus/iter
  bash run_dpdispatcher.sh

Outputs generated during the process are introduced in :ref:`important-outputs-explanation`.
