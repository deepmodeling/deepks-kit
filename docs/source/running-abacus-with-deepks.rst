


Running ABACUS with DeePKS model
================

Once the DeePKS training process is converged,
users may perform ABACUS SCF calculation with 
the DeePKS model loaded. Compared to a normal 
ABACUS SCF job with lcao basis, one needs to 
add the following keywords to ``INPUT`` file: 

.. code-block:: plaintext

    <...other keywords>
    deepks_scf: 1                 # run SCF job with DeePKS model
    deepks_model: model.ptg       # provide the model file; should be correctly located

Note that the path of ``model.ptg`` should be 
provided along with the file itself. The above
input works only if ``model.ptg`` and ``INPUT``
are placed under the same directory. 

Users also need to provide the projector file 
along with the path in ``STRU``:

.. code-block:: plaintext

    <...other keywords>
    NUMERICAL_DESCRIPTOR
    jle.orb

An example of running ABACUS SCF with trained
DeePKS model has been provided `here <https://github.com/deepmodeling/abacus-develop/tree/develop/examples/deepks/lcao_H2O>`_. 
