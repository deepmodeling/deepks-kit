# DeePKS-kit

DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems,
for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

The program provides a command line interface `deepks` that contains five sub-commands, 
- `train`: train an neural network based post-HF energy functional model
- `test`: test the post-HF model with given data and show statistics
- `scf`: run self-consistent field calculation with given energy model
- `stats`: collect and print statistics of the SCF the results
- `iterate`: iteratively train an self-consistent model by combining four commands above

## Installation

DeePKS-kit is a pure python library so it can be installed following the standard `git clone` then `pip install` procedure. Note that the two main requirements `pytorch` and `pyscf` will not be installed automatically so you will need to install them manually in advance. Below is a more detailed instruction that includes installing the required libraries in the environment.

We use `conda` here as an example. So first you may need to install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To reduce the possibility of library conflicts, we suggest create a new environment (named `deepks`) with basic dependencies installed (optional):
```bash
conda create -n deepks numpy scipy h5py ruamel.yaml paramiko
conda activate deepks
```
Now you are in the new environment called `deepks`.
Next, install [PyTorch](https://pytorch.org/get-started/locally/) 
```bash
# assuming a GPU with cudatoolkit 10.2 support
conda install pytorch cudatoolkit=10.2 -c pytorch
```
and [PySCF](https://github.com/pyscf/pyscf).
```bash
# the conda package does not support python >= 3.8 so we use pip
pip install pyscf
```

Once the environment has been setup properly, using pip to install DeePKS-kit:
```bash
pip install git+https://github.com/deepmodeling/deepks-kit/
```

## Usage

An relatively detailed decrisption of the `deepks-kit` library can be found in [here](https://arxiv.org/pdf/2012.14615.pdf). Please also refer to the reference for the description of methods.

Please see [`examples`](./examples) folder for the usage of `deepks-kit` library. A detailed example with executable data for single water molecules can be found [here](./examples/water_single). A more complicated one for training water clusters can be found [here](./examples/water_cluster).

Check [this input file](./examples/water_cluster/args.yaml) for detailed explanation for possible input parameters, and also [this one](./examples/water_cluster/shell.yaml) if you would like to run on local machine instead of using Slurm scheduler.

## References

[1] Chen, Y., Zhang, L., Wang, H. and E, W., 2020. Ground State Energy Functional with Hartree–Fock Efficiency and Chemical Accuracy. The Journal of Physical Chemistry A, 124(35), pp.7155-7165.

[2] Chen, Y., Zhang, L., Wang, H. and E, W., 2021. DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory. Journal of Chemical Theory and Computation, 17(1), pp.170–181.


<!-- ## TODO

- [ ] Print loss separately for E and F in training.
- [ ] Rewrite all `print` function using `logging`.
- [ ] Write a detailed README and more docs.
- [ ] Add unit tests. -->

