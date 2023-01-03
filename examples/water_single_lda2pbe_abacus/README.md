# Example for water (LDA to PBE) : using ABACUS in scf iteration

This is an example on how to use `deepks` library to train a **LDA-to-PBE** model for water molecules, with SCF calculation done by **[ABACUS](https://github.com/deepmodeling/abacus-develop)**. 

The sub-folders are grouped as following:

- `systems` contains atom data with PBE energy and force.
- `iter` contains input files used to train a self consistent model iteratively (DeePKS).

## Running this example locally

To train a self-consistant model with ABACUS, you can `cd iter` and run:

`python -u -m deepks iterate machines.yaml params.yaml systems.yaml scf_abacus.yaml >> log.iter 2> err.iter`

or directly

`cd iter && bash run.sh`

## Running this example with DPDispatcher

If you want the jobs submitted to supercomputer platforms by **[DPDispatcher](https://github.com/deepmodeling/dpdispatcher)** , you need to set `dispatcher` as `"dpdispatcher"`and set `dpdispatcher_machine` and `dpdispatcher_resources` according to [DPDispatcher docs](https://dpdispatcher.readthedocs.io/) (see `machines_dpdispatcher.yaml`). Remember to set the remote `abacus_path` (see `scf_abacus.yaml`). Taking **[Bohrium](https://bohrium.dp.tech/)** as an example, `cd iter` and run:

`python -u -m deepks iterate machines_dpdispatcher.yaml params.yaml systems.yaml scf_abacus.yaml >> log.iter 2> err.iter`

or directly

`cd iter && bash run_dpdispatcher.sh`

## Prameters for ABACUS 
ABACUS parameters are specified in `scf_abacus.yaml`. These parameters can be divided into 3 categories:

- Paras for running ABACUS:
   - `abacus_path`: the path of ABACUS binary executable file (ABACUS) 
   - `run_cmd`: command to run ABACUS, usually `mpirun`

- Paras for `INPUT` file:
    - `ntype`: number of atom types
    - `nbands`: total number of bands to calculate
    - `ecutwfc`: energy cutoff for plane wave functions
    - `scf_thr`: the charge density error threshold between two sequential density from electronic iterations to judge convergence
    - `scf_nmax`: max scf steps
    - `dft_functional`: Exchange-Correlation functional, which can be 'lda', 'gga', 'pbe', etc
    - `gamma_only`: 1 for gamma-point, 0 for multi-kpoints
    - `cal_force`: set 1 to calculate force, default 0
    - `cal_stress`: set 1 to calculate stress, default 0

- Paras for `STRU` file:
    - `orb_files`: paths of atom orbitals, a list of str with the order same as the order of atom types in `atoms.npy`
    - `pp_files`: paths of pseudo potential files, a list of str with the order same as the order of atom types in `atoms.npy`
    - `proj_file`: path of orbital file for descripor basis
    - `lattice_constant`: spacial period, in Bohr
    - `lattice_vector`: spacial period of x,y and z, in Bohr
    - `coord_type`: type of the coordinates, Cartesian or Direct
tips: you can set different STRU parasmeters for each data group, by adding a `stru_abacus.yaml`in systems/group.xx.

There are some other important parameters for using ABACUS in `params.yaml ` and `machines.yaml`:
- `use_abacus`: set `true` to calculate SCF by ABACUS, `false` for using PySCF
- `task_per_node`: how many cpu cores are used for calculating each frame in using ABACUS
- `sub_size`: how many frames are calculated simultaneously

**Caution**: The meanings of `task_per_node` and `sub_size` when using ABACUS are kind of different to using PySCF, because ABACUS supports parallel calculation in **each single frame (configuration)**  while PySCF not. Each frame's calculation is an "ABACUS task" with a unique workdir. 


## Data units
`deepks` accept `.npy` data in following units: 
Property | Unit
---	     | :---:
Length	 | Bohr, Å, or fractional coordinates
Energy	 | $E_h$ (Hartree)
Force	   | $E_h$/Bohr ($E_h$/Å if from xyz)

In this example, each grouped folder contain an `atom.npy` that has shape `n_frames x n_atoms x 4` and the four elements correspond to the nuclear charge of the atom and its three spacial coordinates.
Other properties can be provided as separate files like `energy.npy` and `force.npy`.


## Train a model: DeePHF or DeePKS

Set `n_iter = 0` to train a perturbative energy model, which is a pure machine learning task. Please see [DeePHF paper](https://arxiv.org/pdf/2005.00169.pdf) for a detailed explanation of the construction of the descriptors. 

After SCF calculation, the following result files will be saved in `iter.init/00.scf/data_train(test)` prepared to training:

- descriptor (`dm_eig`) 
- energy labels (`l_e_delta`)
- force labels (`l_f_delta`) , only when `cal_force = 1 `

Set `n_iter` to a positive integer to train a self consistent model, following the iterative approach described in [DeePKS paper](https://arxiv.org/pdf/2008.00167.pdf). In scf calculation , a trained model file `model.pth` will be loaded into ABACUS. And beside above result files, a `grad_vx` file will appear in `iter.xx/00.scf/data_train(test)` when force label is used in training.


## Explaination for output files

### RECORD file

For each iteration, each sub-step would correspond to a row in `RECORD` file, used to indicate which steps have finished. It would have three numbers. The first one correspond to the iteration number. The second one correspond to the sub-folder in the iteration and the third correspond to step in that folder.

- (`X 0 0`): pre process of SCF, generate ABACUS work direcotry and input files in each group of `systems`
- (`X 0 1`): run SCF calculation with given model by ABACUS
- (`X 0 2`): concatenate and check the SCF result and print convergence and accuracy
- (`X 1 0`): train a new model using the old one as starting point
- (`X 1 1`): test the model on all data to see the pure fitting error

### log file

One can check `iter.*/00.scf/log.data` for stats of SCF results, `iter*/01.train/log.train` for training curve and `iter*/01.train/log.test` for model prediction of $E_\delta$ (e_delta).

 
