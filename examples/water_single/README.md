# Example for water

This is an example on how to use `deepqc` library to train a energy functional for water molecules. The sub-folders are grouped as following:

- `systems` contains all data that has been prepared in `deepqc` format.
- `init` contains input files used to train a (perturbative) energy model (DeePHF).
- `iter` contains input files used to train a self consistent model iteratively (DeePKS).
- `withdens` contains input files used to train a SCF model with density labels.


## Prepare data

To prepare data, please first note that `deepqc` use the following convention of units. 

Property | Unit
---	     | :---:
Length	 | Å
Energy	 | $E_h$ (Hartree)
Force	 | $E_h$/Å

`deepqc` accepts data in three formats. 

- **single `xyz` files** with properties saved as separate files sharing same base name.
  e.g. for `0000.xyz`, its energy can be saved as `0000.energy.npy`, and forces as `0000.force.npy`, density matrix as `0000.dm.npy` in the same folder.
- **grouped into folders** with same number of atoms. 
  Such folder should contain an `atom.npy` that has shape `n_frames x n_atoms x 4` and the four elements correspond to the nuclear charge of the atom and its three spacial coordinates.
  Other properties can be provided as separate files like `energy.npy` and `force.npy`.
- **grouped with explicit `type.raw` file** with all frames have same type of elements.
  This is similar as above, only that `atom.npy` is substituted by `coord.npy` containing pure special coordinates and a `type.raw` containing the element type for all the frames of this system. This format is very similar to the one used in DeePMD-Kit, but the `type.raw` must contains real element types here.

Note the property files are optional. For pure SCF calculation, they are not needed. But in order to train a model, they are needed as labels.

The two grouped data formats can be converted from the xyz format by using [this script](../../scripts/convert_xyz.py). As an example, the data in `systems` folder is created using the following command.
```
python ../../scripts/convert_xyz.py some/path/to/all/*.xyz -d systems -G 300 -P group
```


## Train an energy model

To train a perturbative energy model is a pure machine learning task. Please see [DeePHF paper](https://arxiv.org/pdf/2005.00169.pdf) for a detailed explanation of the construction of the descriptors. Here we provide two sub-commands. `deepqc scf` can do the Hartree-Fock calculation and save the descriptor (`dm_eig`) as well as labels (`l_e_delta` for energy and `l_f_delta` for force) automatically. `deepqc train` can use the dumped descriptors and labels to train a neural network model.

To further simplify the procedure, we can combine the two steps together and use `deepqc iterate` to run them sequentially. The required input files and execution scripts can be found in `init` folder. There `machines.yaml` specifies the resources needed for the calculations. `params.yaml` specifies the parameters needed for the Hartree-Fock calculation and neural network training. `systems.yaml` specifies the data needed for training and testing. Note the name `init` is because it also serves as an initialization step of the self consistent training described below. For same reason, the `niter` attribute in `params.yaml` is set to 0, to avoid iterative training.

As shown in `run.sh`, the input files can be loaded and run by 
```
deepqc iterate machines.yaml params.yaml systems.yaml
```
where `deepqc` is a shortcut for `python -m deepqc`. Or one can directly use `./run.sh` to run it in background. Make sure you are in `init` folder before you run the command.


## Train a self consistent model

To train a self consistent model we follow the iterative approach described in [DeePKS paper](https://arxiv.org/pdf/2008.00167.pdf). We provide `deepqc iterate` as a tool to do the iteration automatically. Same as above, the example input file and execution scripts can be found in `iter` folder. Note here instead of splitting the input file into three, we combined all input settings in one `args.yaml` file, to show that `deepqc iterate` can take variable number of input files. The file provided at last will have highest priority.

For each iteration, there will be four steps using four corresponding tools provided by `deepqc`. Each step would correspond to a row in `RECORD` file, used to indicate which steps have finished. It would have three numbers. The first one correspond to the iteration number. The second one correspond to the sub-folder in the iteration and the third correspond to step in that folder.

- `deepqc scf` (`X 0 0`): do the SCF calculation with given model and save the results
- `deepqc stats` (`X 0 1`): check the SCF result and print convergence and accuracy
- `deepqc train` (`X 1 0`): train a new model using the old one as starting point
- `deepqc test` (`X 1 1`): test the model on all data to see the pure fitting error

To run the iteration, again, use `./run.sh` or the following command
```
deepqc iterate args.yaml
```
Make sure you are in `iter` folder before you run the command.

One can check `iter.*/00.scf/log.data` for stats of SCF results, `iter*/01.train/log.train` for training curve and `iter*/01.train/log.test` for model prediction of $E_\delta$ (e_delta).


## Train a self consistent model with density labels

We provide in `withdens` folder a set of inputs of using density labels during the iterative training (as additional penalty terms in the Hamiltonian). We again follow the [DeePKS paper](https://arxiv.org/pdf/2008.00167.pdf) to add first a randomized penalty using Coulomb loss for 5 iterations and then remove it and relax for another 5 iterations.

Most of the inputs are same as the normal iterative training case described in the last section, which we put in the `base.yaml` Only that we are overwritten `scf_input` in `penalty.yaml` to add the penalties. Also we change the number of iteration `n_iter` in both `penalty.yaml` and `relax.yaml`.

`pipe.sh` shows how we combine the different inputs together. A simplified version is as follows:
```
deepqc iterate base.yaml penalty.yaml && deepqc iterate base.yaml relax.yaml
```
The `iterate` command can take multiple input files and the latter ones would overwrite the former ones.

Again, running `./run.sh` in the `withdens` folder would run the commands in the background. You can check the results in `iter.*` folders like above.