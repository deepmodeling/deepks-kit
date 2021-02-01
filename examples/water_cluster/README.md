# Example of water cluster

We provide here a detailed example on generating a DeePHF or DeePKS functional for water clusters, and demonstrate its generalizability with a test on proton transfer of a water hexamer ring.

Here we take `args.yaml` as the configuration file. The iteration can be run directly by execute the [`./run.sh`](./run.sh) file, which contains the following lines:
```bash
nohup python -u -m deepks iterate args.yaml >> log.iter 2> err.iter &
echo $! > PID
```
that runs the iterative learning procedure in background and record its PID in the designated file.

## System preparation

We use randomly generated water monomers, dimers and trimers as training datasets. Each dataset contains 100 near equilibrium configurations. We also include 50 tetramers as a validation dataset. We use energy and force as labels. The reference values are given by CCSD calculation with cc-pVDZ basis. The system configurations and corresponding labels are grouped into different folders by the number of atoms, follow the convention described in [another example](../water_single/README.md). Note that the default length unit in deepks is Bohr. The systems we provided here are in Angstrom, so we add a `unit.raw` file containing "Angstrom" in each system folder to specify the unit different from default. The path to the folders can be specified in the config file as follows:
```yaml
systems_train: # can also be files that containing system paths
  - ./systems/train.n[1-3]
systems_test: # if empty, use the last system of training set
  - ./systems/valid.n4 
```

## Initialization (DeePHF model)

As a first step, we need to train an energy model as the starting point of the iterative learning procedure. This consists of two steps. First, we solve the systems using the baseline method such as HF or PBE and dump descriptors needed for training the energy model. Second, we conduct the training from scratch using the previously dumped descriptors. If there is already an existing model, this step can be skipped, by provide the path of the model to the `init_model` key.

The energy model generated in this step is also a ready-to-use DeePHF model, saved at `iter.init/01.train/model.pth`. If self-consistency is not needed, the rest iteration steps can be ignored. We do not use forces as labels when training the energy model in this example.

The parameters of the init SCF calculation is specified under the `init_scf` key. The same set of parameters is also accepted as a standalone file by the `deepks scf` command when running SCF calculations directly.  We use cc-pVDZ as the calculation basis. The required fields to be dumped are `dm_eig` for descriptors and `l_e_delta` for reference correction energies as labels. In addition, we also include `e_tot` for total energy, `conv` for a record of convergence.
```yaml
dump_fields: [dm_eig, l_e_delta, conv, e_tot]
```
Additional parameters for molecule and SCF calculation can also be provided to `mol_args` and `scf_args` keys, and will be directly passed to corresponding interfaces in PySCF.

The parameters of the init training is specified under the `init_train` key. Similarly, the parameters can also be passed to `deepks train` command as a standalone file. In `model_args`, we set the construction of the neural network model with three hidden layers and 100 neurons per layer, using GELU activation function and skip connections. We also scale the output correction energies by a factor of 100 so that it is of order one and easier to learn. In `preprocess_args`, the descriptors are set to be preprocessed to have zero mean on the training set. A prefitted ridge regression with penalty strength 10 is also added to the model to speed up training. We set in `data_args` the batch size to be 16 and in `train_args` the total number of training epochs to be 50000. The learning rate starts at 3e-4 and decays by a factor of 0.96 for every 500 steps.

## Iterative learning (DeePKS model)

For self-consistency, we take the model acquired in last step and perform several additional iterations of SCF calculation and NN training. The number of iterations is set in the `n_iter` key to be 10. If it is set to 0, no iteration will be performed, which gives the DeePHF model. In the iterative learning procedure, we also include forces as labels to improve accuracy.

The SCF parameters are provided in the `scf_input` key, following the same rules as the `init_scf` key. In order to use forces as labels, we added additional `grad_vx` for the gradients of descriptors and `l_f_delta` for reference correction forces. `f_tot` is also included for the total force results.
```yaml
dump_fields: [conv, e_tot, dm_eig, l_e_delta, f_tot, grad_vx, l_f_delta]
```
Due to the complexity of the neural network functional, we use looser (but still accurate enough) convergence criteria in `scf_args`, with `conv_tol` to be 1e-6.

The training parameters are provided in the `train_input` key, similar to `init_train`. But since we are restarting from the existing model, no `model_args` is needed, and the preprocessing procedure can be turned off. In addition, we add `with_force: true` in `data_args` and `force_factor: 1` in `train_args` to enable using forces in training. The total number of training epochs is also reduced to 5000. The learning rate starts as 1e-4 and decays by a factor of 0.5 for every 1000 steps.

## Machine settings

How the SCF and training tasks are executed is specified in `scf_machine` and `train_machine`, respectively. Currently, both the initial and the following iterations share the same machine settings. In this example, we run our tasks on local computing cluster with Slurm as the job schedular. The platform to run the tasks is specified under the `dispatcher` key, and the computing resources assigned to each task is specified under `resources`. The setting of this part differs on every computing platform. We provide here our `training_machine` settings as an example:
```yaml
dispatcher: 
  context: local
  batch: slurm
  remote_profile: null # unnecessary in local context
resources:
  time_limit: '24:00:00'
  cpus_per_task: 4
  numb_gpu: 1
  mem_limit: 8 # gigabyte
python: "python" # use python in path
```
where we assign four CPU cores and one GPU to the training task, and set its time limit to be 24 hours and memory limit to be 8GB. The detailed settings available for `dispatcher` and `resources` can be found in the document of DP-GEN software, with a slightly different interface.

## Testing the model

During each iteration of the learning procedure, a brief summary on the accuracy of the SCF calculation can be found in `iter.xx/00.scf/log.data`. Average energy and force (if applicable) errors are shown for both training and validation dataset. The results of the SCF calculations is also stored in `iter.xx/00.scf/data_train` and `iter.xx/00.scf/data_test` grouped by training and testing systems.

After we finished our 10 iterations, the resulted DeePKS model can be found at `iter.09/01.train/model.pth`. The model can be used in either a python script creating the extended PySCF class, or directly the `deepks scf` command. As a testing example, we run the SCF calculation using the learned DeePKS model on the simultaneous six proton transfer path of a water hexamer ring. 
The command can be found in [test.sh](./test.sh).
The results of each configuration during the proton transfer are grouped in the `test_result` folder. 

We can see that all the predicted energy falls within the chemical accuracy range of the reference value given by the CCSD calculation. We note that none of the training dataset includes dissociated configurations in the proton transfer case. The DeePKS model trained on up to three water molecules exhibits good transferability, even in the bond breaking case.