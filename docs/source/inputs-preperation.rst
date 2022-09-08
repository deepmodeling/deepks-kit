Input files preperation
=======================

To run DeePKS-kit in connection with ABACUS, a bunch of input files are required so as to iteratively perform the SCF jobs on ABACUS and the training jobs on DeePKS-kit. Here we will use **single water molecule** as an example to show the required input files for the training of an **LDA**-based DeePKS model that provides **PBE** target energies and forces. 

As can be seen in this example, 1000 structures of the single water molecules with corresponding PBE property labels (including energy and force) have been prepared in advance. Four subfolders, i.e., ``group.00-03`` and be found under the folder ``systems``: ``group.00-group.02`` contain 300 frames each and can be applied as training sets, while ``group.03`` contains 100 frames and can be applied as testing set.
The prepared file structure of a ready-to-run DeePKS iterative traning process should basically look like

.. image:: 
  ./deepks_tree.jpg
  :width: 300

scf_abacus.yaml
----------------

This file controls the SCF jobs performed in ABACUS. The ``scf_abacus`` block controls the SCF jobs after the init iteration, i.e., with DeePKS model loaded, while the ``init_scf_abacus`` controls the initial SCF jobs, i.e., bare LDA or PBE SCF calculaiton. The reason to divide this file into two blocks is that after the init iteration, the SCF calculaitons with DeePKS model loaded are sometimes found hard to converge to a tight threshold, e.g., ``scf_thr = 1e-7``. Therefore we might want to slightly loose that threshold after the init iteration.

Below is a sample ``scf_abacus.yaml`` file for single water molecule, with the explanation of each keyword. Please refer to xxx for a more detailed explanation of the input parameters in ABACUS.

.. code-block:: yaml

  scf_abacus:
    # INPUT args; keywords that related to INPUT file in ABACUS
    ntype: 2                    # int; number of different atom species in this calculations, e.g., 2 for H2O
    nbands: 8                   # int; number of bands to be calculated; optional
    ecutwfc: 50                 # real; energy cutoff, unit: Ry
    scf_thr: 1e-7               # real; SCF convergence threshold for density error; 5e-7 and below is acceptable
    scf_nmax: 50                # int; maximum SCF iteration steps
    dft_functional: "lda"       # string; name of the baseline density functional
    gamma_only: 1               # bool; 1 for gamma-only calculation
    cal_force: 1                # bool; 1 for force calculation
    cal_stress: 0               # bool; 1 for stress calculation
    deepks_descriptor_lmax: 2   # int; maximum angular momentum of the descriptor basis; 2 is recommended
    
    # STRU args; keywords that related to INPUT file in ABACUS
    # below are default STRU args, users can also set them for each group in  
    # ../systems/group.xx/stru_abacus.yaml
    orb_files: ["O_gga_6au_60Ry_2s2p1d.orb", "H_gga_6au_60Ry_2s1p.orb"] # atomic orbital file list for each element; 
                                                                        # order should be consistent with that in atom.npy
    pp_files: ["O_ONCV_PBE-1.0.upf", "H_ONCV_PBE-1.0.upf"]              # pseudopotential file list for each element; 
                                                                        # order should be consistent with that in atom.npy             
    proj_file: ["jle.orb"]                                              # projector file; generated in ABACUS; see file desriptions for more details
    lattice_constant: 1                                                 # real; lattice constant
    lattice_vector: [[28, 0, 0], [0, 28, 0], [0, 0, 28]]                # [3, 3] matrix; lattice vectors
    
    # cmd args; keywords that related to running ABACUS
    run_cmd : "mpirun"                                                  # run command
    abacus_path: "/usr/local/bin/abacus"                                # ABACUS executable path
  
  # below is the init_scf_abacus block, which is basically same as above
  # the only thing is that the recommended value for scf_thr is 1e-7
  init_scf_abacus:
    orb_files: ["O_gga_6au_60Ry_2s2p1d.orb", "H_gga_6au_60Ry_2s1p.orb"]
    pp_files: ["O_ONCV_PBE-1.0.upf", "H_ONCV_PBE-1.0.upf"]
    proj_file: ["jle.orb"]
    ntype: 2
    nbands: 8
    ecutwfc: 50
    scf_thr: 1e-7
    scf_nmax: 50
    dft_functional: "lda"
    gamma_only: 1
    cal_force: 0
    deepks_descriptor_lmax: 2
    lattice_constant: 1
    lattice_vector: [[28, 0, 0], [0, 28, 0], [0, 0, 28]]
    #cmd args
    run_cmd : "mpirun"
    abacus_path: "/usr/local/bin/abacus"



machine.yaml
--------------

.. note::

   This file is *not* required when running jobs on Bohrium via DPDispachter. In such case, users need to prepare machine_bohrium.yaml instead.

To run ABACUS-DeePKS training process on a local machine or on a cluster via slurm or PBS, it is recommended to use the DeePKS built-in dispatcher and prepare ``machine.yaml`` file as follows. 

.. code-block:: yaml

  # this is only part of input settings. 
  # should be used together with systems.yaml and params.yaml
  scf_machine:
    group_size: 125        # number of SCF jobs that are grouped and submitted together; these jobs will be run sequentially
    resources:
      cpus_per_task: 1     # number of CPUs for one SCF job
    sub_size: 1            # keyword for PySCF; set to 1 for ABACUS SCF jobs
    dispatcher: 
      context: local       # "local" to run on local machine, or "ssh" to run on a remote machine
      batch: shell         # set to shell to run on local machine, you can also use `slurm` or `pbs`

  train_machine: 
    dispatcher: 
      context: local       # "local" to run on local machine, or "ssh" to run on a remote machine
      batch: shell         # set to shell to run on local machine, you can also use `slurm` or `pbs`
      remote_profile: null # use lazy local
    python: "python"       # use python in path
    # resources are no longer needed, and the task will use gpu automatically if there is one

  # other settings (these are default; can be omitted)
  cleanup: false           # whether to delete slurm and err files
  strict: true             # do not allow undefined machine parameters

  #paras for abacus
  use_abacus: true         # use abacus in scf calculation


machine_bohrium.yaml
-------------------------

.. note::

   This file is *not* required when running jobs on a local machine or on a cluster via slurm or PBS *with the built-in dispatcher*. In such case, users need to prepare machine.yaml instead. That being said, users may also modify keywords in this file to submit jobs to a cluster via slurm or PBS. Please refer to DPDispatcher documentation for more details on slurm/PBS job submission. 

To run ABACUS-DeePKS training process on Bohrium, users need to use DPDispatcher and prepare ``machine_bohrium.yaml`` file as follows. Most of the keyword in this file share the same meaning as those in ``machine.yaml``. The unique part here is to specify keywords in ``dpdispatcher_resources:`` block. 

.. code-block:: yaml

  # this is only part of input settings. 
  # should be used together with systems.yaml and params.yaml
  scf_machine: 
    resources: 
      cpus_per_task: 4
    dispatcher: dpdispatcher 
    dpdispatcher_resources:
      number_node: 1
      cpu_per_node: 8
      group_size: 125
      source_list: [/opt/intel/oneapi/setvars.sh]
    sub_size: 1 
    dpdispatcher_machine: 
      context_type: lebesguecontext
      batch_type: lebesgue
      local_root: ./
      remote_profile:
        email: (your-account-email)
        password: (your-passward)
        program_id: (your-program-id)
        input_data:
          log_file: log.scf 
          err_file: err.scf
          job_type: indicate
          grouped: true
          job_name: deepks-scf
          disk_size: 100
          scass_type: c8_m8_cpu
          platform: ali
          image_name: abacus-workshop
          on_demand: 0
  train_machine: 
    dispatcher: dpdispatcher 
    dpdispatcher_machine: 
      context_type: lebesguecontext
      batch_type: lebesgue
      local_root: ./
      remote_profile:
        email: (your-account-email)
        password: (your-passward)
        program_id: (your-program-id)
        input_data:
          log_file: log.train 
          err_file: err.train
          job_type: indicate
          grouped: true
          job_name: deepks-train
          disk_size: 100
          scass_type: c8_m8_cpu
          platform: ali
          image_name: abacus-workshop
          on_demand: 0
    dpdispatcher_resources:
      number_node: 1
      cpu_per_node: 8
      group_size: 1
      source_list: [~/.bashrc]
    python: "/usr/bin/python3" # use python in path
    # resources are no longer needed, and the task will use gpu automatically if there is one

  # other settings (these are default, can be omitted)
  cleanup: false # whether to delete slurm and err files
  strict: true # do not allow undefined machine parameters

  #paras for abacus
  use_abacus: true # use abacus in scf calculation


params.yaml
------------

projector file
--------------

orbital files and pseudopotential files
---------------------------------------


