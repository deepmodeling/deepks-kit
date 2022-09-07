Input files preperation
=======================

To run DeePKS-kit in connection with ABACUS, a bunch of input files are required so as to iteratively perform the SCF jobs on ABACUS and the training jobs on DeePKS-kit. Here we will use **single water molecule** as an example to show the required input files for the training of an **LDA**-based DeePKS model that provides **PBE** target energies and forces. 

As can be seen in this example, 1000 structures of the single water molecules with corresponding PBE property labels (including energy and force) have been prepared in advance. Four subfolders, i.e., ``group.00-03`` and be found under the folder ``systems``: ``group.00-group.02`` contain 300 frames each and can be applied as training sets, while ``group.03`` contains 100 frames and can be applied as testing set.

scf_abacus.yaml
----------------


scf_abacus_dpdispatcher.yaml
-----------------------------


machine.yaml
--------------



machine_dpdispatcher.yaml
-------------------------
scf_machine: 
    | type: ``dict``
    | argument path: ``machine``

    resources: 
        | type: ``dict``
        | argument path: ``scf_machine/resources``
        
        cpus_per_task:
            | type: ``int``
            | argument path: ``scf_machine/resources/cpu_per_task``
            
            The number of CPUs running for a single SCF job. 
            
    dispatcher:
        | tpye: ``string``
        | argument path: ``scf_machine/dispatcher``
        
        The type of dispatcher chosen for job submission, which should be set as ``dpdispatcher`` here.
        
        The batch job system type. Option: Slurm, PBS, Lebesgue, Shell
        
    dpdispatcher_resources:
        | tpye: ``dict``
        | argument path: ``scf_machine/dpdispatcher_resources``
        
        number_node:
            | type: ``int``
            | argument path: ``scf_machine/resources/cpu_per_task``

    local_root: 
        | type: ``str`` | ``NoneType``
        | argument path: ``machine/local_root``

        The dir where the tasks and relating files locate. Typically the project dir.

    remote_root: 
        | type: ``str`` | ``NoneType``, optional
        | argument path: ``machine/remote_root``

        The dir where the tasks are executed on the remote machine. Only needed when context is not lazy-local.

    clean_asynchronously: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``machine/clean_asynchronously``

        Clean the remote directory asynchronously after the job finishes.


    Depending on the value of *context_type*, different sub args are accepted. 

    context_type:
        | type: ``str`` (flag key)
        | argument path: ``machine/context_type`` 
        | possible choices: LocalContext, LazyLocalContext, LebesgueContext, SSHContext, HDFSContext, DpCloudServerContext

        The connection used to remote machine. Option: LocalContext, SSHContext, HDFSContext, DpCloudServerContext, LazyLocalContext, LebesgueContext


    When *context_type* is set to ``LocalContext`` (or its aliases ``localcontext``, ``Local``, ``local``): 

    remote_profile: 
        | type: ``dict``, optional
        | argument path: ``machine[LocalContext]/remote_profile``

        The information used to maintain the connection with remote machine. This field is empty for this context.


    When *context_type* is set to ``LazyLocalContext`` (or its aliases ``lazylocalcontext``, ``LazyLocal``, ``lazylocal``): 

    remote_profile: 
        | type: ``dict``, optional
        | argument path: ``machine[LazyLocalContext]/remote_profile``

        The information used to maintain the connection with remote machine. This field is empty for this context.


    When *context_type* is set to ``LebesgueContext`` (or its aliases ``lebesguecontext``, ``Lebesgue``, ``lebesgue``): 

    remote_profile: 
        | type: ``dict``
        | argument path: ``machine[LebesgueContext]/remote_profile``

        The information used to maintain the connection with remote machine.

        email: 
            | type: ``str``
            | argument path: ``machine[LebesgueContext]/remote_profile/email``

            Email

        password: 
            | type: ``str``
            | argument path: ``machine[LebesgueContext]/remote_profile/password``

            Password

        program_id: 
            | type: ``int``
            | argument path: ``machine[LebesgueContext]/remote_profile/program_id``

            Program ID

        keep_backup: 
            | type: ``bool``, optional
            | argument path: ``machine[LebesgueContext]/remote_profile/keep_backup``

            keep download and upload zip

        input_data: 
            | type: ``dict``
            | argument path: ``machine[LebesgueContext]/remote_profile/input_data``

            Configuration of job


    When *context_type* is set to ``SSHContext`` (or its aliases ``sshcontext``, ``SSH``, ``ssh``): 

    remote_profile: 
        | type: ``dict``
        | argument path: ``machine[SSHContext]/remote_profile``

        The information used to maintain the connection with remote machine.

        hostname: 
            | type: ``str``
            | argument path: ``machine[SSHContext]/remote_profile/hostname``

            hostname or ip of ssh connection.

        username: 
            | type: ``str``
            | argument path: ``machine[SSHContext]/remote_profile/username``

            username of target linux system

        password: 
            | type: ``str``, optional
            | argument path: ``machine[SSHContext]/remote_profile/password``

            (deprecated) password of linux system. Please use `SSH keys <https://www.ssh.com/academy/ssh/key>`_ instead to improve security.

        port: 
            | type: ``int``, optional, default: ``22``
            | argument path: ``machine[SSHContext]/remote_profile/port``

            ssh connection port.

        key_filename: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``machine[SSHContext]/remote_profile/key_filename``

            key filename used by ssh connection. If left None, find key in ~/.ssh or use password for login

        passphrase: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``machine[SSHContext]/remote_profile/passphrase``

            passphrase of key used by ssh connection

        timeout: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``machine[SSHContext]/remote_profile/timeout``

            timeout of ssh connection

        totp_secret: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``machine[SSHContext]/remote_profile/totp_secret``

            Time-based one time password secret. It should be a base32-encoded string extracted from the 2D code.


    When *context_type* is set to ``HDFSContext`` (or its aliases ``hdfscontext``, ``HDFS``, ``hdfs``): 

    remote_profile: 
        | type: ``dict``, optional
        | argument path: ``machine[HDFSContext]/remote_profile``

        The information used to maintain the connection with remote machine. This field is empty for this context.


    When *context_type* is set to ``DpCloudServerContext`` (or its aliases ``dpcloudservercontext``, ``DpCloudServer``, ``dpcloudserver``): 

    remote_profile: 
        | type: ``dict``
        | argument path: ``machine[DpCloudServerContext]/remote_profile``

        The information used to maintain the connection with remote machine.

        email: 
            | type: ``str``
            | argument path: ``machine[DpCloudServerContext]/remote_profile/email``

            Email

        password: 
            | type: ``str``
            | argument path: ``machine[DpCloudServerContext]/remote_profile/password``

            Password

        program_id: 
            | type: ``int``
            | argument path: ``machine[DpCloudServerContext]/remote_profile/program_id``

            Program ID

        keep_backup: 
            | type: ``bool``, optional
            | argument path: ``machine[DpCloudServerContext]/remote_profile/keep_backup``

            keep download and upload zip

        input_data: 
            | type: ``dict``
            | argument path: ``machine[DpCloudServerContext]/remote_profile/input_data``

            Configuration of job


params.yaml
------------

projector file
--------------

orbital files and pseudopotential files
---------------------------------------


