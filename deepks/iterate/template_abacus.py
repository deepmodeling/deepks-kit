from build.lib.deepks.scf import run
import os
from posixpath import supports_unicode_filenames
import sys
import numpy as np
from glob import glob
from deepks.utils import check_list
from deepks.utils import flat_file_list, load_dirs
from deepks.utils import get_sys_name, load_sys_paths
from deepks.task.task import PythonTask, ShellTask
from deepks.task.task import BatchTask, GroupBatchTask
from deepks.task.workflow import Sequence
from deepks.iterate.template import check_system_names, make_cleanup


DATA_TRAIN = "data_train"
DATA_TEST  = "data_test"
MODEL_FILE = "model.pth"
CMODEL_FILE = "model.ptg"
PROJ_BASIS = "jle.orb"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"

SYS_TRAIN = "systems_train"
SYS_TEST = "systems_test"
DEFAULT_TRAIN = "systems_train.raw"
DEFAULT_TEST = "systems_test.raw"

NAME_TYPE = {   'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
            'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
        'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
        'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
        'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
        'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
        'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
        'Ba': 56, #'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            ## 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            ## 'Er': 68, 'Tm': 69, 'Yb': 70, 
            ## 'Lu': 71, 
        'Hf': 72, 'Ta': 73,
        'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
        'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 
            ## 'Po': 84, #'At': 85,
            ## 'Rn': 86, #'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
            ## 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
            ## 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            ## 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
            ## 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
            ## 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
        } #dict
WEIGHT_TYPE = {   '1.007': 1, '4.002': 2, '6.938': 3, '9.012': 4, '10.806': 5, '12.0096': 6, '14.00643': 7,
            '15.999': 8, '18.998': 9, '20.1797': 10, '22.99': 11, '24.305': 12, '26.98': 13,
        '28.084': 14, '30.973': 15, '32.059': 16, '35.446': 17, '39.948': 18, '39.0983': 19,
        '40.078': 20, '44.956': 21, '47.867': 22, '50.9415': 23, '51.9961': 24, '54.938': 25,
        '55.845': 26, '58.933': 27, '58.6934': 28, '63.546': 29, '65.38': 30, '69.732': 31,
        '72.63': 32, '74.9216': 33, '78.96': 34, '79.904': 35, '83.798': 36, '85.4678': 37,
        '87.62': 38, '88.90585': 39, '91.224': 40, '92.90638': 41, '95.96': 42, '97.907': 43,
        '101.07': 44, '102.9055': 45, '106.42': 46, '107.8682': 47, '112.411': 48, '114.818': 49,
        '118.71': 50, '121.76': 51, '127.6': 52, '126.90447': 53, '131.293': 54, '132.9': 55,
        '137.327': 56, #'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            ## 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            ## 'Er': 68, 'Tm': 69, 'Yb': 70, 
            ## 'Lu': 71, 
        '178.49': 72, '180.94788': 73,
        '183.84': 74, '186.207': 75, '190.23': 76, '192.217': 77, '195.084': 78, '196.9666': 79,
        '200.59': 80, '204.382': 81, '207.2': 82, '208.98': 83, 
            ## 'Po': 84, #'At': 85,
            ## 'Rn': 86, #'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
            ## 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
            ## 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            ## 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
            ## 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
            ## 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
        }#dict , modification needed
TYPE_NAME ={v:k for k, v in NAME_TYPE.items()}
TYPE_WEIGHT ={v:k for k, v in WEIGHT_TYPE.items()}

def make_scf_abacus(systems_train, systems_test=None, *,
             train_dump="data_train", test_dump="data_test", cleanup=None, 
             dispatcher=None, resources =None, 
             no_model=False, workdir='00.scf', share_folder='share', source_model=None,
             orb_file=[], pp_file=[], pbasis_file=[], 
             **scf_abacus):
    orb_file=[os.path.abspath(s) for s in flat_file_list(orb_file)]
    pp_file=[os.path.abspath(s) for s in flat_file_list(pp_file)]
    pbasis_file=[os.path.abspath(s) for s in flat_file_list(pbasis_file)]
    pre_scf_abacus = make_convert_scf_abacus(
            systems_train=systems_train, systems_test=systems_test,
            no_model=no_model, workdir=SCF_STEP_DIR, share_folder=share_folder, 
            source_model=source_model, 
            orb_file=orb_file, pp_file=pp_file, pbasis_file=pbasis_file, **scf_abacus)
    run_scf_abacus = make_run_scf_abacus(systems_train, systems_test,
        train_dump=train_dump, test_dump=test_dump, 
        no_model=no_model, group_data=False,
        workdir=SCF_STEP_DIR, outlog="log.scf", share_folder=share_folder, 
        dispatcher=dispatcher, resources=resources, **scf_abacus)
    """
    post_scf_abacus = make_stat_scf_abacus(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=train_dump, test_dump=test_dump, workdir=".", 
        outlog="log.data", group_data=False
    )
    """
    """
    #share orb_file and pp_file
        for i in range (len(orb_file)):
            orb_file[i] = check_share_folder(orb_file[i], orb_file[i], share_folder)
        for i in range (len(pp_file)):
            pp_file[i] = check_share_folder(pp_file[i], pp_file[i], share_folder)
        #share the traced model file
    """
    # concat
    #seq = [pre_scf_abacus, run_scf_abacus, post_scf_abacus]
    seq = [pre_scf_abacus, run_scf_abacus]
    #seq = [pre_scf_abacus]
    if cleanup:
        clean_scf = make_cleanup(
            ["slurm-*.out", "task.*/err", "fin.record"],
            workdir=".")
        seq.append(clean_scf)
    #make sequence
    return Sequence(
        seq,
        workdir=workdir
    )
### need parameters: orb_file, pp_file, proj_file
def convert_data(systems_train, systems_test=None, *, 
            source_model=None, pbasis_file=[], 
            orb_file=[], pp_file=[], lattice_constant=10, lattice_vector=["1 0 0", "0 1 0", "0 0 1"], 
            nbands=0, ecutwfc=50, conv_tol="1e-6", 
            max_scf_iter=50, gamma_only=1, proj_lmax=0, 
            no_model=True, isforce=0, abacus_path="/usr/local/bin/ABACUS.mpi",
            run_cmd="mpirun", cpus_per_task=1, **pre_args):
    #trace a model (if necessary)
    if not no_model:
        if source_model is not None:
            from deepks.model import CorrNet
            model = CorrNet.load(MODEL_FILE)
            model.compile_save(CMODEL_FILE)
    # split systems into groups
    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    #ntask_trn = int(np.ceil(nsys_trn / sub_size))
    #ntask_tst = int(np.ceil(nsys_tst / sub_size))
    train_sets = [systems_train[i::nsys_trn] for i in range(nsys_trn)]
    test_sets = [systems_test[i::nsys_tst] for i in range(nsys_tst)]
    systems=systems_train+systems_test
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    #create a shell script to run ABACUS
    if not os.path.isfile("./run_abacus.sh"):
        from pathlib import Path
        Path("./run_abacus.sh").touch()
    run_file=open("./run_abacus.sh","w")
    for i, sset in enumerate(train_sets+test_sets):
        atom_data = np.load(f"{sys_paths[i]}/atom.npy")
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]
        atoms = atom_data[1,:,0]
        types = np.unique(atoms) #index in type list
        ntype = types.size
        from collections import Counter
        nta = Counter(atoms) #dict {itype: nta}, natom in each type
        lat0=lattice_constant
        lat1=lattice_vector[0]
        lat2=lattice_vector[1]
        lat3=lattice_vector[2]
        if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
            os.mkdir(f"{sys_paths[i]}/ABACUS")
        for f in range(nframes):
            if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
                os.mkdir(f"{sys_paths[i]}/ABACUS/{f}")
            ###create STRU file
            if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/STRU"):
                from pathlib import Path
                Path(f"{sys_paths[i]}/ABACUS/{f}/STRU").touch()
            #write STRU file
            stru_file = open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w")
            stru_file.write("ATOMIC_SPECIES"+"\n")
            for it in range(ntype):
                stru_file.write(f"{TYPE_NAME[int(types[it])]} {TYPE_WEIGHT[int(types[it])]} {pp_file[it]}"+"\n")
            stru_file.write("\n"+"NUMERICAL_ORBITAL"+"\n")
            for it in range(ntype):
                stru_file.write(orb_file[it]+"\n")
            stru_file.write("\n"+"NUMERICAL_DESCRIPTOR"+"\n")
            stru_file.write(pbasis_file[0]+"\n")
            stru_file.write("\n"+"LATTICE_CONSTANT"+"\n")
            stru_file.write(str(lat0)+"\n")
            stru_file.write("\n"+"LATTICE_VECTORS"+"\n")
            stru_file.write(lat1+"\n")
            stru_file.write(lat2+"\n")
            stru_file.write(lat3+"\n")
            stru_file.write("\n"+"ATOMIC_POSITIONS"+"\n")
            stru_file.write("Cartesian"+"\n"+"\n")
            #stru_file.write("Direct")
            iat=0
            for it in range (ntype):
                stru_file.write(TYPE_NAME[int(types[it])]+"\n")
                stru_file.write("0"+"\n")
                stru_file.write(str(nta[types[it]])+"\n")
                for ia in range (nta[types[it]]):
                    stru_file.write(f"{atom_data[f, iat, 1]} {atom_data[f, iat, 2]} {atom_data[f, iat, 3]} 1 1 1"+"\n")
                    iat=iat+1
            stru_file.close()
            ### end for STRU file
            # create INPUT file
            if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/INPUT"):
                from pathlib import Path
                Path(f"{sys_paths[i]}/ABACUS/{f}/INPUT").touch()
            #write INPUT file
            input_file = open(f"{sys_paths[i]}/ABACUS/{f}/INPUT","w")
            input_file.write("INPUT_PARAMETERS"+"\n")
            input_file.write("suffix  abacus"+"\n")
            input_file.write("calculation  scf"+"\n")
            input_file.write(f"ntype  {ntype}"+"\n")
            input_file.write(f"nbands  {nbands}"+"\n")
            input_file.write("symmetry  0"+"\n")
            input_file.write(f"ecutwfc  {ecutwfc}"+"\n")
            input_file.write(f"dr2  {conv_tol}"+"\n")
            input_file.write(f"niter  {max_scf_iter}"+"\n")
            input_file.write("basis_type  lcao"+"\n")
            input_file.write(f"gamma_only  {gamma_only}"+"\n")
            input_file.write("smearing  gaussian"+"\n")
            input_file.write("sigma  0.02"+"\n")
            input_file.write("mixing_type  pulay"+"\n")
            input_file.write("mixing_beta  0.4"+"\n")
            input_file.write("\n"+"#DeePKS args"+"\n")
            input_file.write("out_descriptor  1"+"\n")
            input_file.write(f"lmax_descriptor  {proj_lmax}"+"\n")
            input_file.write(f"deepks_scf  {1-int(no_model)}"+"\n")
            input_file.write(f"force {isforce}"+"\n")
            input_file.close()
            ###end for INPUT file
            ###create KPT file 
            if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/KPT"):
                from pathlib import Path
                Path(f"{sys_paths[i]}/ABACUS/{f}/KPT").touch()
            #write KPT file (gamma_only)
            if gamma_only==1:
                kpt_file=open(f"{sys_paths[i]}/ABACUS/{f}/KPT","w")
                kpt_file.write("K_POINTS"+"\n")
                kpt_file.write("0"+"\n")
                kpt_file.write("Gamma"+"\n")
                kpt_file.write("1 1 1 0 0 0"+"\n")
                kpt_file.close()
            ###end for KPT file
        run_file.write(f"cd {sys_paths[i]}/ABACUS"+ "\n")
        run_file.write("i=0"+"\n")
        run_file.write(f"while (( $i < {nframes} ))"+ "\n")
        run_file.write("do"+ "\n")
        run_file.write("\t"+"cd ${i}"+ "\n")
        run_file.write("\t"+f"{run_cmd} -n {cpus_per_task} {abacus_path}"+ "\n")
        run_file.write("\t"+"cd .."+ "\n")
        run_file.write("\t"+"let \"i++\""+ "\n")
        run_file.write("done"+ "\n")
    run_file.close()
    ###end for run_file
    
def make_convert_scf_abacus(systems_train, systems_test=None, *,
                no_model=False, workdir='.', source_model=None, 
                pbasis_file=[],  orb_file=[], pp_file=[],
                lattice_constant = 10, nbands=0, ecutwfc=50, conv_tol="1e-6", 
                max_scf_iter=50, gamma_only=1, proj_lmax=0, 
                isforce=0, run_cmd="mpirun", abacus_path="/usr/local/bin/ABACUS.mpi", 
                cpus_per_task = 1, **pre_args):
    # if no test systems, use last one in train systems
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    check_system_names(systems_train)
    check_system_names(systems_test)
    #update pre_args
    pre_args.update(
        systems_train=systems_train, 
        systems_test=systems_test,
        pbasis_file=pbasis_file,
        orb_file=orb_file,
        pp_file=pp_file, 
        source_model=source_model, 
        lattice_constant=lattice_constant,
        nbands=nbands, 
        ecutwfc=ecutwfc, 
        conv_tol=conv_tol, 
        max_scf_iter=max_scf_iter,
        gamma_only=gamma_only,
        proj_lmax=proj_lmax, 
        no_model=no_model, 
        isforce=isforce, 
        run_cmd=run_cmd, 
        abacus_path=abacus_path, 
       cpus_per_task=cpus_per_task,
       **pre_args)
    return PythonTask(
        convert_data,
        call_kwargs=pre_args,
        outlog="convert.log",
        errlog="err",
        workdir=workdir, 
        #link_share_files=link_share_files
    )

DEFAULT_SCF_SUB_RES_ABACUS = {
    "numb_node": 1,
    "task_per_node": 1,
    "cpus_per_task": 1, 
    "exclusive": True
}
ABACUS_CMD="bash run_abacus.sh"
def make_run_scf_abacus(systems_train, systems_test=None, 
    train_dump="data_train", test_dump="data_test", resources=None, 
    dispatcher=None, share_folder="share", workdir=".", outlog="out.log", 
    **task_args):
    #cmd
    command = ABACUS_CMD
    #basic args
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    link_abs = task_args.pop("link_abs_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])
    #get systems
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    check_system_names(systems_train)
    check_system_names(systems_test)
    systems=systems_train+systems_test
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    sys_base = [get_sys_name(s) for s in sys_paths]
    sys_name = [os.path.basename(s) for s in sys_base]
    #prepare backward(download) files
    if train_dump:
        if sys_name:
            for nm in sys_name:
                backward_files.append(os.path.join(train_dump, nm))
        else:  # backward whole folder, may cause problem
            backward_files.append(train_dump)
    if test_dump:
        if sys_name:
            for nm in sys_name:
                backward_files.append(os.path.join(test_dump, nm))
    #make task
    return BatchTask(
        command, 
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        link_abs_files=link_abs,
        forward_files=forward_files,
        backward_files=backward_files,
    )
"""
def make_stat_scf_abacus(systems_train, systems_test=None, *, 
                  train_dump="data_train", test_dump="data_test", group_data=False,
                  workdir='.', outlog="log.data", **stat_args):
    #concatenate .npy result
    #copy them into shared folder
    return 

ABACUS_CMD= " ".join([
    "{run_cmd} -n 1",
    "{abacus_bin_dir} "
])

#single sub task
def make_scf_task_abacus(*, workdir=".",
        abacus_bin_dir = "/usr/local/bin/ABACUS.mpi", run_cmd = "mpirun", 
        dispatcher=None, resources=None, outlog="log.scf",
        share_folder="share", **task_args):
    #set up basic args
    command = ABACUS_CMD.format(run_cmd=mpirun, abacus_bin_dir=abacus_bin_dir)
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    link_abs = task_args.pop("link_abs_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])
    sys_name = None
    return BatchTask(
        command,
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        link_abs_files=link_abs,
        forward_files=forward_files,
        backward_files=backward_files,
        **task_args
    )
"""