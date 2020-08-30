import os
import sys
import numpy as np
from deepqc.utils import load_sys_paths, check_list
from deepqc.task.task import PythonTask, ShellTask
from deepqc.task.task import BatchTask, GroupBatchTask
from deepqc.task.workflow import Sequence
from deepqc.utils import QCDIR


SCF_CMD = " ".join([
    "{python} -u",
    os.path.join(QCDIR, "scf/main.py")
])

DEFAULT_SCF_RES = {
    "time_limit": "24:00:00",
    "cpus_per_task": 8,
    "mem_limit": 8,
    "envs": {
        "PYSCF_MAX_MEMORY": 8000
    }
}

TRN_CMD = " ".join([
    "{python} -u",
    os.path.join(QCDIR, "train/main.py")
])

DEFAULT_TRN_RES = {
    "time_limit": "24:00:00",
    "mem_limit": 8,
    "numb_gpu": 1
}


def make_cleanup(pattern="slurm-*.out", workdir=".", **task_args):
    pattern = check_list(pattern)
    pattern = " ".join(pattern)
    assert pattern
    return ShellTask(
        f"rm -r {pattern}",
        workdir=workdir,
        **task_args
    )


def make_scf_task(*, workdir=".",
                  arg_file="scf_input.yaml", source_arg=None,
                  model_file="model.pth", source_model=None,
                  systems="systems.raw", dump_dir="results", 
                  share_folder="share", outlog="log.scf", group_data=None,
                  dispatcher=None, resources=None, 
                  python="python", **task_args):
    # set up basic args
    command = SCF_CMD.format(python=python)
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    #set up optional args
    if arg_file:
        command += f" {arg_file}"
        if source_arg is not None:
            link_share.append((source_arg, arg_file))
    if model_file:
        command += f" -m {model_file}"
        if source_model is not None:
            link_prev.append((source_model, model_file))
    if systems:
        if not isinstance(systems, str):
            systems = " ".join(check_list(systems, nullable=False))
        command += f" -s {systems}"
    if dump_dir:
        command += f" -d {dump_dir}"
    if group_data is not None:
        command += " -G" if group_data else " -NG"
    if resources is None:
        resources = {}
    resources = {**DEFAULT_SCF_RES, **resources}
    # make task
    return BatchTask(
        command, 
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        **task_args
    )


def make_run_scf(systems_train, systems_test=None, *,
                 train_dump="data_train", test_dump="data_test", 
                 no_model=False, group_data=None,
                 workdir='.', share_folder='share', outlog="log.scf",
                 source_arg="scf_input.yaml", source_model="model.pth",
                 dispatcher=None, resources=None, 
                 sub_size=1, group_size=1, ingroup_parallel=1, 
                 sub_res=None, python='python', **task_args):
    # if no test systems, use last one in train systems
    systems_train = load_sys_paths(systems_train)
    systems_test = load_sys_paths(systems_test)
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    # split systems into groups
    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    ntask_trn = int(np.ceil(nsys_trn / sub_size))
    ntask_tst = int(np.ceil(nsys_tst / sub_size))
    train_sets = [systems_train[i::ntask_trn] for i in range(ntask_trn)]
    test_sets = [systems_test[i::ntask_tst] for i in range(ntask_tst)]
    # make subtasks
    model_file = "../model.pth" if not no_model else "NONE"
    nd = max(len(str(ntask_trn+ntask_tst)), 2)
    trn_tasks = [
        make_scf_task(systems=sset, workdir=f"task.trn.{i:0{nd}}", 
                      arg_file="../scf_input.yaml", source_arg=None,
                      model_file=model_file, source_model=None,
                      dump_dir=f"../{train_dump}", group_data=group_data,
                      resources=sub_res, python=python)
        for i, sset in enumerate(train_sets)
    ]
    tst_tasks = [
        make_scf_task(systems=sset, workdir=f"task.tst.{i:0{nd}}", 
                      arg_file="../scf_input.yaml", source_arg=None,
                      model_file=model_file, source_model=None,
                      dump_dir=f"../{test_dump}", group_data=group_data, 
                      resources=sub_res, python=python)
        for i, sset in enumerate(test_sets)
    ]
    # set up optional args
    link_share = task_args.pop("link_share_files", [])
    link_share.append((source_arg, "scf_input.yaml"))
    link_prev = task_args.pop("link_prev_files", [])
    if not no_model:
        link_prev.append((source_model, "model.pth"))
    if resources is None:
        resources = {}
    resources = {**DEFAULT_SCF_RES, "numb_node": ingroup_parallel, **resources}
    # make task
    return GroupBatchTask(
        trn_tasks + tst_tasks,
        group_size=group_size,
        ingroup_parallel=ingroup_parallel,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev
    )


def make_stat_scf(systems_train, systems_test=None, *, 
                  train_dump="data_train", test_dump="data_test", group_data=False,
                  workdir='.', outlog="log.data", **stat_args):
    # follow same convention for systems as run_scf
    systems_train = load_sys_paths(systems_train)
    systems_test = load_sys_paths(systems_test)
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    # load stat function
    from deepqc.scf.tools import print_stat
    stat_args.update(
        systems=systems_train,
        test_sys=systems_test,
        dump_dir=train_dump,
        test_dump=test_dump,
        group=group_data)
    # make task
    return PythonTask(
        print_stat,
        call_kwargs=stat_args,
        outlog=outlog,
        workdir=workdir
    )


def make_scf(systems_train, systems_test=None, *,
             train_dump="data_train", test_dump="data_test",
             no_model=False, workdir='00.scf', share_folder='share',
             source_arg="scf_input.yaml", source_model="model.pth",
             dispatcher=None, resources=None, 
             sub_size=1, group_size=1, ingroup_parallel=1, 
             sub_res=None, python='python', 
             cleanup=False, **task_args):
    run_scf = make_run_scf(
        systems_train, systems_test,
        train_dump=train_dump, test_dump=test_dump, 
        no_model=no_model, group_data=False,
        workdir=".", outlog="log.scf", share_folder=share_folder, 
        source_arg=source_arg, source_model=source_model,
        dispatcher=dispatcher, resources=resources, 
        group_size=group_size, ingroup_parallel=ingroup_parallel,
        sub_size=sub_size, sub_res=sub_res, python=python, **task_args
    )
    post_scf = make_stat_scf(
        systems_train=systems_train, systems_test=systems_test,
        train_dump=train_dump, test_dump=test_dump, workdir=".", 
        outlog="log.data", group_data=False
    )
    if cleanup:
        clean_scf = make_cleanup(
            ["slurm-*.out", "task.*/err", "fin.record"],
            workdir=".")
    # concat
    seq = [run_scf, post_scf]
    if cleanup:
        seq.append(clean_scf)
    # make sequence
    return Sequence(
        seq,
        workdir=workdir
    )


def make_train_task(*, workdir=".",
                    arg_file="train_input.yaml", source_arg=None,
                    restart=None, source_model=None, 
                    save_model="model.pth", group_data=False,
                    data_train="data_train", source_train=None,
                    data_test="data_test", source_test=None,
                    share_folder="share", outlog="log.train",
                    dispatcher=None, resources=None, 
                    python="python", **task_args):
    # set up basic args
    command = TRN_CMD.format(python=python)
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    # set up optional args
    if arg_file:
        command += f" {arg_file}"
        if source_arg is not None:
            link_share.append((source_arg, arg_file))
    if restart:
        command += f" -r {restart}"
        if source_model is not None:
            link_prev.append((source_model, restart))
    if data_train:
        command += f" -d {data_train}" + ("" if group_data else "/*")
        if source_train is not None:
            link_prev.append((source_train, data_train))
    if data_test:
        command += f" -t {data_test}" + ("" if group_data else "/*")
        if source_test is not None:
            link_prev.append((source_test, data_test))
    if save_model:
        command += f" -o {save_model}"
    if resources is None:
        resources = {}
    resources = {**DEFAULT_TRN_RES, **resources}
    # make task
    return BatchTask(
        command,
        workdir=workdir,
        dispatcher=dispatcher,
        resources=resources,
        outlog=outlog,
        share_folder=share_folder,
        link_share_files=link_share,
        link_prev_files=link_prev,
        **task_args
    )


def make_run_train(source_train="data_train", source_test="data_test", *,
                   restart=True, source_model="model.pth", 
                   save_model="model.pth", source_arg="train_input.yaml", 
                   workdir=".", share_folder="share", outlog="log.train",
                   dispatcher=None, resources=None, 
                   python="python", **task_args):
    # just add some presetted arguments of make_train_task
    # have not implement parrallel training for multiple models
    if restart:
        restart = "old_model.pth"
    return make_train_task(
        workdir=workdir, 
        arg_file="train_input.yaml", source_arg=source_arg,
        restart=restart, source_model=source_model, 
        save_model=save_model, group_data=False,
        data_train="data_train", source_train=source_train,
        data_test="data_test", source_test=source_test,
        share_folder=share_folder, outlog=outlog,
        dispatcher=dispatcher, resources=resources,
        python=python, **task_args
    )


def make_test_train(data_paths, model_file="model.pth", *,
                    output_prefix="test", group_results=True, 
                    workdir='.', outlog="log.test", **test_args):
    from deepqc.train.test import main as test_func
    test_args.update(
        data_paths=data_paths,
        model_file=model_file,
        output_prefix=output_prefix,
        group=group_results)
    # make task
    return PythonTask(
        test_func,
        call_kwargs=test_args,
        outlog=outlog,
        workdir=workdir
    )


def make_train(source_train="data_train", source_test="data_test", *,
               restart=True, source_model="model.pth", 
               save_model="model.pth", source_arg="train_input.yaml", 
               workdir="01.train", share_folder="share",
               dispatcher=None, resources=None, 
               python="python", cleanup=False, **task_args):
    run_train = make_run_train(
        source_train=source_train, source_test=source_test,
        restart=restart, source_model=source_model, save_model=save_model,
        source_arg=source_arg, workdir=".", share_folder=share_folder,
        outlog="log.train", dispatcher=dispatcher, resources=resources,
        python=python, **task_args
    )
    post_train = make_test_train(
        data_paths=["data_train/*","data_test/*"],
        model_file=save_model, output_prefix="test", group_results=True,
        workdir=".", outlog="log.test"
    )
    if cleanup:
        clean_train = make_cleanup(
            ["slurm-*.out", "err", "fin.record", "tag_*finished"],
            workdir=".")
    # concat
    seq = [run_train, post_train]
    if cleanup:
        seq.append(clean_train)
    # make sequence
    return Sequence(
        seq,
        workdir=workdir
    )