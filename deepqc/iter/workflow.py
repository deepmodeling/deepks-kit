import os
import sys
import shutil
import subprocess as sp
from copy import deepcopy
from pathlib import Path
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from .job.dispatcher import Dispatcher


def link_file(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        os.symlink(os.path.relpath(src, dst.parent), dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(os.path.relpath(src, dst.parent), dst)

def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)

def create_dir(dirname, backup=False):
    dirname = Path(dirname)
    if not dirname.exists():
        os.makedirs(dirname)
    elif backup and dirname != Path('.'):
        os.makedirs(dirname.parent, exist_ok=True)
        counter = 0
        bckname = str(dirname) + f'.bck.{counter:03d}'
        while os.path.exists(bckname):
            counter += 1
            bckname = str(dirname) + f'.bck.{counter:03d}'
        dirname.rename(bckname)
        os.makedirs(dirname)
    else:
        assert dirname.is_dir(), f'{dirname} is not a dir'

def check_arg_list(args):
    if args is None:
        return []
    if not isinstance(args, list):
        return [args]
    return args

def get_abs_path(p):
    if p is None:
        return None
    else:
        return Path(p).absolute()


class AbstructStep(object):
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        
    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} with workdir: {self.workdir}>'
        
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def prepend_workdir(self, path):
        self.workdir = path / self.workdir
        
    def append_workdir(self, path):
        self.workdir = self.workdir / path


class AbstructTask(AbstructStep):
    def __init__(self, workdir='.', backup=False, prev_task=None,
                 prev_folder=None, link_prev_files=None, copy_prev_files=None, 
                 share_folder=None, link_share_files=None, copy_share_files=None):
        # workdir has to be relative in order to be chained
        # prev_task is dereferenced to folder dynamically.
        # folders are absolute.
        super().__init__(workdir)
        self.backup = backup
        assert prev_task is None or prev_folder is None
        self.prev_task = prev_task
        self.prev_folder = get_abs_path(prev_folder)
        self.share_folder = get_abs_path(share_folder)
        self.link_prev_files = check_arg_list(link_prev_files)
        self.copy_prev_files = check_arg_list(copy_prev_files)
        self.link_share_files = check_arg_list(link_share_files)
        self.copy_share_files = check_arg_list(copy_share_files)
        
    def preprocess(self):
        create_dir(self.workdir, self.backup)
        if self.prev_folder is None and (self.link_prev_files or self.copy_prev_files):
            self.prev_folder = self.prev_task.workdir
        for f in self.link_prev_files:
            link_file(self.prev_folder / f, self.workdir / f)
        for f in self.copy_prev_files:
            copy_file(self.prev_folder / f, self.workdir / f)
        for f in self.link_share_files:
            link_file(self.share_folder / f, self.workdir / f)
        for f in self.copy_share_files:
            copy_file(self.share_folder / f, self.workdir / f)
    
    def execute(self):
        raise NotImplementedError

    def postprocess(self):
        pass
        
    def run(self, *args, **kwargs):
        self.preprocess()
        self.olddir = os.getcwd()
        os.chdir(self.workdir)
        self.execute()
        os.chdir(self.olddir)
        self.postprocess()
    
    def set_prev_task(self, task):
        assert isinstance(task, AbstructTask)
        self.prev_folder = None
        self.prev_task = task

    def set_prev_folder(self, path):
        self.prev_task = None
        self.prev_folder = path


class BlankTask(AbstructTask):
    def execute(self):
        pass


class PythonTask(AbstructTask):
    def __init__(self, pycallable, 
                 call_args=None, call_kwargs=None, 
                 outlog=None, errlog=None,
                 **task_args):
        super().__init__(**task_args)
        self.pycallable = pycallable
        self.call_args = call_args if call_args is not None else []
        self.call_kwargs = call_kwargs if call_kwargs is not None else {}
        self.outlog = outlog
        self.errlog = errlog
    
    def execute(self):
        with (open(self.outlog, 'w', 1) if self.outlog is not None 
                else nullcontext(sys.stdout)) as fo, \
             redirect_stdout(fo), \
             (open(self.errlog, 'w', 1) if self.errlog is not None 
                else nullcontext(sys.stderr)) as fe, \
             redirect_stderr(fe):
            self.pycallable(*self.call_args, **self.call_kwargs)


class ShellTask(AbstructTask):
    def __init__(self, cmd, env=None,
                 outlog=None, errlog=None,
                 **task_args):
        super().__init__(**task_args)
        self.cmd = cmd
        self.env = env
        self.outlog = outlog
        self.errlog = errlog

    def execute(self):
        with (open(self.outlog, 'w', 1) if self.outlog is not None 
                else nullcontext()) as fo, \
             (open(self.errlog, 'w', 1) if self.errlog is not None 
                else nullcontext()) as fe:
            sp.run(self.cmd, env=self.env, shell=True, stdout=fo, stderr=fe)


class BatchTask(AbstructTask):
    def __init__(self, cmds, 
                 dispatcher=None, resources=None, 
                 outlog='log', errlog='err', 
                 forward_files=None, backward_files=None,
                 **task_args):
        super().__init__(**task_args)
        self.cmds = check_arg_list(cmds)
        if dispatcher is None:
            dispatcher = Dispatcher()
        elif isinstance(dispatcher, dict):
            dispatcher = Dispatcher(**dispatcher)
        assert isinstance(dispatcher, Dispatcher)
        self.dispatcher = dispatcher
        self.resources = resources
        self.outlog = outlog
        self.errlog = errlog
        self.forward_files = check_arg_list(forward_files)
        self.backward_files = check_arg_list(backward_files)
    
    def execute(self):
        tdict = self.make_dict(base=self.workdir)
        self.dispatcher.run_jobs([tdict], group_size=1, work_path='.', 
                                 resources=self.resources, forward_task_deref=True,
                                 outlog=self.outlog, errlog=self.errlog)

    def make_dict(self, base='.'):
        return {'dir': str(self.workdir.relative_to(base)),
                'cmds': self.cmds,
                'forward_files': self.forward_files,
                'backward_files': self.backward_files}


class GroupBatchTask(AbstructTask):
    # after grouping up, the following individual setting would be ignored:
    # dispatcher, resources, outlog, errlog
    # only grouped one setting in this task would be effective
    def __init__(self, batch_tasks, group_size=1,
                 dispatcher=None, resources=None, 
                 outlog='log', errlog='err', forward_common_files=None,
                 **task_args):
        super().__init__(**task_args)            
        self.batch_tasks = [deepcopy(task) for task in batch_tasks]
        for task in self.batch_tasks:
            assert isinstance(task, BatchTask), f'given task is instance of {task.__class__}'
            assert not task.workdir.is_absolute()
            task.prepend_workdir(self.workdir)
            if task.prev_folder is None:
                task.prev_folder = self.prev_folder
        self.group_size = group_size
        if dispatcher is None:
            dispatcher = Dispatcher()
        elif isinstance(dispatcher, dict):
            dispatcher = Dispatcher(**dispatcher)
        assert isinstance(dispatcher, Dispatcher)
        self.dispatcher = dispatcher
        self.resources = resources
        self.outlog = outlog
        self.errlog = errlog
        self.common_files = check_arg_list(forward_common_files)

    def execute(self):
        tdicts = [t.make_dict(base=self.workdir) for t in self.batch_tasks]
        self.dispatcher.run_jobs(tdicts, group_size=self.group_size, work_path='.', 
                                 resources=self.resources, forward_task_deref=False,
                                 forward_common_files=self.common_files,
                                 outlog=self.outlog, errlog=self.errlog)
    
    def preprocess(self):
        # if (self.workdir / 'fin.record').exists():
        #     return
        super().preprocess()
        for t in self.batch_tasks:
            t.preprocess()

    def prepend_workdir(self, path):
        super().prepend_workdir(path)
        for t in self.batch_tasks:
            t.prepend_workdir(path)
    
    def set_prev_task(self, task):
        super().set_prev_task(task)
        for t in self.batch_tasks:
            t.set_prev_task(task)
    
    def set_prev_folder(self, path):
        super().set_prev_folder(path)
        for t in self.batch_tasks:
            t.set_prev_folder(path)


class Workflow(AbstructStep):
    def __init__(self, child_tasks, workdir='.', record_file=None):
        super().__init__(workdir)
        self.child_tasks = [deepcopy(task) for task in child_tasks]
        for task in self.child_tasks:
            assert not task.workdir.is_absolute()
            task.prepend_workdir(self.workdir)
        self.set_record_file(record_file)
        
    def run(self, parent_tag=(), restart_tag=None):
        start_idx = 0
        if restart_tag is not None:
            last_idx = restart_tag[0]
            rest_tag = restart_tag[1:]
            if rest_tag:
                last_tag = parent_tag+(last_idx,)
                self.child_tasks[last_idx].run(last_tag, restart_tag=rest_tag)
                self.write_record(last_tag)
            start_idx = last_idx + 1
        for i in range(start_idx, len(self.child_tasks)):
            curr_tag = parent_tag + (i,)
            print('# starting step:', curr_tag) 
            task = self.child_tasks[i]
            task.run(curr_tag)
            self.write_record(curr_tag)
            
    def prepend_workdir(self, path):
        super().prepend_workdir(path)
        for task in self.child_tasks:
            task.prepend_workdir(path)

    def set_record_file(self, record_file):
        self.record_file = get_abs_path(record_file)
        for task in self.child_tasks:
            if isinstance(task, Workflow):
                task.set_record_file(record_file)

    def write_record(self, tag):
        if self.record_file is None:
            return
        if isinstance(tag, (list, tuple)):
            tag = ' '.join(map(str,tag))
        with self.record_file.open('a') as lf:
            lf.write(tag + '\n')

    def max_depth(self):
        if all(isinstance(task, AbstructTask) for task in self.child_tasks):
            return 1
        else:
            return 1 + max(task.max_depth() for task in self.child_tasks if isinstance(task, Workflow))
            
    def restart(self):
        if not self.record_file.exists():
            print('# no record file, starting from scratch')
            self.run(())
            return
        with self.record_file.open() as lf:
            all_tags = [tuple(map(int, l.split())) for l in lf.readlines()]
        # assert max(map(len, all_tags)) == self.max_depth()
        restart_tag = all_tags[-1]
        print('# restarting after step', restart_tag)
        self.run((), restart_tag=restart_tag)
        
    def __getitem__(self, idx):
        return self.child_tasks[idx]


class Sequence(Workflow):
    def __init__(self, child_tasks, workdir='.', record_file=None, init_folder=None):
        # would reset all tasks' prev folder into their prev task, except for the first one
        super().__init__(child_tasks, workdir, record_file)
        self.chain_tasks()
        start = self.child_tasks[0]
        while isinstance(start, Workflow):
            start = start.child_tasks[0]
        if start.prev_folder is None:
            start.set_prev_folder(get_abs_path(init_folder))
        
    def chain_tasks(self):    
        for prev, curr in zip(self.child_tasks[:-1], self.child_tasks[1:]):
            while isinstance(prev, Workflow):
                prev = prev.child_tasks[-1]
            while isinstance(curr, Workflow):
                curr = curr.child_tasks[0]
            curr.set_prev_task(prev)


class Iteration(Sequence):
    def __init__(self, task, iternum, workdir='.', record_file=None, init_folder=None):
        # iterated task should have workdir='.' to avoid redundant folders
        # handle multple tasks by first make a sequence
        if not isinstance(task, AbstructStep):
            task = Sequence(task)
        iter_tasks = [deepcopy(task) for i in range(iternum)]
        nd = len(str(iternum))
        for ii, itask in enumerate(iter_tasks):
            itask.prepend_workdir(f'iter.{ii:0>{nd}d}')
        super().__init__(iter_tasks, workdir, record_file, init_folder)

