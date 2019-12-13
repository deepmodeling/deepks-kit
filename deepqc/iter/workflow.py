import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path


def link_file(src, dst):
    assert src.exists()
    if not dst.exists():
        os.symlink(os.path.relpath(src, dst.parent), dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(os.path.relpath(src, dst.parent), dst)

def copy_file(src, dst):
    assert src.exists()
    if not dst.exists():
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)


class AbstructStep(object):
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        
    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} with workdir: {self.workdir}>'
    
    def preprocess(self):
        pass
        
    def execute(self, *args, **kwargs):
        raise NotImplementedError
        
    def postprocess(self):
        pass
        
    def run(self, *args, **kwargs):
        self.preprocess()
        self.execute(*args, **kwargs)
        self.postprocess()
    
    def prepend_workdir(self, path):
        self.workdir = path / self.workdir
        
    def append_workdir(self, path):
        self.workdir = self.workdir / path


class AbstructTask(AbstructStep):
    def __init__(self, workdir='.', prev_task=None,
                 prev_folder=None, require_prev_files=None, 
                 share_folder='.', require_share_files=None):
        # workdir has to be relative in order to be chained
        # prev_task is dereferenced to folder dynamically.
        # folders are absolute.
        super().__init__(workdir)
        assert prev_task is None or prev_folder is None
        self.prev_task = prev_task
        self.prev_folder = Path(prev_folder).absolute() if prev_folder is not None else None
        self.share_folder = Path(share_folder).absolute()

        if isinstance(require_prev_files, str):
            require_prev_files = [require_prev_files]
        if require_prev_files is None:
            require_prev_files = []
        self.prev_files = require_prev_files
        
        if isinstance(require_share_files, str):
            require_share_files = [require_share_files]
        if require_share_files is None:
            require_share_files = []
        self.share_files = require_share_files
        
    def preprocess(self):
        if not self.workdir.exists():
            os.makedirs(self.workdir)
        else:
            assert self.workdir.is_dir()
        if self.prev_folder is None:
            self.prev_folder = self.prev_task.workdir
        for f in self.prev_files:
            link_file(self.prev_folder / f, self.workdir / f)
        for f in self.share_files:
            link_file(self.share_folder / f, self.workdir / f)
        self.olddir = os.getcwd()
        os.chdir(self.workdir)
    
    def postprocess(self):
        os.chdir(self.olddir)
        
    def set_prev_task(self, task):
        assert isinstance(task, AbstructTask)
        self.prev_folder = None
        self.prev_task = task


class BlankTask(AbstructTask):
    def execute(self, *args, **kwargs):
        pass


class PythonTask(AbstructTask):
    def __init__(self, pycallable, call_args=None, call_kwargs=None, **task_args):
        super().__init__(**task_args)
        self.pycallable = pycallable
        self.call_args = call_args if call_args is not None else []
        self.call_kwargs = call_kwargs if call_kwargs is not None else {}
    
    def execute(self, *args, **kwargs):
        self.pycallable(*self.call_args, **self.call_kwargs)


class Workflow(AbstructStep):
    def __init__(self, child_tasks, workdir='.', logfile=None):
        super().__init__(workdir)
        self.child_tasks = [deepcopy(task) for task in child_tasks]
        self.logfile = Path(logfile).absolute() if logfile is not None else None
        for task in self.child_tasks:
            assert not task.workdir.is_absolute()
            task.prepend_workdir(self.workdir)
            if isinstance(task, Workflow):
                task.logfile = self.logfile
        
    def execute(self, parent_tag=(), restart_tag=None):
        start_idx = 0
        if restart_tag is not None:
            last_idx = restart_tag[0]
            rest_tag = restart_tag[1:]
            if rest_tag:
                last_tag = parent_tag+(last_idx,)
                self.child_tasks[last_idx].run(last_tag, rest_tag)
                self.write_log(last_tag)
            start_idx = last_idx + 1
        for i in range(start_idx, len(self.child_tasks)):
            curr_tag = parent_tag + (i,)
            print('# starting task:', curr_tag) 
            task = self.child_tasks[i]
            task.run(curr_tag)
            self.write_log(curr_tag)
            
    def prepend_workdir(self, path):
        super().prepend_workdir(path)
        for task in self.child_tasks:
            task.prepend_workdir(path)
            
    def write_log(self, tag):
        if self.logfile is None:
            return
        if isinstance(tag, (list, tuple)):
            tag = ' '.join(map(str,tag))
        with self.logfile.open('a') as lf:
            lf.write(tag + '\n')

    def max_depth(self):
        if all(isinstance(task, AbstructTask) for task in self.child_tasks):
            return 1
        else:
            return 1 + max(task.max_depth() for task in self.child_tasks if isinstance(task, Workflow))
            
    def restart(self):
        with self.logfile.open() as lf:
            all_tags = [tuple(map(int, l.split())) for l in lf.readlines()]
        assert max(map(len, all_tags)) == self.max_depth()
        restart_tag = all_tags[-1]
        print('# restart after task', restart_tag)
        self.run((), restart_tag)
        
    def __getitem__(self, idx):
        return self.child_tasks[idx]


class Sequence(Workflow):
    def __init__(self, child_tasks, workdir='.', logfile=None, init_folder='.'):
        super().__init__(child_tasks, workdir, logfile)
        self.chain_tasks()
        start = self.child_tasks[0]
        while isinstance(start, Workflow):
            start = start.child_tasks[0]
        if start.prev_folder is None:
            start.prev_folder = Path(init_folder)
        
    def chain_tasks(self):    
        for prev, curr in zip(self.child_tasks[:-1], self.child_tasks[1:]):
            while isinstance(prev, Workflow):
                prev = prev.child_tasks[-1]
            while isinstance(curr, Workflow):
                curr = curr.child_tasks[0]
            curr.set_prev_task(prev)


class Iteration(Sequence):
    def __init__(self, task, iternum, workdir='.', logfile=None, init_folder='.'):
        # iterated task should have workdir='.' to avoid redundant folders
        iter_tasks = [deepcopy(task) for i in range(iternum)]
        nd = len(str(iternum))
        for ii, itask in enumerate(iter_tasks):
            itask.prepend_workdir(f'iter.{ii:0>{nd}d}')
        super().__init__(iter_tasks, workdir, logfile, init_folder)

