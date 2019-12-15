import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path
from contextlib import nullcontext, redirect_stdout, redirect_stderr


def link_file(src, dst):
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        os.symlink(os.path.relpath(src, dst.parent), dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(os.path.relpath(src, dst.parent), dst)

def copy_file(src, dst):
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)

def check_file_list(files):
    if isinstance(files, str):
        return [files]
    if files is None:
        return []
    return files

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
    def __init__(self, workdir='.', prev_task=None,
                 prev_folder=None, link_prev_files=None, copy_prev_files=None, 
                 share_folder=None, link_share_files=None, copy_share_files=None):
        # workdir has to be relative in order to be chained
        # prev_task is dereferenced to folder dynamically.
        # folders are absolute.
        super().__init__(workdir)
        assert prev_task is None or prev_folder is None
        self.prev_task = prev_task
        self.prev_folder = get_abs_path(prev_folder)
        self.share_folder = get_abs_path(share_folder)
        self.link_prev_files = check_file_list(link_prev_files)
        self.copy_prev_files = check_file_list(copy_prev_files)
        self.link_share_files = check_file_list(link_share_files)
        self.copy_share_files = check_file_list(copy_share_files)
        
    def preprocess(self):
        if not self.workdir.exists():
            os.makedirs(self.workdir)
        else:
            assert self.workdir.is_dir(), f'{self.workdir} is not a dir'
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
        self.olddir = os.getcwd()
        os.chdir(self.workdir)
    
    def execute(self):
        raise NotImplementedError

    def postprocess(self):
        os.chdir(self.olddir)
        
    def run(self, *args, **kwargs):
        self.preprocess()
        self.execute()
        self.postprocess()
    
    def set_prev_task(self, task):
        assert isinstance(task, AbstructTask)
        self.prev_folder = None
        self.prev_task = task


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
        with (open(self.outlog, 'w') if self.outlog is not None 
                else nullcontext(sys.stdout)) as fo, \
             redirect_stdout(fo), \
             (open(self.errlog, 'w') if self.errlog is not None 
                else nullcontext(sys.stderr)) as fe, \
             redirect_stderr(fe):
            self.pycallable(*self.call_args, **self.call_kwargs)


class Workflow(AbstructStep):
    def __init__(self, child_tasks, workdir='.', record_file=None):
        super().__init__(workdir)
        self.child_tasks = [deepcopy(task) for task in child_tasks]
        self.record_file = get_abs_path(record_file)
        for task in self.child_tasks:
            assert not task.workdir.is_absolute()
            task.prepend_workdir(self.workdir)
            if isinstance(task, Workflow):
                task.record_file = self.record_file
        
    def run(self, parent_tag=(), restart_tag=None):
        start_idx = 0
        if restart_tag is not None:
            last_idx = restart_tag[0]
            rest_tag = restart_tag[1:]
            if rest_tag:
                last_tag = parent_tag+(last_idx,)
                self.child_tasks[last_idx].run(last_tag, rest_tag)
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
        with self.record_file.open() as lf:
            all_tags = [tuple(map(int, l.split())) for l in lf.readlines()]
        # assert max(map(len, all_tags)) == self.max_depth()
        restart_tag = all_tags[-1]
        print('# restart after step', restart_tag)
        self.run((), restart_tag)
        
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
            start.prev_folder = get_abs_path(init_folder)
        
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
        iter_tasks = [deepcopy(task) for i in range(iternum)]
        nd = len(str(iternum))
        for ii, itask in enumerate(iter_tasks):
            itask.prepend_workdir(f'iter.{ii:0>{nd}d}')
        super().__init__(iter_tasks, workdir, record_file, init_folder)

