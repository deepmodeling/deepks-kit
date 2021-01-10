from copy import deepcopy
from deepks.utils import check_list
from deepks.utils import get_abs_path
from deepks.task.task import AbstructStep


__all__ = ["Workflow", "Sequence", "Iteration"]


class Workflow(AbstructStep):
    def __init__(self, child_tasks, workdir='.', record_file=None):
        super().__init__(workdir)
        self.record_file = get_abs_path(record_file)
        self.child_tasks = [self.make_child(task) for task in child_tasks]
        self.postmod_hook()
        # self.set_record_file(record_file)
    
    def make_child(self, task):
        if not isinstance(task, AbstructStep):
            raise TypeError("Workflow only accept tasks and other task as childs, "
                            "but got " + type(task).__name__)
        assert not task.workdir.is_absolute()
        copied = deepcopy(task)
        copied.prepend_workdir(self.workdir)
        if isinstance(task, Workflow):
            copied.set_record_file(self.record_file)
        return copied
    
    def postmod_hook(self):
        pass
        
    def run(self, parent_tag=(), restart_tag=None):
        start_idx = 0
        if restart_tag is not None:
            last_idx = restart_tag[0]
            rest_tag = restart_tag[1:]
            if last_idx >= len(self.child_tasks):
                print(f'# restart tag {last_idx} out of range, stop now')
                return
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
        if not any(isinstance(task, Workflow) for task in self.child_tasks):
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

    def __setitem__(self, idx, task):
        self.child_tasks[idx] = self.make_child(task)
        self.postmod_hook()

    def __delitem__(self, idx):
        self.child_tasks.__delitem__(idx)
        self.postmod_hook()
    
    def __len__(self):
        return len(self.child_tasks)

    def __iter__(self):
        return self.child_tasks.__iter__()
    
    def insert(self, index, task):
        self.child_tasks.insert(index, self.make_child(task))
        self.postmod_hook()
    
    def append(self, task):
        self.child_tasks.append(self.make_child(task))
        self.postmod_hook()

    def prepend(self, task):
        self.child_tasks.insert(0, self.make_child(task))
        self.postmod_hook()


class Sequence(Workflow):
    def __init__(self, child_tasks, workdir='.', record_file=None, init_folder=None):
        # would reset all tasks' prev folder into their prev task, except for the first one
        super().__init__(child_tasks, workdir, record_file)
        if init_folder is not None:
            self.set_init_folder(init_folder)
        
    def chain_tasks(self):    
        for prev, curr in zip(self.child_tasks[:-1], self.child_tasks[1:]):
            while isinstance(prev, Workflow):
                prev = prev.child_tasks[-1]
            while isinstance(curr, Workflow):
                curr = curr.child_tasks[0]
            curr.set_prev_task(prev)
    
    def set_init_folder(self, init_folder):
        start = self.child_tasks[0]
        while isinstance(start, Workflow):
            start = start.child_tasks[0]
        start.set_prev_folder(get_abs_path(init_folder))

    def postmod_hook(self):
        self.chain_tasks()


class Iteration(Sequence):
    def __init__(self, task, iternum, workdir='.', record_file=None, init_folder=None):
        # iterated task should have workdir='.' to avoid redundant folders
        # handle multple tasks by first make a sequence
        if not isinstance(task, AbstructStep):
            task = Sequence(task)
        iter_tasks = [deepcopy(task) for i in range(iternum)]
        nd = max(len(str(iternum)), 2)
        for ii, itask in enumerate(iter_tasks):
            itask.prepend_workdir(f'iter.{ii:0>{nd}d}')
        super().__init__(iter_tasks, workdir, record_file, init_folder)

