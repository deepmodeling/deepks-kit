import os,sys,time,random

from .local_context import LocalSession
from .local_context import LocalContext
from .lazy_local_context import LazyLocalContext
from .ssh_context import SSHSession
from .ssh_context import SSHContext
from .slurm import Slurm
from .shell import Shell
from .job_status import JobStatus
from hashlib import sha1
# from monty.serialization import dumpfn,loadfn
from copy import deepcopy
import json


def _split_tasks(tasks,
                 group_size):
    ntasks = len(tasks)
    ngroups = ntasks // group_size
    if ngroups * group_size < ntasks:
        ngroups += 1
    chunks = [[]] * ngroups
    tot = 0
    for ii in range(ngroups) :
        chunks[ii] = (tasks[ii::ngroups])
        tot += len(chunks[ii])
    assert(tot == len(tasks))
    return chunks

    
class Dispatcher(object):
    def __init__ (self,
                  context_type='local',
                  batch_type='slurm',
                  remote_profile=None):
        if remote_profile is None:
            assert 'local' in context_type
            context_type = 'lazy-local'
        self.remote_profile = remote_profile
        if context_type == 'local':
            self.session = LocalSession(remote_profile)
            self.context_fn = LocalContext
            self.uuid_names = False
        elif context_type == 'lazy-local':
            self.session = None
            self.context_fn = LazyLocalContext
            self.uuid_names = True
        elif context_type == 'ssh':
            self.session = SSHSession(remote_profile)
            self.context_fn = SSHContext
            self.uuid_names = False
        else :
            raise RuntimeError('unknown context')
        if batch_type == 'slurm':
            self.batch_fn = Slurm            
        elif batch_type == 'shell':
            self.batch_fn = Shell
        else :
            raise RuntimeError('unknown batch ' + batch_type)

    # customize deepcopy to make sure copied instances share same session
    def __deepcopy__(self, memo):
        d = id(self)
        if d in memo:
            return memo[d]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[d] = result
        for k, v in self.__dict__.items():
            if k == "session":
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def run_jobs(self,
                 tasks,
                 group_size=1,
                 para_deg=1,
                 work_path='.',
                 resources=None,
                 forward_task_deref=True,
                 forward_common_files=[],
                 backward_common_files=[],
                 outlog='log',
                 errlog='err') :
        # tasks is a list of dict [t1, t2, t3, ...]
        # with each element t = {'dir': job_dir, 
        #                        'cmds': list of cmds, 
        #                        'forward_files': list of files to be forward,
        #                        'backward_files': list of files to be pull back}
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            assert isinstance(task, dict)
            if isinstance(task['cmds'], str):
                task['cmds'] = [task['cmds']]
            for field in ('forward_files', 'backward_files'):
                if task.get(field) is None:
                    task[field] = []
            task['_label'] = f'{{dir:{task["dir"]}, cmds:{task["cmds"]}}}'

        task_chunks = _split_tasks(tasks, group_size)    
        _pmap=PMap(work_path)
        path_map=_pmap.load()
        _fr = FinRecord(work_path, len(task_chunks))        

        job_list = []
        task_chunks_=['+'.join(t['_label'] for t in chunk) for chunk in task_chunks]
        job_fin = _fr.get_record()
        assert(len(job_fin) == len(task_chunks))
        for ii,chunk in enumerate(task_chunks) :
            if not job_fin[ii] :
                # map chunk info. to uniq id    
                chunk_sha1 = sha1(task_chunks_[ii].encode('utf-8')).hexdigest() 
                # if hash in map, recover job, else start a new job
                if chunk_sha1 in path_map:
                    # job_uuid = path_map[chunk_sha1][1].split('/')[-1]
                    job_uuid = path_map[chunk_sha1][2]
                    # dlog.debug("load uuid %s for chunk %s" % (job_uuid, task_chunks_[ii]))
                else:
                    job_uuid = None
                # communication context, bach system
                context = self.context_fn(work_path, self.session, job_uuid)
                batch = self.batch_fn(context, uuid_names = self.uuid_names)
                rjob = {'context':context, 'batch':batch}
                # upload files
                if (not isinstance(rjob['context'], LazyLocalContext) and
                    not rjob['context'].check_file_exists('tag_upload')):
                    rjob['context'].upload('.',
                                           forward_common_files)
                    for task in chunk:
                        rjob['context'].upload([task['dir']],
                                               task['forward_files'], 
                                               dereference = forward_task_deref)
                    rjob['context'].write_file('tag_upload', '')
                    # dlog.debug('uploaded files for %s' % task_chunks_[ii])
                # submit new or recover old submission
                dirs = [task['dir'] for task in chunk]
                commands = [task['cmds'] for task in chunk]
                para_res = [task['resources'] for task in chunk]
                if job_uuid is None:
                    rjob['batch'].submit(dirs, commands, res = resources, 
                                         outlog=outlog, errlog=errlog, 
                                         para_deg=para_deg, para_res=para_res)
                    job_uuid = rjob['context'].job_uuid
                    # dlog.debug('assigned uudi %s for %s ' % (job_uuid, task_chunks_[ii]))
                    print('# new submission of %s' % job_uuid)
                else:
                    rjob['batch'].submit(dirs, commands, res = resources, 
                                         outlog=outlog, errlog=errlog, 
                                         para_deg=para_deg, para_res=para_res,
                                         restart = True)
                    print('# restart from old submission %s ' % job_uuid)
                # record job and its hash
                job_list.append(rjob)
                path_map[chunk_sha1] = [context.local_root, context.remote_root, job_uuid]
            else :
                # finished job, append a None to list
                job_list.append(None)
        _pmap.dump(path_map)

        assert(len(job_list) == len(task_chunks))
        fcount = [0]*len(job_list)
        while not all(job_fin) :
            # dlog.debug('checking jobs')
            for idx,rjob in enumerate(job_list) :
                chunk = task_chunks[idx]
                if not job_fin[idx] :
                    status = rjob['batch'].check_status()
                    job_uuid = rjob['context'].job_uuid
                    if status == JobStatus.terminated :
                        fcount[idx] += 1
                        if fcount[idx] > 3:
                            raise RuntimeError('Job %s failed for more than 3 times' % job_uuid)
                        print('# job %s terminated, submit again'% job_uuid)
                        # dlog.debug('try %s times for %s'% (fcount[idx], job_uuid))
                        dirs = [task['dir'] for task in chunk]
                        commands = [task['cmds'] for task in chunk]
                        para_res = [task['resources'] for task in chunk]
                        rjob['batch'].submit(dirs, commands, res = resources, 
                                             outlog=outlog, errlog=errlog, 
                                             para_deg=para_deg, para_res=para_res,
                                             restart = True)
                    elif status == JobStatus.finished :
                        print('# job %s finished' % job_uuid)
                        rjob['context'].download('.', backward_common_files)
                        for task in chunk:
                            rjob['context'].download([task['dir']], task['backward_files'])
                        rjob['context'].clean()
                        job_fin[idx] = True
                        _fr.write_record(job_fin)
            time.sleep(60)
        # delete path map file when job finish
        _pmap.delete()


class FinRecord(object):
    def __init__ (self, path, njobs, fname = 'fin.record'):
        self.path = os.path.abspath(path)
        self.fname = os.path.join(self.path, fname)
        self.njobs = njobs

    def get_record(self):
        if not os.path.exists(self.fname):
            return [False] * self.njobs
        else :
            with open(self.fname) as fp:
                return [bool(int(ii)) for ii in fp.read().split()]

    def write_record(self, job_fin):
        with open(self.fname, 'w') as fp:
            for ii in job_fin:
                if ii:
                    fp.write('1 ')
                else:
                    fp.write('0 ')


class PMap(object):
    '''
    Path map class to operate {read,write,delte} the pmap.json file
    '''

    def __init__(self,path,fname="pmap.json"):
        self.f_path_map=os.path.join(path,fname)

    def load(self):
        f_path_map=self.f_path_map
        if os.path.isfile(f_path_map):
            # path_map=loadfn(f_path_map)
            with open(f_path_map, 'r') as fp:
                path_map = json.load(fp)
        else:
            path_map={}
        return path_map

    def dump(self,pmap,indent=4):
        f_path_map=self.f_path_map
        # dumpfn(pmap,f_path_map,indent=indent)
        with open(f_path_map, "w") as fp:
            json.dump(pmap, fp, indent=indent)

    def delete(self):
        f_path_map=self.f_path_map
        try:
            os.remove(f_path_map)
        except:
            pass
