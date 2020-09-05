import os,sys,time,random,json,glob
from hashlib import sha1
from copy import deepcopy

from .local_context import LocalSession
from .local_context import LocalContext
from .lazy_local_context import LazyLocalContext
from .ssh_context import SSHSession
from .ssh_context import SSHContext
from .slurm import Slurm
from .shell import Shell
from .job_status import JobStatus


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


def _hash_task_chunk(task_chunk):
    task_chunk_str = '+'.join(t['_label'] for t in task_chunk)
    task_hash = sha1(task_chunk_str.encode('utf-8')).hexdigest()
    return task_hash

    
class Dispatcher(object):
    def __init__ (self,
                  context='local',
                  batch='slurm',
                  remote_profile=None,
                  job_record = 'jr.json'):
        if remote_profile is None:
            assert 'local' in context
            context = 'lazy-local'
        self.remote_profile = remote_profile
        if context == 'local':
            self.session = LocalSession(remote_profile)
            self.context_fn = LocalContext
            self.uuid_names = False
        elif context == 'lazy-local':
            self.session = None
            self.context_fn = LazyLocalContext
            self.uuid_names = True
        elif context == 'ssh':
            self.session = SSHSession(remote_profile)
            self.context_fn = SSHContext
            self.uuid_names = False
        else :
            raise RuntimeError('unknown context')
        if batch == 'slurm':
            self.batch_fn = Slurm            
        elif batch == 'shell':
            self.batch_fn = Shell
        else :
            raise RuntimeError('unknown batch ' + batch)
        self.jrname = job_record

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
                 mark_failure = False,
                 outlog='log',
                 errlog='err') :
        # tasks is a list of dict [t1, t2, t3, ...]
        # with each element t = {
        #     'dir': job_dir, 
        #     'cmds': list of cmds, 
        #     'forward_files': list of files to be forward,
        #     'backward_files': list of files to be pull back,
        #     'resources': dict of resources used for the substep, when para_deg > 1
        # }
        job_handler = self.submit_jobs(
            tasks,
            group_size,
            para_deg,
            work_path,
            resources,
            forward_task_deref,
            forward_common_files,
            backward_common_files,
            outlog,
            errlog
        )
        while not self.all_finished(job_handler, mark_failure) :
            time.sleep(60)
        # delete path map file when job finish
        # _pmap.delete()


    def submit_jobs(self,
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
        task_hashes = [_hash_task_chunk(chunk) for chunk in task_chunks]
        job_record = JobRecord(work_path, task_chunks, fname = self.jrname)
        job_record.dump()
        nchunks = len(task_chunks)
        
        job_list = []
        for ii in range(nchunks) :            
            cur_chunk = task_chunks[ii]
            cur_hash = task_hashes[ii]
            if not job_record.check_finished(cur_hash):                
                # chunk is not finished
                # check if chunk is submitted
                submitted = job_record.check_submitted(cur_hash)
                if not submitted:
                    job_uuid = None
                else :
                    job_uuid = job_record.get_uuid(cur_hash)
                    # dlog.debug("load uuid %s for chunk %s" % (job_uuid, cur_hash))
                # communication context, bach system
                context = self.context_fn(work_path, self.session, job_uuid)
                batch = self.batch_fn(context, uuid_names = self.uuid_names)
                rjob = {'context':context, 'batch':batch}
                # upload files
                if (not isinstance(rjob['context'], LazyLocalContext) and
                    not rjob['context'].check_file_exists(rjob['batch'].upload_tag_name)):
                    rjob['context'].upload('.',
                                           forward_common_files)
                    for task in cur_chunk:
                        rjob['context'].upload([task['dir']],
                                               task['forward_files'], 
                                               dereference = forward_task_deref)
                    rjob['context'].write_file(rjob['batch'].upload_tag_name, '')
                # submit new or recover old submission
                dirs = [task['dir'] for task in cur_chunk]
                commands = [task['cmds'] for task in cur_chunk]
                para_res = [task['resources'] for task in cur_chunk]
                if not submitted:
                    rjob['batch'].submit(dirs, commands, res = resources, 
                                         outlog=outlog, errlog=errlog, 
                                         para_deg=para_deg, para_res=para_res)
                    job_uuid = rjob['context'].job_uuid
                    # dlog.debug('assigned uuid %s for %s ' % (job_uuid, task_chunks_str[ii]))
                    print('# new submission of %s for chunk %s' % (job_uuid, cur_hash))
                else:
                    rjob['batch'].submit(dirs, commands, res = resources, 
                                         outlog=outlog, errlog=errlog, 
                                         para_deg=para_deg, para_res=para_res,
                                         restart = True)
                    print('# restart from old submission %s for chunk %s' % (job_uuid, cur_hash))
                # record job and its remote context
                job_list.append(rjob)
                ip = None
                instance_id = None
                if (self.remote_profile is not None and
                    'cloud_resources' in self.remote_profile):
                    ip = self.remote_profile['hostname']
                    instance_id = self.remote_profile['instance_id']
                job_record.record_remote_context(cur_hash,                                                 
                                                 context.local_root, 
                                                 context.remote_root, 
                                                 job_uuid,
                                                 ip,
                                                 instance_id)
                job_record.dump()
            else :
                # finished job, append a None to list
                job_list.append(None)
        assert(len(job_list) == nchunks)
        job_handler = {
            'task_chunks': task_chunks,
            'job_list': job_list,
            'job_record': job_record,
            'resources': resources,
            'para_deg': para_deg,
            'outlog': outlog,
            'errlog': errlog,
            'backward_common_files': backward_common_files
        }
        return job_handler


    def all_finished(self, 
                     job_handler, 
                     mark_failure,
                     clean=True):
        task_chunks = job_handler['task_chunks']
        task_hashes = [_hash_task_chunk(chunk) for chunk in task_chunks]
        job_list = job_handler['job_list']
        job_record = job_handler['job_record']
        tag_failure_list = ['tag_failure_%d' % ii 
            for ii in range(max(len(t['cmds']) for c in task_chunks for t in c))]
        resources = job_handler['resources']
        para_deg = job_handler['para_deg']
        outlog = job_handler['outlog']
        errlog = job_handler['errlog']
        backward_common_files = job_handler['backward_common_files']
        # dlog.debug('checking jobs')
        nchunks = len(task_chunks)
        for idx in range(nchunks) :
            cur_chunk = task_chunks[idx]
            cur_hash = task_hashes[idx]
            rjob = job_list[idx]
            if not job_record.check_finished(cur_hash) :
                # chunk not finished according to record
                status = rjob['batch'].check_status()
                job_uuid = rjob['context'].job_uuid
                # dlog.debug('checked job %s' % job_uuid)
                if status == JobStatus.terminated :
                    job_record.increase_nfail(cur_hash)
                    if job_record.check_nfail(cur_hash) > 3:
                        raise RuntimeError('Job %s failed for more than 3 times' % job_uuid)
                    print('# job %s terminated, submit again'% job_uuid)
                    # dlog.debug('try %s times for %s'% (job_record.check_nfail(cur_hash), job_uuid))
                    dirs = [task['dir'] for task in cur_chunk]
                    commands = [task['cmds'] for task in cur_chunk]
                    para_res = [task['resources'] for task in cur_chunk]
                    rjob['batch'].submit(dirs, commands, res = resources, 
                                         outlog=outlog, errlog=errlog, 
                                         para_deg=para_deg, para_res=para_res,
                                         restart = True)                
                elif status == JobStatus.finished :
                    print('# job %s finished' % job_uuid)
                    rjob['context'].download('.', backward_common_files)
                    for task in cur_chunk:
                        if mark_failure:
                            rjob['context'].download([task['dir']], tag_failure_list, 
                                check_exists = True, mark_failure = False)
                            rjob['context'].download([task['dir']], task['backward_files'], 
                                check_exists = True, mark_failure = True)
                        else:
                            rjob['context'].download([task['dir']], task['backward_files'])
                    if clean:
                        rjob['context'].clean()
                    job_record.record_finish(cur_hash)
                    job_record.dump()
        job_record.dump()
        return job_record.check_all_finished()


class JobRecord(object):
    def __init__ (self, path, task_chunks, fname = 'job_record.json', ip=None):
        self.path = os.path.abspath(path)
        self.fname = os.path.join(self.path, fname)
        self.task_chunks = task_chunks
        if not os.path.exists(self.fname):
            self._new_record()
        else :
            self.load()

    def check_submitted(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['context'] is not None

    def record_remote_context(self, 
                              chunk_hash, 
                              local_root, 
                              remote_root, 
                              job_uuid,
                              ip=None,
                              instance_id=None):
        self.valid_hash(chunk_hash)
        # self.record[chunk_hash]['context'] = [local_root, remote_root, job_uuid, ip, instance_id]
        self.record[chunk_hash]['context'] = {}
        self.record[chunk_hash]['context']['local_root'] = local_root
        self.record[chunk_hash]['context']['remote_root'] = remote_root
        self.record[chunk_hash]['context']['job_uuid'] = job_uuid
        self.record[chunk_hash]['context']['ip'] = ip
        self.record[chunk_hash]['context']['instance_id'] = instance_id

    def get_uuid(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['context']['job_uuid']

    def check_finished(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['finished']

    def check_all_finished(self):
        flist = [self.record[ii]['finished'] for ii in self.record]
        return all(flist)

    def record_finish(self, chunk_hash):
        self.valid_hash(chunk_hash)
        self.record[chunk_hash]['finished'] = True

    def check_nfail(self,chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['fail_count']

    def increase_nfail(self,chunk_hash):
        self.valid_hash(chunk_hash)
        self.record[chunk_hash]['fail_count'] += 1

    def valid_hash(self, chunk_hash):
        if chunk_hash not in self.record.keys():
            raise RuntimeError('chunk hash %s not in record, a invalid record may be used, please check file %s' % (chunk_hash, self.fname))

    def dump(self):
        with open(self.fname, 'w') as fp:
            json.dump(self.record, fp, indent=4)

    def load(self):
        with open(self.fname) as fp:
            self.record = json.load(fp)

    def _new_record(self):
        task_hashes = [_hash_task_chunk(chunk) for chunk in self.task_chunks]
        self.record = {}
        for ii,jj in zip(task_hashes, self.task_chunks):
            self.record[ii] = {
                'context': None,
                'finished': False,
                'fail_count': 0,
                'task_chunk': jj,
            }

