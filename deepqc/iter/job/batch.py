import os,sys,time
from .job_status import JobStatus


class Batch(object) :
    def __init__ (self,
                  context, 
                  uuid_names = False) :
        self.context = context
        if uuid_names:
            self.finish_tag_name = '%s_tag_finished' % self.context.job_uuid
            self.sub_script_name = '%s.sub' % self.context.job_uuid
            self.job_id_name = '%s_job_id' % self.context.job_uuid
        else:
            self.finish_tag_name = 'tag_finished'
            self.sub_script_name = 'run.sub'
            self.job_id_name = 'job_id'

    def check_status(self) :
        raise RuntimeError('abstract method check_status should be implemented by derived class')        
        
    def default_resources(self, res) :
        raise RuntimeError('abstract method sub_script_head should be implemented by derived class')        

    def sub_script_head(self, res) :
        raise RuntimeError('abstract method sub_script_head should be implemented by derived class')        

    def sub_script_cmd(self, cmd, arg, res):
        raise RuntimeError('abstract method sub_script_cmd should be implemented by derived class')        

    def do_submit(self,
                  job_dirs,
                  cmds,
                  args = None, 
                  res = None,
                  outlog = 'log',
                  errlog = 'err'):
        '''
        submit a single job, assuming that no job is running there.
        '''
        raise RuntimeError('abstract method check_status should be implemented by derived class')        

    def sub_script(self,
                   job_dirs,
                   cmds,
                   args = None,
                   res  = None,
                   outlog = 'log',
                   errlog = 'err') :
        """
        make submit script

        job_dirs(list):         directories of jobs. size: n_job
        cmds(list):             commands to be executed in each dir. size: n_job x n_cmd
        args(list of list):     args of commands. size: n_job x n_cmd
                                can be None
        res(dict):              resources available
        outlog(str):            file name for output
        errlog(str):            file name for error
        """
        res = self.default_resources(res)
        ret = self.sub_script_head(res)
        if not isinstance(job_dirs, list):
            job_dirs = [job_dirs]
        if not isinstance(cmds, list):
            cmds = [cmds]
        if not isinstance(cmds[0], list):
            cmds = [cmds for d in job_dirs]
        if args is None:
            args = [['' for c in jcmd] for jcmd in cmds]
        # loop over dirs 
        for jdir, jcmd, jarg in zip(job_dirs, cmds, args):            
            # for one dir
            ret += self._sub_script_inner(jdir,
                                          jcmd,
                                          jarg,
                                          res,
                                          outlog=outlog,
                                          errlog=errlog)
        ret += '\nwait\n'
        ret += '\ntouch %s\n' % self.finish_tag_name
        return ret

    def submit(self,
               job_dirs,
               cmds,
               args = None,
               res = None,
               restart = False,
               sleep = 0,
               outlog = 'log',
               errlog = 'err'):
        if restart:
            status = self.check_status()
            if status in [  JobStatus.unsubmitted, JobStatus.unknown, JobStatus.terminated ]:
                # dlog.debug('task restart point !!!')
                self.do_submit(job_dirs, cmds, args, res, outlog=outlog, errlog=errlog)
            elif status==JobStatus.waiting:
                pass
                # dlog.debug('task is waiting')
            elif status==JobStatus.running:
                pass
                # dlog.debug('task is running')
            elif status==JobStatus.finished:
                pass
                # dlog.debug('task is finished')
            else:
                raise RuntimeError('unknow job status, must be wrong')
        else:
            # dlog.debug('new task')
            self.do_submit(job_dirs, cmds, args, res, outlog=outlog, errlog=errlog)
        time.sleep(sleep) # For preventing the crash of the tasks while submitting        

    def check_finish_tag(self) :
        return self.context.check_file_exists(self.finish_tag_name)

    def _sub_script_inner(self, 
                          job_dir,
                          cmds,
                          args,
                          res,
                          outlog = 'log',
                          errlog = 'err') :
        ret = ""
        ret += 'cd %s\n' % job_dir
        ret += 'test $? -ne 0 && exit\n\n'
        try:
            allow_failure = res['allow_failure']
        except:
            allow_failure = False
        for idx, (icmd, iarg) in enumerate(zip(cmds, args)) :
            ret += 'if [ ! -f tag_%d_finished ] ;then\n' % idx
            ret += '  %s 1>> %s 2>> %s \n' % (self.sub_script_cmd(icmd, iarg, res), outlog, errlog)
            if not allow_failure:
                ret += '  if test $? -ne 0; then exit; else touch tag_%d_finished; fi \n' % idx
            else :
                ret += '  touch tag_%d_finished \n' % idx
            ret += 'fi\n\n'
        ret += 'cd %s\n' % self.context.remote_root
        ret += 'test $? -ne 0 && exit\n\n\n'
        return ret
