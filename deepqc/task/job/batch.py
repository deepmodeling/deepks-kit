import os,sys,time
from itertools import zip_longest
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
        raise NotImplementedError('abstract method check_status should be implemented by derived class')        
        
    def default_resources(self, res) :
        raise NotImplementedError('abstract method sub_script_head should be implemented by derived class')        

    def sub_script_head(self, res) :
        raise NotImplementedError('abstract method sub_script_head should be implemented by derived class')        

    def sub_script_cmd(self, cmd, arg, res):
        raise NotImplementedError('abstract method sub_script_cmd should be implemented by derived class')        

    def exec_sub_script(self, script_str):
        raise NotImplementedError('abstract method exec_sub_script should be implemented by derived class')

    def check_before_sub(self, res):
        pass

    def sub_step_head(self, step_res=None):
        return ""

    def do_submit(self,
                  job_dirs,
                  cmds,
                  args = None, 
                  res = None,
                  outlog = 'log',
                  errlog = 'err',
                  para_deg = 1,
                  para_res = None):
        '''
        submit a single job, assuming that no job is running there.
        '''
        if res is None:
            res = self.default_resources(res)
        self.check_before_sub(res)
        script_str = self.sub_script(job_dirs, cmds, args=args, res=res, 
                                     outlog=outlog, errlog=errlog,
                                     para_deg=para_deg, para_res=para_res)
        self.exec_sub_script(script_str=script_str)

    def sub_script(self,
                   job_dirs,
                   cmds,
                   args = None,
                   res  = None,
                   outlog = 'log',
                   errlog = 'err',
                   para_deg = 1,
                   para_res = None) :
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
        if not isinstance(para_res, list):
            para_res = [para_res for d in job_dirs]
        # loop over cmds 
        for jj, (jcmds, jargs) in enumerate(zip(zip_longest(*cmds), zip_longest(*args))):
            # for one cmd per dir
            ret += self._sub_script_inner(job_dirs,
                                          jcmds,
                                          jargs,
                                          res,
                                          jj,
                                          outlog=outlog,
                                          errlog=errlog,
                                          para_deg=para_deg,
                                          para_res=para_res)
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
               errlog = 'err',
               para_deg = 1,
               para_res = None):
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
            self.do_submit(job_dirs, cmds, args, res, 
                           outlog=outlog, errlog=errlog, 
                           para_deg=para_deg, para_res=para_res)
        time.sleep(sleep) # For preventing the crash of the tasks while submitting        

    def check_finish_tag(self) :
        return self.context.check_file_exists(self.finish_tag_name)

    def _sub_script_inner(self, 
                          job_dirs,
                          cmds,
                          args,
                          res,
                          step = 0,
                          outlog = 'log',
                          errlog = 'err',
                          para_deg = 1,
                          para_res = None) :
        try:
            allow_failure = res['allow_failure']
        except:
            allow_failure = False
        ret = ""
        # iter over job dirs
        for idx, (idir, icmd, iarg) in enumerate(zip(job_dirs, cmds, args)) :
            ret += 'cd %s\n' % idir
            ret += 'test $? -ne 0 && exit\n\n'

            if para_deg > 1:
                tmp_cmd = self.sub_script_cmd(icmd, iarg, res)
                if not res.get("with_mpi", False):
                    tmp_cmd = self.sub_step_head(para_res[idx]) + tmp_cmd
                ret += '  %s 1>> %s 2>> %s &\n\n' % (tmp_cmd, outlog, errlog)
                
            else:
                ret += 'if [ ! -f tag_%d_finished ] ;then\n' % step
                tmp_cmd = self.sub_script_cmd(icmd, iarg, res)
                ret += '  %s 1>> %s 2>> %s \n' % (tmp_cmd, outlog, errlog)
                if not allow_failure:
                    ret += '  if test $? -ne 0; then exit; else touch tag_%d_finished; fi \n' % step
                else :
                    ret += '  touch tag_%d_finished \n' % step
                ret += 'fi\n\n'

            ret += 'cd %s\n' % self.context.remote_root
            ret += 'test $? -ne 0 && exit\n'
            if para_deg > 1 and (idx+1) % para_deg == 0:
                ret += "\nwait\n"
            ret += "\n\n"

        ret += "\nwait\n"
        return ret
