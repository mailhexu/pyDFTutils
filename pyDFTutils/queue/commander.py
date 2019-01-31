#/usr/bin/env python
from __future__ import print_function
import string
import os
import time
import subprocess
import os.path


def nic4script(command='abinit', **kwargs):
    defaults = dict(
        jobname='abinit',
        time="21:01:00",
        ntask=16,
        ntask_per_node=16,
        mem_per_cpu=1000)
    if command == 'abinit':
        defaults['command'] = 'mpirun abinit < abinit.files > abinit.log;'
    elif command == 'vasp':
        defaults['command'] = 'mpirun vasp_std > log;'
    else:
        defaults['command'] = command
    for key in kwargs:
        defaults[key] = kwargs[key]
    with open(os.path.expanduser('~/.ase/nic4.tmpl')) as myfile:
        tmpl = string.Template(myfile.read())
        text = tmpl.substitute(defaults)
    return text


def zenobescript(
        command='abinit',
        infile='abinit.files',
        outfile='abinit.txt',
        workdir='./',
        #queue_type='pbspro',
        jobname='unamed',
        queue='large',
        group='spinphon',
        time="23:00:00",
        ngroup=4,
        mpiprocs=24,
        ompthreads=1,
        mem_per_cpu=2400):
    ncpus = mpiprocs * ompthreads
    if ncpus == 24 and ngroup >= 4:
        queue = 'large'
    else:
        queue = 'main'
    if mpiprocs * ngroup == 1:
        mpirun = ''
    else:
        mpirun = 'mpirun'
    defaults = dict(
        jobname=jobname,
        queue=queue,
        time=time,
        ngroup=ngroup,
        ncpus=ncpus,
        mpiprocs=mpiprocs,
        group=group,
        mem_per_group=mem_per_cpu * ncpus,
        ompthreads=ompthreads,
        mpirun=mpirun)
    if command == 'abinit':
        infile=os.path.abspath(os.path.join(workdir, 'abinit.files'))
        outfile=os.path.abspath(os.path.join(workdir, 'abinit.log'))
        defaults[
            'command'] = r'/home/acad/ulg-phythema/hexu/.local/abinit/abinit_8.6.1/bin/abinit' # + infile + ' >' + outfile
        defaults[
            'fullcommand'] = r'mpirun /home/acad/ulg-phythema/hexu/.local/abinit/abinit_8.6.1/bin/abinit <%s >%s'%(infile, outfile) 

        defaults['infile']=infile
        defaults['outfile']=outfile
    elif command == 'vasp':
        defaults[
            'command'] = r'/home/acad/ulg-phythema/hexu/.local/bin/vasp_hexu544 >log'
        defaults[
                'fullcommand'] = r'mpirun /home/acad/ulg-phythema/hexu/.local/bin/vasp_hexu544>log'
        defaults['infile']=''
        defaults['outfile']=''
    else:
        defaults['command'] = command
        defaults['fullcommand'] = command
    with open(os.path.expanduser('~/.ase/zenobe.tmpl')) as myfile:
        tmpl = string.Template(myfile.read())
        text = tmpl.substitute(defaults)
    return text


class commander(object):
    def __init__(self, job_fname='job.sh', workdir='./', jobname='unamed', **kwargs):
        self.workdir = workdir
        self.jobname=jobname
        self.job_fname = os.path.join(workdir, job_fname)
        self.queue_type = None
        self.set_parameter(**kwargs)

    def set_parameter(self,
                      queue_type='pbspro',
                      command='abinit',
                      max_time=24 * 60 * 60,
                      wait=True,
                      **kwargs):
        """
        set script parameters.
        Parameters:
        =============
        command: string.
           'abinit'|'vasp'| user defined com.
        **kwargs: jobname, time, ntask, ntask_per_node, mem_per_cpu
        """
        self.queue_type = queue_type
        if queue_type == 'slurm':
            self.jobfile_text = nic4script(command, **kwargs)
        elif queue_type == 'pbspro':
            self.jobfile_text = zenobescript(command, jobname=self.jobname, workdir=self.workdir, **kwargs)
        self.max_time = max_time
        self.wait = wait

    def run_zenobe(self):
        # write job script.
        fname = self.job_fname
        with open(fname, 'w') as myfile:
            myfile.write(self.jobfile_text)
        os.system('chmod +x %s' % fname)

        p = subprocess.Popen(
            ['qsub', self.job_fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = p.communicate()
        job_id = out.decode().strip()
        if self.wait:
            exitcode = wait_job_success(job_id, maxtime=self.max_time)
        else:
            exitcode = 0

        return exitcode

    def run(self):
        if self.queue_type == 'pbspro':
            return self.run_zenobe()
        if self.queue_type == 'slurm':
            raise NotImplementedError('Slurm is not on zenobe')
        else:
            raise NotImplementedError('%s is not on zenobe' % self.queue_type)


def zenobe_abinit_large(queue_type='pbspro',
                        command='abinit',
                        workdir='./',
                        jobname='unamed',
                        queue='large',
                        group='spinphon',
                        time="23:00:00",
                        ngroup=4,
                        mpiprocs=24,
                        ompthreads=1,
                        mem_per_cpu=2400):
    mycommander = commander(
        queue_type=queue_type,
        command=command,
        jobname=jobname,
        workdir=workdir,
        queue=queue,
        group=group,
        time=time,
        ngroup=ngroup,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        mem_per_cpu=mem_per_cpu)
    return mycommander


def zenobe_abinit_main(queue_type='pbspro',
                       command='abinit',
                       jobname='unamed',
                       workdir='./',
                       queue='main',
                       group='spinphon',
                       time="23:00:00",
                       ngroup=12,
                       mpiprocs=1,
                       ompthreads=1,
                       mem_per_cpu=1900):
    mycommander = commander(
        queue_type=queue_type,
        command=command,
        jobname=jobname,
        workdir=workdir,
        queue=queue,
        group=group,
        time=time,
        ngroup=ngroup,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        mem_per_cpu=mem_per_cpu)
    return mycommander


def zenobe_vasp_large(queue_type='pbspro',
                      command='vasp',
                      jobname='unamed',
                      queue='large',
                      group='spinphon',
                      time="23:00:00",
                      ngroup=4,
                      mpiprocs=24,
                      ompthreads=1,
                      mem_per_cpu=2400,
                      wait=True):
    mycommander = commander(
        queue_type=queue_type,
        command=command,
        jobname=jobname,
        queue=queue,
        group=group,
        time=time,
        ngroup=ngroup,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        mem_per_cpu=mem_per_cpu,
        wait=wait)
    return mycommander


def zenobe_vasp_main(queue_type='pbspro',
                     command='vasp',
                     jobname='unamed',
                     queue='main',
                     group='spinphon',
                     time="23:00:00",
                     ngroup=1,
                     mpiprocs=12,
                     ompthreads=1,
                     mem_per_cpu=1400,
                     wait=True):
    mycommander = commander(
        queue_type=queue_type,
        command=command,
        jobname=jobname,
        queue=queue,
        group=group,
        time=time,
        ngroup=ngroup,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        mem_per_cpu=mem_per_cpu,
        wait=wait)
    return mycommander


def zenobe_wannier90(queue_type='pbspro',
                     command='/home/acad/ulg-phythema/hexu/.local/bin/wannier90.x wannier90.up.win',
                     queue='large',
                     group='spinphon',
                     time="2:00:00",
                     ngroup=1,
                     mpiprocs=1,
                     ompthreads=12,
                     mem_per_cpu=1900,
                     wait=False):
    mycommander = commander(
        queue_type=queue_type,
        command=command,
        queue=queue,
        group=group,
        time=time,
        ngroup=ngroup,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        mem_per_cpu=mem_per_cpu,
        wait=wait)
    return mycommander


def zenobe_run_wannier90(spin=None, **kwargs):
    if spin is None:
        command = '/home/acad/ulg-phythema/hexu/.local/bin/wannier90.x wannier90.win'
    elif spin == 'up':
        command = '/home/acad/ulg-phythema/hexu/.local/bin/wannier90.x wannier90.up.win'
    elif spin == 'dn' or spin == 'down':
        command = '/home/acad/ulg-phythema/hexu/.local/bin/wannier90.x wannier90.dn.win'
    else:
        raise NotImplementedError("spin should be None|up|dn")
    mycommander = zenobe_wannier90(command=command, **kwargs)
    mycommander.run()


class remote_commander(commander):
    def __init__(self, job_fname='slurmrun.sh', delete_remote_dir=True):
        self.job_fname = job_fname
        self.delete_remote_dir = delete_remote_dir
        pass

    def set_server(self,
                   remote_address,
                   username,
                   password,
                   remote_dir='~/Data',
                   key_filename=None,
                   local_dir='./tmp'):
        """
        set remote server.
        """
        self.rr = remote_run(
            remote_address,
            username,
            password,
            remote_dir,
            key_filename=key_filename,
            local_dir=local_dir)
        self.local_dir = local_dir
        with open('remote_path.txt', 'a') as myfile:
            myfile.write('%s:%s' % (remote_address, remote_dir))

    def set_parameter(self,
                      queue_type='slurm',
                      command='abinit',
                      max_time=24 * 60 * 60,
                      **kwargs):
        """
        set script parameters.
        Parameters:
        =============
        command: string.
           'abinit'|'vasp'| user defined com.
        **kwargs: jobname, time, ntask, ntask_per_node, mem_per_cpu
        """
        if queue_type == 'slurm':
            self.jobfile_text = nic4script(command, **kwargs)
        elif queue_type == 'pbspro':
            self.jobfile_text = zenobescript(command, **kwargs)
        self.max_time = max_time

    def set_command(self, command):
        self.command = command

    def run(self):
        # write job script.
        fname = os.path.join(self.local_dir, self.job_fname)
        with open(fname, 'w') as myfile:
            myfile.write(self.jobfile_text)
        os.system('chmod +x %s' % fname)
        # sync to remote
        self.rr.sync()
        # run remote commands
        out = self.rr.exec_command(self.command)
        id = out.strip().split()[-1]
        if self.rr.remote_address == 'nic4.segi.ulg.ac.be':
            fname = '/home/ulg/phythema/hexu/.jobs/%s.txt' % id
        elif self.rr.remote_address == 'zenobe.hpc.cenaero.be':
            fname = '/home/acad/ulg-phythema/hexu/.jobs/%s.txt' % id
        else:
            raise ValueError('only support nic4 and zenobe')
        print("Waiting for job %s..." % id)
        step = 1
        wait_time = 0
        exist = False
        while (not exist) and wait_time < self.max_time:
            exist = self.rr.exists(fname)
            #print self.rr.exec_command('cat %s'%fname)
            time.sleep(step)
            wait_time += step
        print("End waiting.")
        self.rr.sync_back()
        if self.rr.exists(fname):
            #print('Succeeded')
            if self.delete_remote_dir:
                self.rr.delete_remote_dir()


def wait_job_success(job_id, maxtime=20 * 60 * 60):
    """
    wait for a job. (when the job is done, a file in written to ~/.jobs/id.txt
    The content of the file:
    Success/Fail
    )
    """
    import os
    import time
    fname = os.path.expanduser("~/.jobs/%s.txt" % job_id)
    print("waiting for job %s" % job_id)
    while not os.path.exists(fname):
        time.sleep(10)
    if os.path.isfile(fname):
        text = open(fname).read()
        if text == "Success":
            return True
        else:
            return False


if __name__ == '__main__':
    zenobe_run_wannier90(ompthreads=12)
    #test_paramiko_nic4()
    #print(nic4run(jobname='name'))

    #mycommander = commander(
    #    command='abinit',
    #    queue_type='pbspro',
    #    #local_dir='./tmpz',
    #    job_fname='tscript.sh',
    #    queue='main',
    #    time="23:00:00",
    #    ngroup=1,
    #    mpiprocs=1,
    #    mem_per_cpu=400)
    #mycommander.run()
