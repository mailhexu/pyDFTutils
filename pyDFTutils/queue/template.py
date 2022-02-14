slurm_template = """#!/bin/bash

# Submission script for NIC4 
#SBATCH --job-name={jobname}
#SBATCH --time={time} #hh:mm:ss
#SBATCH --output=mpitest.txt
#SBATCH --ntasks={ntask}
#SBATCH --ntasks-per-node={ntask_per_node}
#
{modules}
{command}
"""

# print(slurm_template.format(jobname='test', time="10:00:00",
#                            ntask=32, ntask_per_node=32, modules='', command='vasp'))
