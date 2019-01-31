#!/bin/bash

# Submission script for NIC4 
#SBATCH --job-name=phonon
#SBATCH --time=23:05:00 # hh:mm:ss
#SBATCH --output=mpitest.txt
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=1000 # megabytes 
#SBATCH --partition=defq 
#
#SBATCH --mail-user=mailhexu@gmail.com
#SBATCH --mail-type=ALL
#
#SBATCH --comment=proj 
module load intel/compiler/64/14.0/2013_sp1.1.106 mvapich2/qlc/intel/64/1.9 intel/mkl/64/11.1/2013_sp1.3.174  abinit/8.0.7 
mkdir -p "$GLOBALSCRATCH/$SLURM_JOB_ID"
echo "copy to scratch"
echo "SLURM Submit DIR: $SLURM_SUBMIT_DIR"
echo "Scratch dir: $GLOBALSCRATCH/$SLURM_JOB_ID"
rsync -a "$SLURM_SUBMIT_DIR"/* "$GLOBALSCRATCH/$SLURM_JOB_ID" &&\
echo "Copy to Scratch Succeeded! Scratch dir: " "$GLOBALSCRATCH/$SLURM_JOB_ID"

( cd "$GLOBALSCRATCH/$SLURM_JOB_ID" ;

echo "Cd to Scratch Succeeded! Scratch dir: " "$GLOBALSCRATCH/$SLURM_JOB_ID";
echo "Current dir: $(pwd)";
echo "$(ls -a)";
mpirun abinit_dev < abinit.files > abinit.log;
)
rsync -a "$GLOBALSCRATCH/$SLURM_JOB_ID"/* "$SLURM_SUBMIT_DIR/" --exclude '*WF*' --exclude '*DEN*' &&\ 
rm -rf "$GLOBALSCRATCH/$SLURM_JOB_ID" &&\
echo "Cleaning Scratch: Succeeded!"
echo "Success" > "$HOME"/.jobs/"$SLURM_JOB_ID".txt

