#!/bin/bash
cd /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w0/t0
# OpenMp Environment
export OMP_NUM_THREADS=1
mpirun -n 1 abinit --timelimit 0-12:0:0 < /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w0/t0/run.files > /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w0/t0/run.log 2> /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w0/t0/run.err
