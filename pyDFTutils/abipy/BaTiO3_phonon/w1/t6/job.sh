#!/bin/bash
cd /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w1/t6
# OpenMp Environment
export OMP_NUM_THREADS=1
mpirun -n 1 abinit < /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w1/t6/run.files > /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w1/t6/run.log 2> /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w1/t6/run.err
