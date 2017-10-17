#!/bin/bash
cd /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w2/t4
# OpenMp Environment
export OMP_NUM_THREADS=1
mpirun -n 1 abinit < /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w2/t4/run.files > /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w2/t4/run.log 2> /Users/hexu/pyDFTutils/pyDFTutils/abipy/BaTiO3_phonon/w2/t4/run.err
