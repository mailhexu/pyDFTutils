#!/usr/bin/env python
from pyDFTutils.queue.commander import zenobe_run_wannier90
import sys
if __name__=='__main__':
    if len(sys.argv)==2:
        spin = sys.argv[1]
    else:
        spin=None
    zenobe_run_wannier90(spin=spin)
