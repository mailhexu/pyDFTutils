#!/usr/bin/env python
import math,cmath
import collections
from numpy.linalg import norm
class hamiltonian(object):
    """
    n_wann:
    """
    def __init__(n_wann,n_R):
        n_wann=0
        n_R=0
        degeneracy=dict()
        HmnR=dict()

def ham_parser(filename='wannier90_hr.dat',cutoff=None):
    """
    wannier90 hr file phaser.

    :param cutoff: the energy cutoff.  None | number | list (of Emin, Emax).
    """
    with open(filename,'r') as myfile:
        lines=myfile.readlines()
    n_wann=int(lines[1].strip())
    n_R=int(lines[2].strip())

    # The lines of degeneracy of each R point. 15 per line.
    nline=int(math.ceil(n_R/15.0))
    dlist=[]
    for i in range(3,3+nline):
        d=map(float,lines[i].strip().split())
        dlist+=d
    H_mnR=dict()
    for i in range(3+nline,3+nline+n_wann**2*n_R):
        t=lines[i].strip().split()
        R=tuple(map(int,t[:3]))
        m,n=map(int,t[3:5])
        m=m-1
        n=n-1
        H_real,H_imag=map(float, t[5:])
        val=H_real+1j*H_imag
        if cutoff is not None:
            if isinstance(cutoff,collections.Iterable):
                if cutoff[0]< abs(val) <cutoff[1] and not (m==n and norm(R)<0.001):
                    H_mnR[(m,n,R)]=val
            elif abs(val)>cutoff and not (m==n and norm(R)<0.001):
                H_mnR[(m,n,R)]=val
            elif m==n and norm(R)<0.001:
                H_mnR[(m,n,R)]=val
        else:
            H_mnR[(m,n,R)]=val

    #print(n_wann,n_R,len(dlist))
    #for k in H_mnR:
    #    if abs(H_mnR[k])>0.1:# and k[2]==(0,1,1):
    #        print k,abs(H_mnR[k])
    return H_mnR


if __name__ == '__main__':
    ham_phaser()
