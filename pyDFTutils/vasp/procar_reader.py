#!/usr/bin/env python

from numpy import zeros,inner
import numpy as np
import re
from pyDFTutils.ase_utils import symbol_number
import matplotlib.pyplot as plt
def fix_line(line):
    line=re.sub("(\d)-(\d)", r'\1 -\2',line)
    return line

class procar_reader():
    def __init__(self,fname='PROCAR'):
        self.read(fname=fname)

    def get_dos(self,iion,orb_name,iband):
        dos=inner(self.dos_array[iion,self.orb_dict[orb_name],iband,:],self.weight)
        return dos

    def filter_band(self,iion,orb_name,thr=0.01):
        """
        return a energy array 2D. energy(iband,nkpt).
        """
        band_ids=[]
        for iband in range(self.nbands):
            d=self.get_dos(iion,orb_name,iband)
            print(d)
            if d>thr:
                band_ids.append(iband)
        return self.energy[np.array(band_ids,dtype=int)]

    def plot_band(self,iion,orb_name,thr=0.01):
        earray=self.filter_band(iion,orb_name,thr=thr)
        for k_e in earray:
            plt.plot(k_e)
        plt.ylim(-3,2)
        #plt.show()
    def plot_band_alpha(self,iion,orb_name,color='k'):
        for iband in range(self.nbands):
            d=self.get_dos(iion,orb_name,iband)
            print(d)
            plt.plot(self.energy[iband],color,linewidth=d*50,alpha=0.5)

    def read(self,fname='PROCAR'):
        lines=open(fname).readlines()
        iline=0
        self.has_phase=bool(lines[iline].rfind('phase'))
        iline=1
        p=lines[iline].split()
        self.nkpts=int(p[3])
        self.nbands=int(p[7])
        self.nions=int(p[11])

        self.dos_label=lines[7].split()[1:-1]

        self.norb=len(self.dos_label)
        self.orb_dict=dict(list(zip(self.dos_label,list(range(self.norb)))))
        print(self.orb_dict)

        self.dos_array=zeros((self.nions,self.norb,self.nbands,self.nkpts),dtype='float')

        self.weight=zeros(self.nkpts,dtype='float')
        self.energy=zeros((self.nbands,self.nkpts))
        self.band_occ=zeros((self.nbands,self.nkpts))
        self.kpts=zeros((self.nkpts,3))

        iline+=1
        for ikpts in range(self.nkpts):
            iline+=1

            line_k=fix_line( lines[iline]).split()

            self.kpts[ikpts]=[float(x) for x in line_k[3:6]]
            self.weight[ikpts]=float(line_k[-1])

            iline+=2
            for iband in range(self.nbands):

                line_b=lines[iline].split()

                self.energy[iband,ikpts]=float(line_b[4])
                self.band_occ[iband,ikpts]=float(line_b[7])

                iline+=3
                for iion in range(self.nions):

                    #print iline
                    line_dos=lines[iline].strip().split()
                    #print iline
                    #print line_dos
                    self.dos_array[iion,:,iband,ikpts]=[float(x) for x in line_dos[1:-1]]
                    iline+=1
                    #if self.has_phase:
                    #iline+=1+self.nions*2
                iline+=3
        self.efermi=np.max(self.energy[self.band_occ>0.5])
        print(self.efermi)
        self.energy=self.energy-self.efermi
def test(iion=0,orb_name='dx2',thr=0.005):
    p=procar_reader()
    #for e in p.filter_band(0,orb_name,thr=thr):
    #    plt.plot(e,'.',color='green')
    #for e in p.filter_band(1,'dx2',thr=thr):
    #    plt.plot(e,'-',color='red')
        #plt.plot(p.filter_band(iion,'dz2',thr=thr))
    p.plot_band_alpha(1,'dx2',color='r')
    p.plot_band_alpha(1,'dz2',color='g')
    plt.ylim(-5,5)
    plt.show()


if __name__=='__main__':
    test()
