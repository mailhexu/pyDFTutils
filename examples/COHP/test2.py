#!/usr/bin/env python
import pythtb
import matplotlib.pyplot as plt
import numpy as np
from wann_ham import anatb
from wannier_band_plot import plot_band_weight
import ase
def test_1d(dt=0.0,t=1):
    mymodel=pythtb.w90(path='../test',prefix='wannier90.up').model(min_hopping_norm=0.001)
    #zs=[0.0]
    kpath=np.array([[0.0,0,0],[0,1,0],[1,1,0],[0,0,0],[1,1,1],[0,1,0]])/2.0
    kpath=np.dot(kpath,np.array([[1,-1,0],[1,1,0],[0,0,2]]))
    kpoints, zs, special_zs=ase.dft.kpoints.bandpath(kpath,cell=np.eye(3),npoints=210)
    #wa.set_kpath(kpath,['GM','X','M','G','R','X'],npoints=410)
    xticks=(['GM','X','M','GM','R','X'],special_zs)
    evals=mymodel.solve_all(kpoints)
    #for i in range(evals.shape[0]):
    #    plt.plot(zs,evals[i,:])
    myanatb=anatb(mymodel,kpts=kpoints)
    #myanatb.plot_COHP_fatband(k_x=zs,xticks=xticks,iblock=range(20),jblock=range(20,56))
    #myanatb.plot_COHP_fatband(k_x=zs,xticks=xticks,iblock=range(0,36),jblock=range(0,36))
    myanatb.plot_COHP_fatband(k_x=zs,xticks=xticks,iblock=range(0,36),jblock=range(36,37))
    return
    #wks=myanatb.get_cohp_block_pair(iblock=range(20),jblock=range(20))
    #wks=np.moveaxis(wks,0,-1)
    #kslist=zs*2
    #ekslist=evals
    #wkslist=wks
    #print(wks)
    #axis=plot_band_weight(kslist,ekslist,wkslist=wks,efermi=None,yrange=None,output=None,style='color',color='blue',axis=None,width=10,xticks=xticks)
    #axis.set_xlabel('k ($\pi/a$)')
    #plt.show()
test_1d()

