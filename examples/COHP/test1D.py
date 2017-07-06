#!/usr/bin/env python
import pythtb
import matplotlib.pyplot as plt
import numpy as np
from wann_ham import anatb
from wannier_band_plot import plot_band_weight
def test_1d(dt=0.0,t=1):
    E0=0.0
    mymodel=pythtb.tb_model(dim_k=3,dim_r=3,lat=np.eye(3),orb=[[0,0,0],[0,0,0.1]])
    mymodel.set_onsite(-E0,0)
    mymodel.set_onsite(E0,1)
    mymodel.set_hop(-t-dt,0,1,[0,0,0])
    mymodel.set_hop(-t+dt,1,0,[0,0,1])
    zs=np.arange(-1,1,0.01)
    #zs=[0.0]
    kpoints= [[0,0,x] for x in zs]
    evals=mymodel.solve_all(kpoints)
    #for i in range(evals.shape[0]):
    #    plt.plot(zs,evals[i,:])
    myanatb=anatb(mymodel,kpts=kpoints)
    myanatb.plot_COHP_fatband(k_x=zs)
    return
    #wks=np.abs(myanatb.get_cohp_block_pair([0],[1]))
    #wks=myanatb.get_cohp_block_pair([0],[1])
    wks=myanatb.get_cohp_all_pair()
    wks=np.moveaxis(wks,0,-1)
    kslist=zs*2
    ekslist=evals
    wkslist=wks
    print(wks)
    axis=plot_band_weight(kslist,ekslist,wkslist=wks,efermi=None,yrange=None,output=None,style='color',color='blue',axis=None,width=10,xticks=None)
    axis.set_xlabel('k ($\pi/a$)')
    plt.show()
test_1d()
