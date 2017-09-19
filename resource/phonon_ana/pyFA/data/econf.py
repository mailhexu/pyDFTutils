#!/usr/bin/env python
import os.path
import os
from pyFA.psp import get_oncv_econf,econf
def gen_database():
    data={}
    edict=eval(open('./oncvpbesol.txt').read())
    for elem in edict:
        pspfile=edict[elem]
        print elem
        print pspfile
        try:
            elem,z,nc,nv= get_oncv_econf(elem,pspfile)
            print(econf(elem).wannier_conf(ncore=nc,valence=0))
            data[elem]=econf(elem).wannier_conf(ncore=nc,valence=0)
        except:
            pass
    with open('ONCV_pbesol_conf.py','w') as myfile:
        myfile.write(str(data))


gen_database()

