#!/usr/bin/env python
"""
wannier 90 interface with tight binding.
"""
from pythtb import tb_model
from wannier_interface import Wannier_interface
import numpy as np
from numpy.linalg import norm
from TBpack.myTB import mytb
class wannTB():
    """
    wannier90 interface to tight binding model.
    """
    def __init__(self,name,cutoff=0.01):
        """
        init wannier to tight binding interface.

        :param name: name of the wannier.
        :param cutoff: cutoff energy of the overlap.
        """
        self.name=name
        self.cutoff=cutoff
        self.dim_r=3
        self.lat=None
        self.orb=None
        self.per=None
        self.nspin=1
        self.reciprocal_cell=None

        self.tb=None

    def read_wannier(self,remove_basis_list=[],remove_site_list=[],poscar='POSCAR',check_center=False):
        """
        read from the wannier90 output.
        """
        wint=Wannier_interface(name=self.name)
        wint.read_win(poscar=poscar)
        wint.read_centers()
        if check_center:
            wint.check_center()
        wint.print_basis_list()
        wint.read_hamiltonian(cutoff=self.cutoff)
        if remove_basis_list:
            wint.remove_basis(remove_basis_list)
        if remove_site_list:
            wint.remove_sites(remove_site_list)

        self.lat=wint.atoms.get_cell()
        self.dim_r=len(self.lat)
        self.orb = wint.wannier_centers
        self.hamiltonian =wint.hamiltonian



    def build_tb(self,dim_k=None):
        """
        generate tb_model.
        dim_k: dimension of kpoints. Default: None, aka same as dim_r.
        """
        if dim_k is None:
            dim_k=self.dim_r
        tb=mytb(dim_k,self.dim_r, self.lat,self.orb,nspin=self.nspin)
        for key,val in self.hamiltonian.items():
            m,n,R=key
            if m==n and norm(R)==0:
                print m
                tb.set_onsite(val.real,ind_i=m)
            else:
                if m<n:
                    tb.set_hop(val,m,n,ind_R=R,allow_conjugate_pair=False,mode="set")
                else:
                    pass
        self.tb=tb
        return tb
