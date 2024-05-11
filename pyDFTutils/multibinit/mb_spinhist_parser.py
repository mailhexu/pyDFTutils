"""
multibinit spin hist file parser
"""

import numpy as np
import os
import netCDF4 as nc
from pathlib import Path


class SpinHistParser():
    def __init__(self, path):
        self.path = path
        self.data = None
        self._parse_all()
        self.get_mapping_supercell_to_primcell()
        self.get_mapping_primcell_to_supercell()

    def _parse_all(self):
        with nc.Dataset(self.path, 'r') as ncfile:
            # dimensions
            self.nspin = ncfile.dimensions['nspin'].size
            self.nsublatt = ncfile.dimensions['nsublatt'].size
            self.prim_nspins = ncfile.dimensions['prim_nspins'].size
            self.ntime = ncfile.dimensions['ntime'].size
            self.three = 3

            # variables
            self.time = ncfile.variables['time'][:]
            self.etotal = ncfile.variables['etotal'][:]
            self.S = ncfile.variables['S'][:]

            # mapping between primcell spins and supercell spins
            self.Rvec = ncfile.variables['Rvec'][:]
            self.ispin_prim = ncfile.variables['ispin_prim'][:]


    def get_S(self,itime, ispin):
        return self.S[itime, ispin]

    def get_mapping_supercell_to_primcell(self):
        """
        return a mapping between supercell spins and primcell spins
        """
        self.sc_to_pc = {}
        for i in range(self.nspin):
            R = tuple(self.Rvec[i])
            ispin_prim = self.ispin_prim[i]

            self.sc_to_pc[i] = (R, ispin_prim)
        return self.sc_to_pc

    def get_mapping_primcell_to_supercell(self):
        """
        return a mapping between primcell spins and supercell spins
        """
        self.pc_to_sc = {}
        for i in range(self.nspin):
            R = tuple(self.Rvec[i])
            ispin_prim = self.ispin_prim[i]
            self.pc_to_sc[(R, ispin_prim)] = i
        return self.pc_to_sc

    def Rset(self):
        """
        return a set of R vectors
        """
        Rset = set(tuple(R) for R in self.Rvec)
        return Rset
    
        
def average_S1dotS2(fname, itime):
    """
    calculate the average of S1 dot S2, where S1 and S2 are spins on different sublattices
    """
    hist = SpinHistParser(fname)
    S = hist.S
    print(S[0, 0, :])
    print(S[0, 1, :])
    Sd=np.sum(S[itime, ::2, :] * S[itime, 1::2, :], axis=1)
    Sdv=np.average(Sd)
    print(Sdv)
    Rset = hist.Rset()
    total = 0
    for R in Rset:
        ispin1 = hist.pc_to_sc[(R, 1)]
        ispin2 = hist.pc_to_sc[(R, 2)]
        S1 = hist.get_S(itime, ispin1)
        S2 = hist.get_S(itime, ispin2)
        Sdot = np.dot(S1, S2)
        total += Sdot
    return total/len(Rset)
    

def testSpinHistParser():
    fname= "mb.out_T0016_spinhist.nc"
    hist = SpinHistParser(fname)
    print(f"nspin: {hist.nspin}")
    print(f"nsublatt: {hist.nsublatt}")
    print(f"prim_nspins: {hist.prim_nspins}")
    print(average_S1dotS2(fname, 0))





if __name__ == "__main__":
    testSpinHistParser()
