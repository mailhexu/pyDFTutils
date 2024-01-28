"""
Load abinit hist netcdf file. Parser of abinit hist file which is a netcdf file.
"""
from pathlib import Path
import numpy as np
import netCDF4 as nc
from ase import Atoms
from ase.units import Ang, Bohr, Hartree, eV, GPa
from pymatgen.core import Structure

class AbihistFile():
    """ The class to read abinit hist file. 
    """
    def __init__(self, fname):
        self.fname = fname
        self.read()

    def read(self):
        self.g = nc.Dataset(self.fname, "r")
        self.ntime = self.g.dimensions["time"].size
        self.natom = self.g.dimensions["natom"].size
        self.ntypat = self.g.dimensions["ntypat"].size
        self.npsp = self.g.dimensions["npsp"].size
        self.typat = self.g.variables["typat"][:]
        self.znucl = self.g.variables["znucl"][:]
        self.amu = self.g.variables["amu"][:]
        self.xcart = self.g.variables["xcart"][:] * (Bohr / Ang)
        self.xred = self.g.variables["xred"][:]
        self.fcart = self.g.variables["fcart"][:] * (Hartree / Bohr) / (eV / Ang)
        self.fred = self.g.variables["fred"][:] * (Hartree / eV)
        self.vel = self.g.variables["vel"][:]
        self.acell = self.g.variables["acell"][:] * (Bohr / Ang)
        self.rprimd = self.g.variables["rprimd"][:] * (Bohr / Ang)
        self.etotal = self.g.variables["etotal"][:] * (Hartree / eV)
        self.numbers = [int(self.znucl[int(i) - 1]) for i in self.typat]
        self.strten = self.g.variables["strten"][:] * (Hartree / Bohr**3) / GPa 

    def get_pymatgen_structure(self, i):
        """Return pymatgen structure at step i.
        """
        s = Structure(
            lattice=self.rprimd[i],
            species=self.numbers,
            coords=self.xred[i],
            coords_are_cartesian=False,
        )
        return s

    def get_ase_atoms(self, i):
        """Return ase atoms at step i.
        """
        atoms= Atoms(numbers=self.numbers, positions=self.xcart[i], cell=self.rprimd[i])
        return atoms

    def get_stress(self, i):
        """Return stress at step i.
        """
        ti=self.strten[i]
        stress=np.array([[ti[0], ti[5], ti[4]], 
                         [ti[5], ti[1], ti[3]], 
                         [ti[4], ti[3], ti[2]]])
        return stress 

    def get_energy(self, i):
        """Return energy at step i.
        """
        return self.etotal[i]
    
    def get_fcart(self, i):
        """Return force at step i.
        """
        return self.fcart[i]
    
    def get_fred(self, i):
        """Return force at step i.
        """
        return self.fred[i]
    
    def get_vel(self, i):
        """Return velocity at step i.
        """
        return self.vel[i]

    def get_ntime(self):
        return self.ntime
    

def read_structures_from_hist(fname):
    """Read from netcdf file and return list of Atoms objects, energies, forces, and stresses.
    :params:
        fname: str
            path to netcdf file
    :returns:
        atoms_list: list of Atoms objects
        energies: list of floats
        forces: list of numpy arrays
        stresses: list of numpy arrays
    """
    g = nc.Dataset(fname, "r")
    ntime = g.dimensions["time"].size
    natom = g.dimensions["natom"].size
    ntypat = g.dimensions["ntypat"].size
    npsp = g.dimensions["npsp"].size

    typat = g.variables["typat"][:]
    znucl = g.variables["znucl"][:]
    amu = g.variables["amu"][:]
    xcart = g.variables["xcart"][:] * (Bohr / Ang)
    xred = g.variables["xred"][:]
    fcart = g.variables["fcart"][:] * (Hartree / Bohr) / (eV / Ang)
    fred = g.variables["fred"][:] * (Hartree / eV)
    vel = g.variables["vel"][:]
    acell = g.variables["acell"][:] * (Bohr / Ang)
    rprimd = g.variables["rprimd"][:] * (Bohr / Ang)
    etotal = g.variables["etotal"][:] * (Hartree / eV)
    numbers = [int(znucl[int(i) - 1]) for i in typat]
    strten = g.variables["strten"][:] * (Hartree / Bohr**3) / GPa

    # put stress into a 3x3 matrix
    #stresses = np.array(
    #    [(strten[:, 0], strten[:, 5], strten[:, 4]),
    #    (strten[:, 5], strten[:, 1], strten[:, 3]),
    #    (strten[:, 4], strten[:, 3], strten[:, 2])]
    #)
    #stresses=np.swapaxes(stresses, 0, 2)

    # put the atoms into pymatgen structures
    structures = []
    stresses= [] 
    for i in range(ntime):
        s = Structure(
            lattice=rprimd[i],
            species=numbers,
            coords=xred[i],
            coords_are_cartesian=False,
        )
        structures.append(s)
        ti=strten[i]
        stress=np.array([[ti[0], ti[5], ti[4]], 
                         [ti[5], ti[1], ti[3]], 
                         [ti[4], ti[3], ti[2]]])
        stresses.append(stress)
    stresses= np.array(stresses)


    # check the unit of the stress
    #if "virials" in system:
    #    # TODO: check this against https://github.com/mailhexu/dpdata/blob/master/dpdata/plugins/ase.py
    #    # kbar-> virial  1e3
    #    # GPar-> virial  1e4
    #    v_pref = 1 * 1e4 / 1.602176621e6
    #    for ii in range(system["cells"].shape[0]):
    #        vol = np.linalg.det(np.reshape(system["cells"][ii], [3, 3]))
    #        system["virials"][ii] *= v_pref * vol

    ## TODO: check the stresses needs some conversion of units or convention.
    return structures, etotal, fcart, stresses


def test_read_structures_from_hist():
    fname = Path("/home/hexu/projects/M3gnetmodels/matgl/training") / "STO_TS.nc"
    structures, etotal, fcart, stresses= read_structures_from_hist(fname)


if __name__ == "__main__":
    test_read_structures_from_hist()
