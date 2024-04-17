"""
Load abinit hist netcdf file. Parser of abinit hist file which is a netcdf file.
"""
from pathlib import Path
import numpy as np
import netCDF4 as nc
from ase import Atoms
from ase.units import Ang, Bohr, Hartree, eV, GPa
from pymatgen.core import Structure
from ase.data import atomic_masses
from abipy.abilab import abiopen


def voigt_to_tensor(voigt):
    """Convert 6x1 voigt notation to 3x3 tensor.
    """
    return np.array([[voigt[0], voigt[5], voigt[4]],
                     [voigt[5], voigt[1], voigt[3]],
                     [voigt[4], voigt[3], voigt[2]]])

def tensor_to_voigt(tensor):
    """Convert 3x3 tensor to 6x1 voigt notation.
    """
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     tensor[1, 2], tensor[0, 2], tensor[0, 1]])


class AbihistFile():
    """ The class to read abinit hist file. 
    """
    def __init__(self, fname):
        self.fname = fname

    def read(self):
        self.g = nc.Dataset(self.fname, "r")
        self.time = self.g.dimensions["time"].size
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
        self.dtion = self.g.variables["dtion"][:]
        try:
            self.spinat = self.g.variables["spinat"][:]
        except Exception as e:
            self.spinat = np.zeros((self.time, self.natom, 3))
            print("Warning: spinat not found in hist file.")
        self.acell = self.g.variables["acell"][:]
        self.g.close()

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

    def dump_atoms(self, prefix, fmt="vasp", i=None, **kwargs):
        """Write atoms to file. If i is None, write all structures,
                      Otherwise, write the i-th structure.
        :params:
          prefix: str
            the prefix of the output file
          fmt: str
            the format of the output file
          i: int
            the index of the structure to write
          **kwargs:
            additional arguments to write function in ase.
        """
        if i is None:
            for i in range(self.time):
                atoms = self.get_ase_atoms(i)
                atoms.write(f"{prefix}_{i}.{fmt}")
        else:
            atoms = self.get_ase_atoms(i)
            atoms.write(f"{prefix}_{i}.{fmt}")


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

    def get_time(self):
        return self.time
    

def write_ase_traj_to_abihist(fname,  traj, energies, forces, stresses):
    """
    Write ase trajectory to abinit hist netcdf file.
    params:
        fname: str
            path to netcdf file
        traj: list of ase.Atoms objects
    """
    atoms0 = traj[0]
    natom = len(atoms0)
    time = len(traj)
    species = list(set(atoms0.numbers))
    ntypat = len(species)
    znucl = np.array([int(i) for i in species])
    npsp = len(species)
    typat = np.array([species.index(i) + 1 for i in atoms0.numbers])
    amu = np.array([atomic_masses[i] for i in species])

    etotal = np.array(energies) / (Hartree / eV)
    xcart = np.array([atoms.get_positions() for atoms in traj]) / (Bohr / Ang)
    xred = np.array([atoms.get_scaled_positions() for atoms in traj])
    #fcart = np.array([atoms.get_forces() for atoms in traj]) / (Hartree / Bohr) * (eV / Ang)
    rprimd = np.array([atoms.get_cell() for atoms in traj]) / (Bohr / Ang)
    #strten = np.array([atoms.get_stress() for atoms in traj]) / (Hartree / Bohr**3) * GPa
    #etotal = np.array([atoms.get_total_energy() for atoms in traj]) / (Hartree / eV)
    fcart = np.array(forces) / (Hartree / Bohr) * (eV / Ang)
    strten = np.array([tensor_to_voigt(stress) for stress in stresses]) / (Hartree / Bohr**3) * GPa
    acell =  np.linalg.norm(rprimd, axis=2)
    rprim = rprimd / acell[None, None, :]
    spinat=np.zeros((time, natom, 3))
    magmoms = np.array([atoms.get_initial_magnetic_moments() for atoms in traj])
    if len(magmoms.shape) == 2:
        spinat[:, :, 2] = magmoms
    else:
        spinat[:, :, :] = magmoms
    try:
        vel = np.array([atoms.get_velocities() for atoms in traj])
    except:
        vel = np.zeros((time, natom, 3))

    mdtime = np.array([i for i in range(time)], dtype=float)
    ekin = np.zeros(time, dtype=float)


    g = nc.Dataset(fname, "w")
    g.createDimension("natom", natom)
    g.createDimension("time", time)
    g.createDimension("ntypat", ntypat)
    g.createDimension("npsp", npsp)
    g.createDimension("xyz", 3)
    g.createDimension("six", 6)
    g.createVariable("typat", "f8", ("natom",))
    g.createVariable("znucl", "f8", ("ntypat",))
    g.createVariable("amu", "f8", ("ntypat",))
    g.createVariable("xcart", "f8", ("time", "natom", "xyz"))
    g.createVariable("xred", "f8", ("time", "natom", "xyz"))
    g.createVariable("fcart", "f8", ("time", "natom", "xyz"))
    g.createVariable("fred", "f8", ("time", "natom", "xyz"))
    g.createVariable("rprimd", "f8", ("time", "xyz", "xyz"))
    g.createVariable("strten", "f8", ("time", "six"))
    g.createVariable("etotal", "f8", ("time",))
    g.createVariable("ekin", "f8", ("time",))
    g.createVariable("entropy", "f8", ("time",))
    g.createVariable("vel", "f8", ("time", "natom", "xyz"))
    g.createVariable("acell", "f8", ("time", "xyz"))
    g.createVariable("spinat", "f8", ("time", "natom", "xyz"))
    g.createVariable("dtion", "f8", ())
    g.createVariable("mdtime", "f8", ("time"))
    g.createVariable("rprim", "f8", ("time", "xyz", "xyz"))

    g.variables["typat"][:] = typat
    g.variables["znucl"][:] = znucl
    g.variables["amu"][:] = amu
    g.variables["xcart"][:] = xcart
    g.variables["xred"][:] = xred
    g.variables["fcart"][:] = fcart
    #g.variables["fred"][:] = fred
    g.variables["rprimd"][:] = rprimd
    g.variables["rprim"][:] = rprim
    g.variables["strten"][:] = strten
    g.variables["etotal"][:] = etotal
    g.variables["vel"][:] = vel
    g.variables["acell"][:] = acell
    g.variables["spinat"][:] = spinat
    g.variables["dtion"][:] = 1.0
    g.variables["mdtime"][:] = mdtime
    g.close()

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
    time = g.dimensions["time"].size
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
    for i in range(time):
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

def hist_to_structure(hist_fname, prefix, i=None, format="vasp"):
    """
    Read from hist file and write to structure file.
    params:
        hist_fname: str
            path to hist file
        prefix: str
            prefix of the output file
        i: int
            index of the structure to write
        format: str
            format of the output file
    """
    #structures, etotal, fcart, stresses = read_structures_from_hist(hist_fname)
    if i is None:
        for i, s in enumerate(structures):
            s.to_ase_atoms.write(filename=f"{prefix}_{i}.{format}")
    else:
        structures[i].to_ase_atoms.write(filename=f"{prefix}_{i}.{format}")



def read_DDB(fname):
    """Read from DDB file and return list of Atoms objects, energies, forces, and stresses.
    :params:
        fname: str
            path to DDB file
    :returns:
        atoms_list: list of Atoms objects
        energies: list of floats (in eV)
        forces: list of numpy arrays (in eV/Ang)
        stresses: list of numpy arrays (in GPa)
    """
    ddb = abiopen(fname)
    structure = ddb.structure
    atoms=structure.to_ase_atoms()
    energy = ddb.total_energy
    force = ddb.cart_forces
    stress = ddb.cart_stress_tensor

    return atoms, energy, force, stress

def DDBs_to_hist(DDB_fnames, hist_fname):
    """ Read from list of DDB files and write to abinit hist netcdf file.
    :params:
        DDB_fnames: list of str
            list of paths to DDB files
        hist_fname: str
            path to hist netcdf file
    """
    atoms_list=[]
    energies=[] 
    forces=[]
    stresses=[]
    for i, DDB_fname in enumerate(DDB_fnames):
        atoms, energy, force, stress = read_DDB(DDB_fname)
        atoms_list.append(atoms)
        energies.append(energy)
        forces.append(force)
        stresses.append(stress)
    write_ase_traj_to_abihist(hist_fname, atoms_list, energies, forces, stresses)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Read from list of DDB files and write to abinit hist netcdf file.")
    parser.add_argument("DDB_fnames", help="list of paths to DDB files", nargs="+")
    parser.add_argument("--output", "-o", help="path to hist netcdf file")
    args = parser.parse_args()
    DDBs_to_hist(args.DDB_fnames, args.output)

if __name__=="__main__":
    main()
