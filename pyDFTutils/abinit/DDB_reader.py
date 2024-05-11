from abipy.abilab import abiopen
from pyDFTutils.abinit.abihist_file import write_ase_traj_to_abihist

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
