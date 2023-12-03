from pyDFTutils.ase_utils import get_refined_atoms, symbol_number
import numpy as np
from ase.io import read,write 
from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances

def refine(fname="relaxed.vasp", origin="Y1"):
    atoms=read(fname)

    # generate refined atoms
    ratoms_uc=get_refined_atoms(atoms, symprec=2)
    ncell=[[1, -1, 0], [1, 1, 0],[0,0,2]]
    ratoms = make_supercell(ratoms_uc, ncell)

    # put the atoms into the same cell as the reference atoms
    symbols=atoms.get_chemical_symbols()
    sdict=symbol_number(symbols)

    rsdict=symbol_number(ratoms.get_chemical_symbols())

    # move the origin of atoms to 0.
    xred=atoms.get_scaled_positions()
    xred-=xred[sdict[origin]]
    atoms.set_scaled_positions(xred)

    # move the refined atoms origin to 0. 
    rxred=ratoms.get_scaled_positions()
    rxred-=rxred[rsdict[origin]]
    ratoms.set_scaled_positions(rxred)

    # set the refined atoms cell to be the same as the non-refined atoms
    ratoms.set_cell(atoms.get_cell(), scale_atoms=True)

    # match the scaled positions of the refined atoms to the scaled positions of the non-refined atoms
    match_scaled_positions(ratoms, atoms)
    
    # find the indices of the atoms close to Mg1

    #scmat=[[1,-1,0],[1,1,0],[0,0,2]]
    scmat=[[2,0,0],[0,2,0],[0,0,2]]
    sc_atoms=make_supercell(atoms, scmat)
    sc_ratoms=make_supercell(ratoms, scmat)

    match_scaled_positions(sc_ratoms, sc_atoms)
    indices=find_indices_of_neighbours(sc_atoms, origin=origin, cutoff=5)
    print(indices)
    # replace the positions of the atoms close to Mg1 with the positions of the refined atoms
    sc_ratoms=replace_positions(sc_atoms, sc_ratoms, indices)
    return sc_ratoms

def mic(x):
    """
    Minimum image convention
    """
    return x-np.round(x)


def match_scaled_positions(atoms, ref_atoms, origin=None, origin_ref=None):
    """
    and then shift the scaled positions of atoms to match the scaled positions of ref_atoms, considering the periodic boundary condition, so that the two structures are aligned. 
    The scaled positions of atoms are modified so that the distance between the two structures is minimized.
    The order of the atoms are also changed to match the order of ref_atoms.
    """
    xred=atoms.get_scaled_positions(wrap=False)
    xred_ref=ref_atoms.get_scaled_positions(wrap=False)

    xcart=atoms.get_positions()
    xcart_ref=ref_atoms.get_positions()

    cell=atoms.get_cell().array
    cell_ref=ref_atoms.get_cell().array
    symbols=atoms.get_chemical_symbols()
    symbols_ref=ref_atoms.get_chemical_symbols()


    # put the atoms into the same cell as the reference atoms
 
    sdict=symbol_number(symbols)
    rsdict=symbol_number(symbols_ref)

    # move the origin of atoms to 0.
    if origin is not None:
        xred=atoms.get_scaled_positions(wrap=False)
        xred-=xred[sdict[origin]]
        atoms.set_scaled_positions(xred)

    # move the refined atoms origin to 0. 
    if origin_ref is not None:
        rxred=ref_atoms.get_scaled_positions()
        rxred-=rxred[rsdict[origin]]
        ref_atoms.set_scaled_positions(rxred)
    
    


    # for each atom, find the closest atom in ref_atoms
    # and shift the scaled positions of atoms to match the scaled positions of ref_atoms
    # considering the periodic boundary condition
    xred_new=np.zeros_like(xred)
    symbols_new=np.zeros_like(symbols)
    distances=get_distances(xcart, xcart_ref, cell=cell_ref, pbc=True)

    for i, d in enumerate(distances[1]):
        #print(distances[1][i])
        j=np.argmin(d)
        xred_new[j]=xred_ref[j]+ mic(xred[i]-xred_ref[j])
        symbols_new[j]=symbols[i]
    print(symbols_new)
    atoms.set_chemical_symbols(symbols_new)
    atoms.set_scaled_positions(xred_new)
    return atoms


def replace_positions(atoms1, atoms2, indices):
    """
    replace the positions of atoms1 with the positions of atoms2 for the atoms with the given indeces
    """
    xred=atoms1.get_scaled_positions()
    xred_ref=atoms2.get_scaled_positions()
    #print(xred[indices])
    #print(xred_ref[indices])
    xred[indices]=xred_ref[indices]
    atoms1.set_scaled_positions(xred)
    return atoms1

def find_indices_of_neighbours(atoms, origin="Mg1", cutoff=3.0):
    """
    find the indices of the atoms that are within the cutoff distance of the atom with the given origin
    """
    symbols=atoms.get_chemical_symbols()
    sdict=symbol_number(symbols)
    distances=atoms.get_distances(sdict[origin], indices=None, mic=True)
    indices=np.where(distances<cutoff)[0]
    return indices


   


def main():
    atoms=refine("relaxed.vasp")
    write("refined.vasp", atoms, vasp5=True, sort=True, direct=True)


#if __name__=="__main__":
#    main()

