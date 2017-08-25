#! /usr/bin/env python
from ase import Atoms
from pyDFTutils.ase_utils import to_smallest_positive_pos,cut_lattice ,scaled_pos_to_pos
force_in_cell = to_smallest_positive_pos
from  ase.lattice.cubic import SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.geometry.cell import cellpar_to_cell
from numpy import array,sqrt
import numpy as np
from ase.lattice.spacegroup import crystal
import re

# Define cubic perovskite
class PerovskiteCubicFactory(SimpleCubicFactory):
    "A factory for creating perovskite (ABO3) lattices"

    xtal_name = 'cubic perovskite'
    bravais_basis = [
                     [0.,0.,0.],
                     [0.5,0.5,0.5],
                     [0.5,0.5,0.],[0.,0.5,0.5],[0.5,0.,0.5]]
    element_basis = (0,1,2,2,2)

PerovskiteCubic=PerovskiteCubicFactory()
PerovskiteCubic.__doc__="""
Usage:
eg. STO=PerovskiteCubic(['Sr','Ti','O'],latticeconstant=3.905)
"""
# Define double perovskite
class DoublePerovskiteFactory(SimpleTetragonalFactory):
    "A factory for creating double perovskite ((A1,A2)BO3) lattices"
    def __init__(self,oxygen=6.):
        self.oxygen=oxygen
        self.xtal_name = 'double perovskite'
        if self.oxygen==6:
            self.bravais_basis = [[0.5,0.5,0.25],[0.5,0.5,0.75],
                                  [0.,0.,0.],[0.,0.,0.5],
                                  [0.5,0.,0.],[0.,0.5,0.],[0.,0.,0.25],
                                  [0.5,0.,0.5],[0.,0.5,0.5],[0.,0.,0.75]]
            self.element_basis = (0,1,2,2,3,3,3,3,3,3)
        elif self.oxygen==5:
            self.bravais_basis = [[0.5,0.5,0.25],[0.5,0.5,0.75],
                                  [0.,0.,0.],[0.,0.,0.5],
                                  [0.5,0.,0.],[0.,0.5,0.],
                                  [0.5,0.,0.5],[0.,0.5,0.5],[0.,0.,0.75]]
            self.element_basis = (0,1,2,2,3,3,3,3,3)
        else:
            raise ValueError("oxygen keyword only accepts values 5 or 6")
DoublePerovskite=DoublePerovskiteFactory()


class DoublePerovskiteFactory_3(SimpleTetragonalFactory):
    """
    gen (A2B)C3O9 type double perovksite lattice
    symbols: [A, B, C, O]
    cell: [a,b,c]
    """
    def __init__(self):
        self.xtal_name=r'double perovksite (0.33)'
        bravais_basis_A=[(0,0,1.0/3*i) for i in [0,1]]
        bravais_basis_B=[(0,0,1.0/3*i) for i in [2]]
        bravais_basis_C=[(0.5,0.5,1.0/3*i+1.0/6) for i in [0,1,2]]
        bravais_basis_O=[(0.5,0.5,1.0/3*i) for i in [0,1,2]]+[(0.5,0,1.0/3*i+1.0/6) for i in [0,1,2]]+[(0,0.5,1.0/3*i+1.0/6) for i in [0,1,2]]
        self.bravais_basis=bravais_basis_A+bravais_basis_B+bravais_basis_C+bravais_basis_O
        self.element_basis=(0,0,1,2,2,2,3,3,3,3,3,3,3,3,3)

DoublePerovskite_3=DoublePerovskiteFactory_3()



class R3c_factory(TriclinicFactory):
    """
    R3c factory eg. BFO ### NOT IMPLEMENTED
    """
    def __init__(self,latticeconstant=(1,60),e_base=0.25,**kwargs):
        a=latticeconstant[0]
        alpha=latticeconstant[1]
        TriclinicFactory.__init__(latticeconstant={'a':a,'b':a,'c':a ,'alpha':alpha,'beta':alpha,'gamma':alpha})
        raise Exception("Not implemented yet, don't want to do that")


def gen_pnma():
    a=5.742
    b=7.668
    c=5.532
    atoms=crystal(['La','Mn','O','O'],[(0.549,0.25,0.01),(0,0,0),(-0.014,0.25,-0.07),(0.309,0.039,0.244)],spacegroup='pnma',cellpar=[a,b,c,90,90,90])
    atoms.set_pbc([True,True,True])
    mag=[0]*20
    mag[4:8]=[1,-1,1,-1]
    atoms.set_initial_magnetic_moments(mag)
    return atoms


def gen_pbnm():
    a=5.742
    b=7.668
    c=5.532
    atoms=crystal(['La','Mn','O','O'],[(0.549,0.25,0.01),(0,0,0),(-0.014,0.25,-0.07),(0.309,0.039,0.244)],spacegroup='pbnm',cellpar=[a,b,c,90,90,90])
    atoms.set_pbc([True,True,True])
    mag=[0]*20
    mag[4:8]=[1,-1,1,-1]
    atoms.set_initial_magnetic_moments(mag)
    return atoms



def R3c_builder(a,alpha,symbol_list,basis):
    """
    a, alpha: as they are
    symbols: a list. eg. ['Bi','Fe',O]
    basis:
    """

    symbols=symbol_list[0]*2+symbol_list[1]*2+symbol_list[2]*6

    x0=basis[0][0]
    scaled_positions=[(x0,x0,x0)]
    scaled_positions.append(force_in_cell((0.5+x0,0.5+x0,0.5+x0)))

    x1 =basis[1][0]
    scaled_positions.append(force_in_cell((x1,x1,x1)))
    scaled_positions.append(force_in_cell((0.5+x1,0.5+x1,0.5+x1)))

    x,y,z=basis[2]
    scaled_positions.append(force_in_cell(array([x,y,z])))
    scaled_positions.append(force_in_cell(array([z,x,y])))
    scaled_positions.append(force_in_cell(array([y,z,x])))
    ####NOTE: not just the above +0.5 but with a rotation.
    scaled_positions.append(force_in_cell(array([y,x,z])+0.5))
    scaled_positions.append(force_in_cell(array([x,z,y])+0.5))
    scaled_positions.append(force_in_cell(array([z,y,x])+0.5))

    atoms=Atoms(symbols=symbols,scaled_positions=scaled_positions,cell=cellpar_to_cell([a,a,a,alpha,alpha,alpha]))
    return atoms

def cut_R3c_222(atoms):
    cell=atoms.get_cell()
    vec0=np.dot(np.array([0.5,0.5,0.5]),cell)
    vec1=np.dot(np.array([1,0,0]),cell)
    vec2=np.dot(np.array([0,1,0]),cell)
    vec3=np.dot(np.array([0,0,1]),cell)

    new_cell=array([vec1-vec0,vec2-vec0,vec3-vec0])*2
    #print cell_to_cellpar(new_cell)
    #print new_cell

    new_atoms=cut_lattice(atoms,new_cell,nrepeat=4)
    return new_atoms


def cut_cubic_s2s22(atoms):
    """
    cut a cubic -> sqrt(2)*sqrt(2)*2
    """
    cell=atoms.get_cell()
    vec0=np.dot(np.array([0,0,0]),cell)
    vec1=np.dot(np.array([1,-1,0]),cell)
    vec2=np.dot(np.array([1,1,0]),cell)
    vec3=np.dot(np.array([0,0,2]),cell)

    new_cell=array([vec1-vec0,vec2-vec0,vec3-vec0])*2
    #print cell_to_cellpar(new_cell)
    #print new_cell

    new_atoms=cut_lattice(atoms,new_cell,nrepeat=4)
    return new_atoms


def cut_R3c_22(atoms):
    """
    This cut 2*2*2->sqrt(2)*sqrt(2)*2 lattice
    """
    cell=atoms.get_cell()
    vec0=scaled_pos_to_pos([0,0.5,1],cell)
    vec1=scaled_pos_to_pos([0.5,0,1],cell)
    vec2=scaled_pos_to_pos([0.5,1,1],cell)
    vec3=scaled_pos_to_pos([0,0.5,0],cell)
    new_cell=np.asarray([vec1-vec0,vec2-vec0,vec3-vec0])
    new_atoms=cut_lattice(atoms,new_cell,nrepeat=2)
    cell=new_atoms.get_cell()
    #vec=scaled_pos_to_pos([0,0.5,0],cell)
    #new_atoms=translation(new_atoms,vec)
    return new_atoms


def R_3c_builder(a,alpha,symbol_list,basis=[(0,0,0),(0.227,0.227,0.227),(0.542,0.943,0.397)]):
    """
    a, alpha: as they are
    symbols: a list. eg. ['Bi','Fe',O]
    basis:
    """

    symbols=symbol_list[0]*2+symbol_list[1]*2+symbol_list[2]*6

    x0=basis[0][0]
    scaled_positions=[(x0,x0,x0)]
    scaled_positions.append(force_in_cell((0.5+x0,0.5+x0,0.5+x0)))

    x1 =basis[1][0]
    scaled_positions.append(force_in_cell((x1,x1,x1)))
    scaled_positions.append(force_in_cell((0.5+x1,0.5+x1,0.5+x1)))

    x,y,z=basis[2]
    scaled_positions.append(force_in_cell(array([x,y,z])))
    scaled_positions.append(force_in_cell(array([z,x,y])))
    scaled_positions.append(force_in_cell(array([y,z,x])))
    ####NOTE: not just the above +0.5 but with a rotation.
    scaled_positions.append(force_in_cell(array([x,y,z])+0.5))
    scaled_positions.append(force_in_cell(array([z,x,y])+0.5))
    scaled_positions.append(force_in_cell(array([y,z,x])+0.5))

    atoms=Atoms(symbols=symbols,scaled_positions=scaled_positions,cell=cellpar_to_cell([a,a,a,alpha,alpha,alpha]))
    raise NotImplementedError('Implementation Wrong, try to use sth instead , e.g. LaAlO3')
    return atoms

class NaClPrimFactory(TriclinicFactory):
    bravais_basis = [[0, 0, 0], [0.5, 0.5, 0.5]]
    element_basis = (0, 1)


# Rocksalts
#-----------------
rocksalt_prim = NaClPrimFactory()


def gen_rocksalt_prim(name, latticecostant):
    """
    generate primitive cell of rocksalt structure.
    e.g.
      gen_rocksalt_prim('NaCl', 2.8)
    """
    elems = re.findall('[A-Z][a-z]*', name)
    a = latticecostant
    atoms = rocksalt_prim(elems, latticeconstant=(a, a, a, 60, 60, 60))
    return atoms

#-----------------

def test_r3c():
    atoms=R3c_builder(5.52,59.84,symbol_list=['Bi','Fe','O'],basis=[(0,0,0),(0.227,0.227,0.227),(0.542,0.943,0.397)])
    print(atoms.positions)
    atoms=cut_R3c_222(atoms)
    atoms=cut_R3c_22(atoms)
    print(atoms.get_chemical_symbols())
    return atoms



if __name__=='__main__':
    test_r3c()
