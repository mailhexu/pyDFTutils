#! /usr/bin/env python
from .vasp_utils import get_electrons
from pyDFTutils.ase_utils.chemsymbol import get_symdict
from ase.io import read
import matplotlib.pyplot as plt
import os
import numpy as np

def run_bader(filename='CHGCAR'):
    if os.path.exists('AECCAR0') and os.path.exists('AECCAR2'):
        if not os.path.exists('CHGCAR_sum'):
            os.system('chgsum AECCAR0 AECCAR2')
        os.system('bader %s -ref CHGCAR_sum'%filename)
    else:
        os.system('bader %s'%filename)

def bcf_parser(filename='ACF.dat'):
    """
    parse the ACF.dat file and return the {iatom:charge} pairs.
    iatom starts from 0
    """
    text=open(filename,'r').readlines()
    val_dict=dict()
    for line in text[2:-4]:
        i,x,y,z,val,min_dist,volume= [float(x) for x in line.strip().split()]
        iatom=int(i)-1
        val_dict[iatom]=val
    return val_dict

def get_charge(filename='ACF.dat'):
    """
    net charge of the ions.
    """
    val_dict=bcf_parser(filename=filename)
    atoms=read('POSCAR')
    symbols=atoms.get_chemical_symbols()
    elec_dict=get_electrons()
    charges=[]
    for i in val_dict:
        symb=symbols[i]
        elec=elec_dict[symb]
        charges.append(-(val_dict[i]-elec))
    return charges

def get_electron_number(filename='ACF.dat'):
    """
    numbers of electrons of the ions.
    """
    val_dict=bcf_parser(filename=filename)
    return [val_dict[i] for i in range(len(val_dict))]

def get_total_BCF_file(filename='CHGCAR'):
    """
    bader filename
    """
    run_bader(filename=filename)

def get_spin_BCF_file(filename='CHGCAR'):
    """
    split the CHGCAR file to up and down, and do bader analysises, respectively.
    """
    #  chgsplit : -> CHGCAR_up & CHGCAR_down
    os.system('chgsplit %s'%filename)
    #  run bader
    run_bader(filename='CHGCAR_up')
    os.rename('ACF.dat','ACF_up.dat')

    run_bader(filename='CHGCAR_down')
    os.rename('ACF.dat','ACF_down.dat')


def get_atom_charges(sym_numbers=None,run_bader=True):
    """
    get bader charge of atoms.
    sym_numbers: list of sym_number
    run_bader: (bool) run bader first or not.
    """
    result=[]
    symdict=get_symdict()

    if sym_numbers is None:
        sym_numbers=list(get_symdict().keys())

    if run_bader:
        get_total_BCF_file()

    #if sym_numbers is None:
    #    return get_charge(filename='BCF.dat')
    for i in sym_numbers:
        result.append(get_charge(filename='ACF.dat')[symdict[i]])
    return result


def get_atom_electrons(sym_numbers=None,run_bader=False,spin=None):
    """
    get bader numbers electrons of atoms.
    sym_numbers: list of sym_number. None for all the atoms.
    run_bader: (bool) run bader first or not.
    spin=None| 'up' |'down': if None: total charge. else: charge of the spin type.
    """


    result=[]
    symdict=get_symdict()

    if sym_numbers is None:
        sym_numbers=get_symdict()


    if run_bader:
        if spin is None:
            get_total_BCF_file()
        else:
            get_spin_BCF_file()

    if spin is None:
        for i in sym_numbers:
            result.append(get_electron_number(filename='ACF.dat')[symdict[i]])
    elif spin=='up':
        for i in sym_numbers:
            result.append(get_electron_number(filename='ACF_up.dat')[symdict[i]])
    elif spin=='down':
        for i in sym_numbers:
            result.append(get_electron_number(filename='ACF_down.dat')[symdict[i]])
    else:
        raise ValueError('spin should be None |"up"|"down"')

    return result

def get_net_magnetic_moments(sym_numbers=None,run_bader=False,spin=None):
    if run_bader:
        get_spin_BCF_file()

    up_charges=get_atom_electrons(sym_numbers=sym_numbers,spin='up')
    down_charges=get_atom_electrons(sym_numbers=sym_numbers,spin='down')
    delta=np.asarray(up_charges)-np.asarray(down_charges)
    return delta


def get_atom_zpos(sym_numbers=None):
    positions=read('POSCAR').get_positions()
    result=[]
    symdict=get_symdict()
    if sym_numbers is None:
        sym_numbers=list(symdict.keys())

    for i in sym_numbers:
        result.append(positions[symdict[i]][2])
    return result

def sym_number_to_sym(s):
    import re
    return re.findall('[a-zA-Z]+',s)[0]

def plot_atom_charges(sym_numbers):
    positions=get_atom_zpos(sym_numbers=sym_numbers)
    charges=get_atom_charges(sym_numbers=sym_numbers)
    print(positions)
    plt.xlabel('position (ang)')
    plt.ylabel('valence')
    plt.plot(positions,charges,'o-',label=sym_number_to_sym(sym_numbers[0]))


def test():
    print(get_atom_charges(['Fe1','Fe2','Fe3']))

def test_b():
    names=[]
    for i in [12,1,4,5,8,9]:
        names.append('Bi%s'%(i))
    plot_atom_charges(names)

    names=[]
    for i in [4,3,8,7,12,11]:
        names.append('Fe%s'%(i))
    plot_atom_charges(names)

    names=[]
    for i in [4,1,8,5,12,9]:
        names.append('Ti%s'%(i))
    plot_atom_charges(names)

    names=[]
    for i in [2,3,6,7,10,11]:
        names.append('Sr%s'%(i))
    plot_atom_charges(names)


    names=[]
    for i in [36,2,12,14,24,26,42,45,54,57,66,69]:
        names.append('O%s'%(i))

    plot_atom_charges(names)

    names=[]
    for i in [9,8,21,20,33,32]:
        names.append('O%s'%(i))

    plot_atom_charges(names)

    """
    names=[]
    for i in [18,15,20,17,22]:
        names.append('O%s'%(i))
    plot_atom_charges(names)
    """
    plt.legend()
    plt.show()
    test()
if __name__=='__main__':
    #bader
    net_charges=get_atom_charges(run_bader=True)
    net_mags=get_net_magnetic_moments(run_bader=True)

    symnum=get_symdict(filename='CONTCAR')
    with open('bader.txt' ,'w') as myfile:
        myfile.write('#symnum\tnetcharge\tnetmagmom\n')
        for sm in symnum:
            myfile.write('%s\t%s\t%s\n'%(sm,net_charges[symnum[sm]],net_mags[symnum[sm]]))
