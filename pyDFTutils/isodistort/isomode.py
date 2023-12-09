#!/usr/bin/env python3\
# -*- coding: utf-8 -*-
"""
Parser of isodistort output files
"""

import numpy as np
import os
import spglib
from ase import Atoms
from ase.io import read, write
from pyDFTutils.ase_utils.geometry import symbol_number, symnum_to_sym #, vesta_view
from collections import OrderedDict, defaultdict
import re
from itertools import combinations
import copy
import json
import pickle
from itertools import combinations
import dataclasses
import random

def split_symnum(symnum):
    """
    symnum-> sym. eg: Fe1-> Fe
    """
    try:
        a = re.search('[A-Za-z]+', symnum).group()
        b = int(symnum[len(a):])
        return a, b
    except AttributeError:
        raise AttributeError('%s is not a good symbol_number' % symnum)


def parse_mode_name_line(line=None):
    """
    Description: parse a line of mode name,
    eg. P4/mmm[0,0,0]GM1+(a)[Nb2:g:dsp]A1(a) normfactor = 0.00825
    params:
    line: a line of mode name
    Return:
    return a dictionary ret
    ret['kpt']: the kpoint. a list. Note: In python2, 1/2 -> 0 . So it could be wrong in python2.
    ret['kpt_string']: kpt in string. 1/2 is still 1/2
    ret['normfactor']
    ret['label']: eg GM3+
    ret['symmetry']: eg A2u
    fullname:  '[0,0,0]GM1+[Nb2:g:dsp]A1(a)'
    direction: a/b/c
    """
    kpt_string = re.findall(r'\[.*\d\]', line)[0]
    kpt = (eval(kpt_string))
    normfactor_string = re.findall(r'normfactor\s*=\s*(.*)', line)[0]
    normfactor = float(normfactor_string)
    label, symmetry = re.findall(r'\](.*?)\(', line)

    a1 = re.findall(r'(\[.*?)\(', line)[0]
    a2 = re.findall(r'\[.*?\)', line)[1]
    fullname = a1 + a2
    direction = fullname[-2]

    return {
        'kpt': kpt,
        'kpt_string': kpt_string,
        'normfactor': normfactor,
        'label': label,
        'symmetry': symmetry,
        'direction': direction,
        'fullname': fullname,
    }


Subgroup details
1 P1, basis={(1,-1,0),(1,1,0),(0,0,2)}, origin=(-1.00001,-0.99999,-1.50000), s=4, i=192



@dataclasses.dataclass
class SingleMode():
    """
    A single symmetry adapted mode. 
    """
    # fullname: str or None
    fullname: str = None
    displacements: np.ndarray =None
    normfactor: float = None
    label: str = None
    symmetry: str = None
    kpt: list = None
    kpt_string: str = None
    direction: str = None
    directions: str = None

    def __hash__(self):
        return hash(self.fullname)

    def get_displacements(self, factor=1.0):
        """
        Get the displacements of the modes
        """
        return self.displacements*factor

    @staticmethod
    def from_name_line(line):
        """
        Description: parse a line of mode name,
         eg. P4/mmm[0,0,0]GM1+(a)[Nb2:g:dsp]A1(a) normfactor = 0.00825
        params:
        line: a line of mode name
 
        """
        kpt_string = re.findall(r'\[.*\d\]', line)[0]
        kpt = (eval(kpt_string))
        normfactor_string = re.findall(r'normfactor\s*=\s*(.*)', line)[0]
        normfactor = float(normfactor_string)
        label, symmetry = re.findall(r'\](.*?)\(', line)
        longname = line.strip().split()[0]
        a1 = re.findall(r'(\[.*?)\(', line)[0]
        directions = re.findall(r'\((.*?)\)', longname)[0]
        a2 = re.findall(r'\[.*?\)', line)[1]
        direction = longname[-2]
        directions = re.findall(r'\((.*?)\)', longname)[0]
        fullname = f"{a1}({directions}){a2}"
        print(label)
        return SingleMode(fullname=fullname, displacements=None, normfactor=normfactor, label=label, symmetry=symmetry, kpt=kpt, kpt_string=kpt_string, direction=direction, directions=directions)

    def print(self):
        """
        Print the mode
        """
        print(f"{self.fullname=}: {self.normfactor=:f}")
        print(f"{self.mode_name=}")
        print(f"{self.kpt=}")
        print(f"{self.directions=}")
        print(f"{self.mode_name=}")

    def set_displacements(self, displacements):
        """
        Set the displacements of the modes
        """
        self.displacements = np.array(displacements)

    @property
    def mode_name(self):
        """
        Get the short name of the mode
        """
        return f"{self.kpt_string}{self.label}({self.directions})"



@dataclasses.dataclass
class SymmetryMode(dict):
    """
    A set of symmetry adapted modes with same symmetry label and q-point
    It is dirived from a dictionary of single mode objects, and the values are the amplitudes.
    """
    def __init__(self):
        pass

    def add_single_mode(self, mode, amplitude):
        """
        Add a mode to the set of distortions
        """
        if mode.fullname not in self.keys():
            self[mode.fullname] = displacement
        else:
            raise ValueError("Mode already present")

    def get_displacements(self, factor=1.0):
        """
        Get the displacements of the modes
        """
        displacements = 0.0
        for mode, amp in self.items():
            displacements += mode.displacements*amp*factor
        return np.array(displacements)

class MultiModes(dict):
    """
    A set of amplitudes of symmetry adapted modes
    """
    def __init__(self):
        pass

    def add_symmetry_mode(self, symmetry_name, amplitude):
        """
        Add a mode to the set of distortions
        """
        if fullname not in self.keys():
            self[symmetry_name] = amplitude
        else:
            raise ValueError("Mode already present")

    def __str__(self):
        s=""
        for mode, amp in self.items():
            s+="{:s}: {:f}\n".format(mode, amp)
        return s


    def add_single_mode(self, mode, amplitude):
        pass

    def get_amplitudes(self):
        """
        Get the amplitudes of the modes
        """
        return np.array(list(self.values()))

    def get_displacements(self, factors=1.0):
        """
        Get the displacements of the modes
        """
        displacements = 0.0
        for mode, amp in self.items():
            displacements += mode.displacements*amp*factors
        return np.array(displacements)

    def get_mode_symmary(self):
        """
        Get the summary of the modes
        """
        s=""
        for mode, amp in self.items():
            s+=f"{mode:s}: {amp:f}\n"
        return s

        

class IsodistortParser():
    """
    Parser of isodistort output files
    """
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "r") as f:
            self.lines = f.readlines()
        self.all_modes = MultiModes()

        self.undistorted_atoms=self.read_undistorted_supercell()
        self.distorted_atoms=self.read_distorted_supercell()
        self.read_mode_definitions()
        self.read_all_mode_amplitude()


    def read_supercell(self, distorted=False):
        """
        read undisorted supercell
        """
        if distorted:
            name= "Distorted superstructure"
        else:
            name= "Undistorted superstructure"
        inside = False
        sympos = []
        iatom = 0
        # First time: read atoms, the atoms are in order of file
        symdict = {}
        symbols = []
        positions = []
        cellpars = []
        posdict={}
        for iline, line in enumerate(self.lines):
            if line.strip().startswith(name):
                inside = True
                continue
            if inside:
                if line.strip().startswith('a='):
                    segs = line.strip().split(',')
                    for seg in segs:
                        cellpars.append(float(seg.strip().split('=')[1]))
                elif line.strip().startswith('atom'):
                    pass
                elif line.strip() == '':
                    inside = False
                else:
                    symnum, site, x, y, z, occ, displ = line.strip().split()
                    sym, num = split_symnum(symnum)
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    occ = float(occ)
                    displ = float(displ)
                    sympos.append([symnum, x, y, z])
                    symbols.append(sym)
                    posdict[symnum]=np.array([x, y, z])
                    positions.append(np.array([x, y, z]))
                    symdict[symnum] = iatom
                    iatom = iatom + 1
        #symdict = symbol_number(atoms)
        symdict = symbol_number(symbols)
        atoms = Atoms(
            symbols=symbols, scaled_positions=positions, cell=cellpars)
        natom=len(atoms)
        # reorder, by symbol_number
        new_positions=np.zeros((natom,3))
        #symdict = symbol_number(atoms)
        for sn, iatom in symdict.items():
            new_positions[symdict[sn]] = posdict[sn]
            #new_positions[iatom] = positions[symdict[sn]]
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=new_positions,
            cell=cellpars)
        self.natom=natom
        self.symdict=symdict
        return atoms

    def read_undistorted_supercell(self):
        return self.read_supercell(distorted=False)
    
    def read_distorted_supercell(self):
        return self.read_supercell(distorted=True)

    def read_mode_definitions(self):
        """
        read displacive mode definitions. It is a dictionary of mode name and mode.
        mode name: eg. [0,0,0]GM5-[Cs1:d:dsp]Eu(a)
        mode: a numpy array of shape (natom, 3) in primitive cell.
        """
        inside = False
        mode_definitions = {}
        inside_mode = False
        for iline, line in enumerate(self.lines):
            if line.strip().startswith("Displacive mode definitions"):
                inside = True
                continue
            elif line.strip().startswith("Displacive mode amplitudes"):
                inside = False
            elif inside:
                if line.find('normfactor') != -1:  # begining of one mode
                    nameline = line
                    singlemode=SingleMode.from_name_line(nameline)
                    inside_mode = True
                    #deltas = {}
                    mode = np.zeros([self.natom, 3], dtype=float)
                    continue
                if inside_mode:
                    if line.strip() == '':  # end of one mode.
                        singlemode.set_displacements(mode)
                        inside_mode = False
                        mode_definitions[singlemode.fullname] = singlemode
                    elif line.strip().startswith('atom'):
                        pass
                    else:  # a line  of displacement.
                        symnum, x, y, z, dx, dy, dz = line.strip().split()
                        #sym, num = split_symnum(symnum)
                        x, y, z, dx, dy, dz = map(float, (x, y, z, dx, dy, dz))
                        delta = (dx, dy, dz)
                        #deltas[symnum] = delta
                        mode[self.symdict[symnum]] = delta
        self.mode_definitions=mode_definitions
        return mode_definitions

    def read_all_mode_amplitude(self):
        """
        return a dictionary of mode amplitude from isodistort output file.
        The key is the fullname of the mode.
        An example of the section of the output file:
        [0,0,0]GM1+(a)[O1:i:dsp]A1(a)             -0.43243  -0.15289   0.07644
        [0,0,0]GM1+(a)  all                         0.43243   0.15289
        
        [0,0,0]GM2+(a)[O1:i:dsp]A1(a)               0.00000    0.00000   0.00000
        [0,0,0]GM2+(a)  all                         0.00000   0.00000

        Returns:
        summary: a dictionary of mode amplitude. The key is the fullname of the mode, the value is the total amplitude of the mode with the label.
        details: a dictionary of mode amplitude. The key is the fullname of the mode, and the value is a list of amplitude of each mode with the same label.
        """
        summary = dict()
        details = dict()
        inside = False
        for iline, line in enumerate(self.lines):
            if line.strip().startswith("Displacive mode amplitudes"):
                inside = True
                continue
            elif line.strip().startswith("Displacive mode definitions"):
                inside = False
            elif inside:
                if line.strip() == '' or line.strip().startswith('mode'):
                    continue
                elif line.strip().split()[1] == 'all':
                    fullname, _, _, _ = line.strip().split()
                    amp = float(line.strip().split()[2])
                    if abs(amp) > 1e-4:
                        summary[fullname] = amp
                elif line.strip().split()[0] == 'Overall':
                    inside=False
                    continue
                else:
                    print(line)
                    fullname, amp, _, _ = line.strip().split()
                    amp = float(amp)
                    if abs(amp) > 1e-4:
                        details[fullname]=amp
        self.summary_amplitudes=summary
        self.details_amplitudes=details
        return summary, details

    def get_distorted_structure_from_amplitudes(self, amplitudes=None, factors=dict(),default_factor=1.0, summary_amps=None):
        """
        Get the distorted structure from the amplitudes of the modes
        """
        if amplitudes is None:
            amplitudes=self.details_amplitudes

        if summary_amps is not None:
            for key, val in summary_amps.items():
                factors[key] = val/self.summary_amplitudes[key]
            print(f"factors: {factors}")
        atoms=copy.deepcopy(self.read_undistorted_supercell())
        natom=len(atoms)
        disps=np.zeros((natom,3))
        print(self.mode_definitions.keys())
        for fullname, amp in amplitudes.items():
            print(fullname, amp)
            mode=self.mode_definitions[fullname]
            #print(np.linalg.norm(mode.displacements)*mode.normfactor)
            disp_cart = atoms.get_cell().cartesian_positions(mode.displacements)
            #n=np.linalg.norm(disp_cart)
            f=factors.get(mode.label , default_factor)
            print(f"{mode.mode_name=}, {f=}")
            disp=disp_cart * amp*f * mode.normfactor 
            disps+=disp
        #atoms.set_positions(atoms.get_positions()+disps)
        #print(atoms.get_scaled_positions(wrap=False))
        newatoms=copy.deepcopy(atoms)
        newatoms.set_positions(atoms.get_positions()+disps)
        atoms.set_cell(self.read_distorted_supercell().get_cell(), scale_atoms=True)
        return newatoms

            
def reduce_mode(fname="pristine_isodistort.txt", mode_name="R5-", f=1.0):
    myparser = IsodistortParser(fname)
    return myparser.get_distorted_structure_from_amplitudes(factors={mode_name: f})

def reduce_mode_to_amp(fname="pristine_isodistort.txt", ):
    myparser = IsodistortParser(fname)
    return myparser.get_distorted_structure_from_amplitudes(factors={mode_name: f})



def test_isodistort_parser_read_supercell():
    """
    Test isodistort parser of reading undistorred supercell
    """
    
    myparser = IsodistortParser('pristine_isodistort.txt')
    und_atoms=myparser.read_undistorted_supercell()
    dis_atoms=myparser.read_distorted_supercell()
    #make directory for structures
    os.makedirs('structures', exist_ok=True)
    #write structures
    write('structures/undistorted.vasp', und_atoms, vasp5=True, direct=True)
    write('structures/distorted.vasp', dis_atoms, vasp5=True, direct=True)
    #myparser.read_distorted_structure()
    mode_definitions=myparser.read_mode_definitions()
    #print(mode_definitions)
    myparser.read_all_mode_amplitude()
    print(myparser.summary_amplitudes)
    #atoms=myparser.get_distorted_structure_from_amplitudes(amplitudes={"[1/2,1/2,1/2]R5-(a,b,c)[O1:c:dsp]Eu(b)": -0.66608})
    #atoms=myparser.get_distorted_structure_from_amplitudes(factors={"[1/2,1/2,1/2]R5-(a,b,c)": 0.0})
    atoms=myparser.get_distorted_structure_from_amplitudes(factors={"R5-": 0.0})
    #atoms=myparser.get_distorted_structure_from_amplitudes(amplitudes={"[1/2,1/2,0]M2+(a;0;0)[O1:c:dsp]Eu(a)": 1.00083})
    #atoms=myparser.get_distorted_structure_from_amplitudes(amplitudes={"[1/2,1/2,0]M2+(a;0;0)[O1:c:dsp]Eu(a)": 0})
    write('structures/distorted_from_amplitudes.vasp', atoms, vasp5=True, direct=True)


 

def test_isodistort_parser():
    """
    Test isodistort parser
    """
    test_isodistort_parser_read_supercell()

def test_reduce_mode():
    """
    Test reduce mode
    """
    atoms=reduce_mode(fname="pristine_isodistort.txt", mode_name="R5-", f=1.0)
    write('structures/distorted_from_amplitudes.vasp', atoms, vasp5=True, direct=True)


def run_tests():
    """
    Run tests
    """
    test_isodistort_parser()

if __name__ == "__main__":
    #run_tests()
    test_reduce_mode()

