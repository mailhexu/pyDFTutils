#!/usr/bin/env abinit
"""
abinit calculator helper functions. Set default values for the calculations.
See also ase.calculator.abinit, ase_utils.myabinit
"""
from ase.units import eV, Ha, Bohr
from ase_utils.myabinit import Abinit
from ase.data import atomic_masses, atomic_numbers, chemical_symbols, atomic_masses
from ase.atoms import Atoms
import numpy as np


def abinit_calculator(ecut=35 * Ha,
                      xc='LDA',
                      nk=8,
                      mag_order='PM',
                      is_metal=False,
                      pps='ONCV',
                      **kwargs):
    """
    default vasp calculator.

    Parameters
    ------------
    ecut: float
        energy cutoff
    xc: string
        XC functional
    nk: int
        k-point mesh  nk*nk*nk. use kpts=[nk1,nk2,nk3] for unequal nk's.
    mag_order: string
        FM|PM|A|G|C|
    is_metal: bool
        is it metallic? it unkown, set is_metal to True.
    **kwargs:
        args passed to myvasp.set function.

    Returns
    ------------
    A abinit calculator object. derived from ase.calculator.abinit
    """
    calc = Abinit(
        label='abinit',
        xc=xc,
        accuracy=5,
        ecut=ecut * eV,
        kpts=[nk, nk, nk],
        gamma=False,
        #chksymbreak=0,
        pppaths=['/home/hexu/.local/pp/abinit/'],
        pps=pps,
        chksymbreak=0,
        pawecutdg=ecut * 1.8 * eV,
        diemac=5.4,
        diemix=0.7,
        #iprcel=45,
        autoparal=1, )
    if mag_order == 'PM' and is_metal:
        calc.set(occopt=7, nsppol=1, nspinor=1, nspden=1, diemac=1e5)
        calc.set(tsmear=0.001 * Ha)
    elif mag_order == 'PM' and not is_metal:
        calc.set(occopt=1, nsppol=1, nspinor=1, nspden=1)
    elif mag_order == 'FM' or is_metal:
        calc.set(occopt=4, nsppol=2)
        calc.set(tsmear=0.001 * Ha, diemac=1e5)
    else:
        calc.set(occopt=1, nsppol=1, nspinor=1, nspden=2)
    calc.set(iscf=17, nstep=50)
    calc.set(**kwargs)
    return calc


class DDB_reader():
    def __init__(self, fname):
        """
        class for reading DDB files.
        """
        self.fname = fname

    def read_atoms(self):
        """
        read atomic structure from DDB file.
        Returns:
        -----------
        ase.atoms object.
        """
        with open(self.fname) as myfile:
            for line in myfile:
                if line.strip().startswith('natom'):
                    self.natom = int(line.strip().split()[-1])
                if line.strip().startswith("ntypat"):
                    ntypat = int(line.strip().split()[-1])
                if line.strip().startswith("acell"):
                    acell = [
                        float(s.replace('D', 'E')) * Bohr
                        for s in line.strip().split()[1:4]
                    ]
                if line.strip().startswith("typat"):
                    typat = [int(s) for s in line.strip().split()[1:]]
                if line.strip().startswith("znucl"):
                    znucl = [
                        int(float(s.replace('D', 'E')))
                        for s in line.strip().split()[1:4]
                    ]
                if line.strip().startswith("rprim"):
                    rprim0 = [
                        float(s.replace('D', 'E')) * acell[0]
                        for s in line.strip().split()[1:4]
                    ]
                    line = myfile.next()
                    rprim1 = [
                        float(s.replace('D', 'E')) * acell[1]
                        for s in line.strip().split()
                    ]
                    line = myfile.next()
                    rprim2 = [
                        float(s.replace('D', 'E')) * acell[2]
                        for s in line.strip().split()
                    ]

                if line.strip().startswith("xred"):
                    spos = np.zeros((
                        self.natom,
                        3, ))
                    spos[0] = [
                        float(s.replace('D', 'E'))
                        for s in line.strip().split()[-3:]
                    ]
                    for i in range(1, self.natom):
                        line = myfile.next()
                        print line
                        spos[i] = [
                            float(s.replace('D', 'E'))
                            for s in line.strip().split()[-3:]
                        ]
            numbers = [znucl[i - 1] for i in typat]
            self.symbols = [chemical_symbols[i] for i in numbers]
            self.masses = [atomic_masses[i] for i in numbers]
            self.cell = [rprim0, rprim1, rprim2]
            print self.symbols
            self.atoms = Atoms(self.symbols, positions=spos, cell=self.cell)
            return self.atoms

    def read_2DE_DDB(self, mat=True):
        """
        Read total energy 2nd derivatives from DDB files.

        Parameters:
        -------------
        fname: string
          The name of the DDB file.

        Returns:
        -------------
        dict dds.
        The keys are tuples: (idir1, ipert1, idir2, ipert2), values are complex numbers.
        idir, idir2 are the directions (1,2,3), ipert1, ipert2 are perturbations.
        ipert= 1..natom are atomic displacements;
        natom+1: ddk;
        natom+2: electric field;
        natom+3: uniaxial strains;
        natom+4: shear strain.
        """
        dds = {}
        with open(self.fname) as myfile:
            for line in myfile:
                if line.find('**** Database of total energy derivatives ****'
                             ) != -1:
                    l = myfile.next()
                    nblock = int(l.strip().split()[-1])
                    #print "Nblock:",nblock
                    myfile.next()
                    l = myfile.next()
                    nelem = int(l.strip().split()[-1])
                    #print nelem
                    l = myfile.next()
                    self.qpt = [
                        float(x.replace('D', 'E'))
                        for x in l.strip().split()[1:4]
                    ]
                    #print qpts
                    for i in range(nelem):
                        try:
                            l = myfile.next()
                            idir1, ipert1, idir2, ipert2 = [
                                int(x) for x in l.strip().split()[0:4]
                            ]
                            realval, imageval = [
                                float(x.replace('D', 'E'))
                                for x in l.strip().split()[4:6]
                            ]
                            dds[(idir1, ipert1, idir2,
                                 ipert2)] = realval + 1j * imageval
                        except:
                            pass
        self.dynamic_matrix_dict = dds
        return self.dynamic_matrix_dict

    def get_dynamic_matrix(self):
        """
        Parameters:
        ------------
        dds: output or read_2DE_DDB
        Returns:
        ------------
        2D matrix. the indices means: (ipert,idir) = (1,1) (1,2) (1,3) (2,1) ...(natom,3)
        """
        natom = len(self.atoms)
        dynmat = np.zeros((natom * 3, natom * 3), dtype=complex)
        for ipert1 in range(natom):
            for idir1 in range(3):
                for ipert2 in range(natom):
                    for idir2 in range(3):
                        dynmat[ipert1 * 3 + idir1, ipert2 * 3 +
                               idir2] = self.dynamic_matrix_dict[(
                                   idir1+1, ipert1+1, idir2+1, ipert2+1)]
        return dynmat



#myreader = DDB_reader("../test/BaTiO3_bak/abinito_DS2_DDB")
#print myreader.read_atoms().get_positions()
#myreader.read_2DE_DDB()
#print myreader.get_dynamic_matrix()
