#!/usr/bin/env python
import os
from psp import atomconf_to_ionconf, gen_ion_conf_dict
from data.ONCV_PBEsol_conf import ONCV_PBEsol_conf
"""
This module controls the main calculating process.
"""


class task(object):
    def __init__(self, atoms=None, calculator=None):
        """
        Organize the tasks for analyzing the related properties of a structure.

        Parameters
        ----------
        atoms: ase.atoms
            The structure to calculate.
        """
        self.atoms = atoms
        self.calculator = calculator
        self.cwd=os.getcwd()

    def set_atoms(self, atoms):
        """
        set the atom structure to calculate.

        Parameters
        ----------
        atoms: ase.atoms
            The structure to calculate. Should be a ase.atoms object.
        """
        self.atoms = atoms
        if atoms is not None:
            self.atoms.set_calculator(self.calculator)

    def set_calculator(self, calculator):
        """
        set the atom structure to calculate.

        Parameters
        ----------
        atoms: ase.calculator
            The calculator. Should be a ase.calculator object.
        """
        self.calculator = calculator
        if self.atoms is not None:
            self.atoms.set_calculator(self.calculator)

    def set_work_dir(self, wd='./'):
        """
        set the working directory.
        """
        self.cwd = os.getcwd()
        self.wd = wd
        if not os.path.exists(wd):
            os.mkdir(wd)
        os.chdir(wd)

    def close(self):
        os.chdir(self.cwd)

    def relax(self, **kwargs):
        """
        Relax the structure.

        Parameters
        -----------

        Returns
        -----------
        ase.atoms: the relaxed structure.
        """
        #self.calculator.set(**kwargs)
        atoms = self.calculator.relax_calculation(atoms=self.atoms,**kwargs)
        self.atoms = atoms
        return self.atoms

    def scf(self, dos=True, **kwargs):
        """
        Do scf calculation.
        """
        self.calculator.set(**kwargs)
        atoms = self.calculator.scf_calculation(atoms=self.atoms)

    def ldos(self, pdos=True, **kwargs):
        """
        Do DOS calculation, analyze the data, and plot the figures for density of states.
        """
        self.calculator.set(**kwargs)
        atoms = self.calculator.ldos_calculation(atoms=self.atoms)

    def phonon(self, **kwargs):
        """
        Do phonon calculation.

        Parameters
        ---------------
        **kwargs:
          kwargs are passed to abinit.calculate_phonon.
          Defaults are:
        efield=True,
        qpts=[2, 2, 2],
        tolwfr=1e-23,
        tolvrs=1e-9,
        prtwf=0,
        postproc=True,
        ifcout=10240,
        rfasr=1,
        plot_band=True

        """
        self.calculator.calculate_phonon(self.atoms, **kwargs)

    def wannier(self,
                atoms=None,
                wannier_input=None,
                nband= None,
                **kwargs):
        """
        wannier calculation.
        Parameters:
        wannier_input.
        """
        # read number of bands, Fermi energy from output file.
        # prepare the occupied orbitals for initial projection.
        #nbands = self.calculator.read_number_of_bands()
        #econf = gen_ion_conf_dict(valence_dict, econf_dict)
        wa=wannier_input
        wannier_input.write_input()
        if nband is None:
            self.calculator.set(nband=wa.get_nwann())
        else:
            self.calculator.set(nband=nband)
        self.calculator.set(**kwargs)
        self.calculator.scf_calculation(self.atoms)
 
 
