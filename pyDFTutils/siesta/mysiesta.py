import os
from ase.calculators.siesta import Siesta
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.siesta.parameters import format_fdf
from ase.calculators.siesta.parameters import Species
from pyDFTutils.pseudopotential import DojoFinder
import copy


def get_species(atoms, xc, rel='sr'):
    finder = DojoFinder()
    elems = list(dict.fromkeys(atoms.get_chemical_symbols()).keys())
    elem_dict = dict(zip(elems, range(1, len(elems) + 1)))
    pseudo_path = finder.get_pp_path(xc=xc)
    species = [
        Species(symbol=elem,
                pseudopotential=finder.get_pp_fname(elem, xc=xc, rel=rel),
                ghost=False) for elem in elem_dict.keys()
    ]
    return pseudo_path, species


class MySiesta(Siesta):
    def __init__(self,
                 command=None,
                 xc='LDA',
                 spin='UNPOLARIZED',
                 atoms=None,
                 **kwargs):
        finder = DojoFinder()
        elems = list(dict.fromkeys(atoms.get_chemical_symbols()).keys())
        elem_dict = dict(zip(elems, range(1, len(elems) + 1)))
        pseudo_path = finder.get_pp_path(xc=xc)
        if spin == 'spin-orbit':
            rel = 'fr'
        else:
            rel = 'sr'
        species = [
            Species(symbol=elem,
                    pseudopotential=finder.get_pp_fname(elem, xc=xc, rel=rel),
                    ghost=False) for elem in elem_dict.keys()
        ]
        Siesta.__init__(self,
                        pseudo_path=pseudo_path,
                        xc=xc,
                        spin=spin,
                        atoms=atoms,
                        species=species,
                        **kwargs)

    def set_mixer(self,
                  method='pulay',
                  weight=0.05,
                  history=10,
                  restart=25,
                  restart_save=4,
                  linear_after=0,
                  linear_after_weight=0.1):
        pass

    def add_Hubbard_U(self, specy, n=3, l=2, U=0, J=0, rc=0.0, Fermi_cut=0.0):
        if not 'Udict' in self.__dict__:
            self.Udict = dict()
        self.Udict[specy] = {
            'n': n,
            'l': l,
            'U': U,
            'J': J,
            'rc': rc,
            'Fermi_cut': Fermi_cut
        }

    def set_Hubbard_U(self, Udict):
        """
        Udict: {'Fe': {'n':n, 'l':l, 'U':U, 'J', J, 'rc':rc, 'Fermi_cut':Fermi_cut }}
        """
        Ublock = []
        for key, val in Udict.items():
            Ublock.append('  %s %s ' % (key, 1))
            if val['n'] is not None:
                Ublock.append('  n=%s %s' % (val['n'], val['l']))
            else:
                Ublock.append('%s' % (val['l']))
            Ublock.append('  %s  %s' % (val['U'], val['J']))
            if 'rc' in val:
                Ublock.append('  %s  %s' % (val['rc'], val['Fermi_cut']))

        fdf = self['fdf_arguments']
        fdf.update({'LDAU.Proj': Ublock})
        self.set_fdf_arguments(fdf)

    def write_Hubbard_block(self, f):
        pass

    def relax(
        self,
        atoms,
        TypeOfRun='cg',
        VariableCell=True,
        ConstantVolume=False,
        RelaxCellOnly=False,
        MaxForceTol=0.04,
        MaxStressTol=1,
        NumCGSteps=40,
    ):
        pbc = atoms.get_pbc()
        initial_magnetic_moments = atoms.get_initial_magnetic_moments()
        fdf = self['fdf_arguments']
        fdf.update({
            'MD.TypeOfRun': TypeOfRun,
            'MD.VariableCell': VariableCell,
            'MD.ConstantVolume': ConstantVolume,
            'MD.RelaxCellOnly': RelaxCellOnly,
            'MD.MaxForceTol': "%s eV/Ang" % MaxForceTol,
            'MD.MaxStressTol': "%s GPa" % MaxStressTol,
            'MD.NumCGSteps': NumCGSteps,
        })
        self.set_fdf_arguments(fdf)
        self.calculate(atoms)
        self.read(self.prefix + '.XV')
        self.atoms.set_pbc(pbc)
        self.atoms.set_initial_magnetic_moments(initial_magnetic_moments)
        atoms = self.atoms
        self.set_fdf_arguments({
            'MD.NumCGSteps': 0,
        })
        return self.atoms
