import os
import math
import numpy as np
from ase.calculators.siesta import Siesta
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.siesta.parameters import format_fdf
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
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


def cart2sph(vec):
    x, y, z = vec
    r = np.linalg.norm(vec)               # r
    if r < 1e-10:
        theta, phi = 0.0, 0.0
    else:
        # note that there are many conventions, here is the ISO convention.
        phi = math.atan2(y, x) * 180/math.pi                          # phi
        theta = math.acos(z/r) * 180/math.pi                        # theta
    return r, theta, phi


class MySiesta(Siesta):
    def __init__(self,
                 atoms=None,
                 command=None,
                 xc='LDA',
                 spin='non-polarized',
                 ghosts=[],
                 input_basis_set={},
                 **kwargs):
        if atoms is not None:
            finder = DojoFinder()
            elems = list(dict.fromkeys(atoms.get_chemical_symbols()).keys())
            elem_dict = dict(zip(elems, range(1, len(elems) + 1)))
            symbols = atoms.get_chemical_symbols()

            # ghosts
            ghost_symbols = [symbols[i] for i in ghosts]
            ghost_elems = list(dict.fromkeys(ghost_symbols).keys())
            tags = [1 if i in ghosts else 0 for i in range(len(atoms))]
            atoms.set_tags(tags)

            pseudo_path = finder.get_pp_path(xc=xc)
            if spin == 'spin-orbit':
                rel = 'fr'
            else:
                rel = 'sr'
            species = []
            for elem in elem_dict.keys():
                if elem not in input_basis_set:
                    bselem = 'DZP'
                else:
                    bselem = PAOBasisBlock(input_basis_set[elem])
                species.append(Species(symbol=elem,
                                       pseudopotential=finder.get_pp_fname(
                                           elem, xc=xc, rel=rel),
                                       basis_set=bselem,
                                       ghost=False))
            for elem in ghost_elems:
                species.append(
                    Species(symbol=elem,
                            pseudopotential=finder.get_pp_fname(
                                elem, xc=xc, rel=rel),
                            tag=1,
                            ghost=True))

        Siesta.__init__(self,
                        xc=xc,
                        spin=spin,
                        atoms=atoms,
                        pseudo_path=pseudo_path,
                        species=species,
                        **kwargs)

    def set_fdf_arguments(self, fdf_arguments):
        self['fdf_arguments'].update(fdf_arguments)

    def set_mixer(self,
                  method='pulay',
                  weight=0.05,
                  history=10,
                  restart=25,
                  restart_save=4,
                  linear_after=0,
                  linear_after_weight=0.1):
        pass

    def update_fdf_arguments(self, fdf_arguments):
        fdf = self['fdf_arguments'].update(fdf_arguments)

    def add_Hubbard_U(self,
                      specy,
                      n=3,
                      l=2,
                      U=0,
                      J=0,
                      rc=0.0,
                      Fermi_cut=0.0,
                      scale_factor='0.95'):
        if not 'Udict' in self.__dict__:
            self.Udict = dict()
        self.Udict[specy] = {
            'n': n,
            'l': l,
            'U': U,
            'J': J,
            'rc': rc,
            'Fermi_cut': Fermi_cut,
            'scale_factor': scale_factor
        }
        self.set_Hubbard_U(self.Udict)

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
            Ublock.append('    %s' % val['scale_factor'])

        self.update_fdf_arguments(
            {'LDAU.Proj': Ublock, 'LDAU.ProjectorGenerationMethod': 2})

    def write_Hubbard_block(self, f):
        pass

    def relax(
        self,
        atoms,
        TypeOfRun='Broyden',
        VariableCell=True,
        ConstantVolume=False,
        RelaxCellOnly=False,
        MaxForceTol=0.001,
        MaxStressTol=1,
        NumCGSteps=40,
    ):
        pbc = atoms.get_pbc()
        initial_magnetic_moments = atoms.get_initial_magnetic_moments()
        self.update_fdf_arguments({
            'MD.TypeOfRun': TypeOfRun,
            'MD.VariableCell': VariableCell,
            'MD.ConstantVolume': ConstantVolume,
            'MD.RelaxCellOnly': RelaxCellOnly,
            'MD.MaxForceTol': "%s eV/Ang" % MaxForceTol,
            'MD.MaxStressTol': "%s GPa" % MaxStressTol,
            'MD.NumCGSteps': NumCGSteps,
        })
        self.calculate(atoms)
        self.read(self.prefix + '.XV')
        self.atoms.set_pbc(pbc)
        self.atoms.set_initial_magnetic_moments(initial_magnetic_moments)
        atoms = self.atoms
        self.update_fdf_arguments({
            'MD.NumCGSteps': 0,
        })
        return self.atoms

    def _write_structure(self, f, atoms):
        """Translate the Atoms object to fdf-format.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        cell = atoms.cell
        f.write('\n')

        if cell.rank in [1, 2]:
            raise ValueError('Expected 3D unit cell or no unit cell.  You may '
                             'wish to add vacuum along some directions.')

        # Write lattice vectors
        if np.any(cell):
            f.write(format_fdf('LatticeConstant', '1.0 Ang'))
            f.write('%block LatticeVectors\n')
            for i in range(3):
                for j in range(3):
                    s = ('    %.15f' % cell[i, j]).rjust(16) + ' '
                    f.write(s)
                f.write('\n')
            f.write('%endblock LatticeVectors\n')
            f.write('\n')

        self._write_atomic_coordinates(f, atoms)

        # Write magnetic moments.
        magmoms = atoms.get_initial_magnetic_moments()

        # The DM.InitSpin block must be written to initialize to
        # no spin. SIESTA default is FM initialization, if the
        # block is not written, but  we must conform to the
        # atoms object.
        if magmoms is not None:
            if len(magmoms) == 0:
                f.write('#Empty block forces ASE initialization.\n')

            f.write('%block DM.InitSpin\n')
            if len(magmoms) != 0 and isinstance(magmoms[0], np.ndarray):
                for n, Mcart in enumerate(magmoms):
                    M = cart2sph(Mcart)
                    if M[0] != 0:
                        f.write('    %d %.14f %.14f %.14f \n' %
                                (n + 1, M[0], M[1], M[2]))
            elif len(magmoms) != 0 and isinstance(magmoms[0], float):
                for n, M in enumerate(magmoms):
                    if M != 0:
                        f.write('    %d %.14f \n' % (n + 1, M))
            f.write('%endblock DM.InitSpin\n')
            f.write('\n')

    def read_results(self):
        """Read the results.
        """
        self.read_number_of_grid_points()
        self.read_energy()
        self.read_forces_stress()
        # self.read_eigenvalues()
        self.read_kpoints()
        self.read_dipole()
        self.read_pseudo_density()
        # self.read_hsx()
        self.read_dim()
        # if self.results['hsx'] is not None:
        #    self.read_pld(self.results['hsx'].norbitals,
        #                  len(self.atoms))
        #    self.atoms.cell = self.results['pld'].cell * Bohr
        # else:
        #    self.results['pld'] = None

        # self.read_wfsx()
        self.read_ion(self.atoms)

        self.read_bands()
