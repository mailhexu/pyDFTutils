import os
import math
import numpy as np
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.siesta.parameters import format_fdf
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.io import write
from pyDFTutils.pseudopotential import DojoFinder
import copy

import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
#from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
    get_valence_charge, read_vca_synth_block
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
#from pyDFTutils.siesta.pdos import gen_pdos_figure, plot_layer_pdos 


def read_siesta_xv(fd):
    vectors = []
    for i in range(3):
        data = next(fd).split()
        vectors.append([float(data[j]) * Bohr for j in range(3)])

    # Read number of atoms (line 4)
    natoms = int(next(fd).split()[0])

    # Read remaining lines
    speciesnumber, atomnumbers, xyz, V = [], [], [], []
    for line in fd:
        if len(line) > 5:  # Ignore blank lines
            data = line.split()
            speciesnumber.append(int(data[0]))
            atomnumbers.append(int(data[1])%200)
            xyz.append([float(data[2 + j]) * Bohr for j in range(3)])
            V.append([float(data[5 + j]) * Bohr for j in range(3)])

    vectors = np.array(vectors)
    atomnumbers = np.array(atomnumbers)
    xyz = np.array(xyz)
    atoms = Atoms(numbers=atomnumbers, positions=xyz, cell=vectors,
                  pbc=True)
    assert natoms == len(atoms)
    return atoms

def read_xv(fname):
    with open(fname) as myfile:
        atoms=read_siesta_xv(myfile)
    return atoms


def get_species(atoms, xc, rel='sr', accuracy='standard'):
    finder = DojoFinder()
    elems = list(dict.fromkeys(atoms.get_chemical_symbols()).keys())
    elem_dict = dict(zip(elems, range(1, len(elems) + 1)))
    pseudo_path = finder.get_pp_path(xc=xc)
    species = [
        Species(symbol=elem,
                pseudopotential=finder.get_pp_fname(
                    elem, xc=xc, rel=rel, accuracy=accuracy),
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
                 basis_set='DZP',
                 species=None,
                 ghosts=[],
                 synthetic_atoms={},
                 input_basis_set={},
                 pseudo_path=None,
                 input_pp={},
                 pp_accuracy='standard',
                 fincore=False,
                 **kwargs):
        # non-perturnbative polarized orbital.
        self.npt_elems = set()
        self.synthetic_atoms=synthetic_atoms

        if atoms is not None:
            finder = DojoFinder()
            elems = list(dict.fromkeys(atoms.get_chemical_symbols()).keys())
            self.elem_dict = dict(zip(elems, range(1, len(elems) + 1)))
            symbols = atoms.get_chemical_symbols()

            # ghosts
            ghost_symbols = [symbols[i] for i in ghosts]
            ghost_elems = list(dict.fromkeys(ghost_symbols).keys())
            tags = [1 if i in ghosts else 0 for i in range(len(atoms))]
            atoms.set_tags(tags)

            if pseudo_path is None:
                pseudo_path = finder.get_pp_path(xc=xc, accuracy=pp_accuracy)

            if spin == 'spin-orbit':
                rel = 'fr'
            else:
                rel = 'sr'
            species = []
            for elem, index in self.elem_dict.items():
                if elem not in input_basis_set:
                    bselem = basis_set
                    if elem in ['Li', 'Be', 'Na', 'Mg']:
                        self.npt_elems.add(f"{elem}.{index}")
                else:
                    bselem = PAOBasisBlock(input_basis_set[elem])
                if elem not in input_pp:
                    pseudopotential = finder.get_pp_fname(
                        elem, xc=xc, rel=rel, accuracy=pp_accuracy, fincore=fincore)
                else:
                    pseudopotential = os.path.join(
                        pseudo_path, input_pp[elem])

                if elem in self.synthetic_atoms:
                    excess_charge = 0
                else:
                    excess_charge = None

                species.append(Species(symbol=elem,
                                       pseudopotential=pseudopotential,
                                       basis_set=bselem,
                                       ghost=False, 
                                       excess_charge=excess_charge))
            for elem in ghost_elems:
                species.append(
                    Species(symbol=elem,
                            pseudopotential=finder.get_pp_fname(
                                elem, xc=xc, rel=rel, accuracy=pp_accuracy, 
                                fincore=fincore),
                            tag=1,
                            ghost=True))


        Siesta.__init__(self,
                        xc=xc,
                        spin=spin,
                        atoms=atoms,
                        pseudo_path=pseudo_path,
                        species=species,
                        **kwargs)
        self.set_npt_elements()
        self.set_synthetic_atoms()

    def _write_species(self, fd, atoms):
        """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, species_numbers = self.species(atoms)

        if self['pseudo_path'] is not None:
            pseudo_path = self['pseudo_path']
        elif 'SIESTA_PP_PATH' in os.environ:
            pseudo_path = os.environ['SIESTA_PP_PATH']
        else:
            mess = "Please set the environment variable 'SIESTA_PP_PATH'"
            raise Exception(mess)

        fd.write(format_fdf('NumberOfSpecies', len(species)))
        fd.write(format_fdf('NumberOfAtoms', len(atoms)))

        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        synth_blocks = []
        for species_number, spec in enumerate(species):
            species_number += 1
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                    pseudopotential = label + '.psf'
                else:
                    label = '.'.join([symbol, self.pseudo_qualifier()])
                    pseudopotential = label + '.psf'
            else:
                pseudopotential = spec['pseudopotential']
                label = os.path.basename(pseudopotential)
                label = '.'.join(label.split('.')[:-1])

            if not os.path.isabs(pseudopotential):
                pseudopotential = join(pseudo_path, pseudopotential)

            if not os.path.exists(pseudopotential):
                mess = "Pseudopotential '%s' not found" % pseudopotential
                raise RuntimeError(mess)

            name = os.path.basename(pseudopotential)
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number

            name = '.'.join(name)
            pseudo_targetpath = self.getpath(name)

            if join(os.getcwd(), name) != pseudopotential:
                if islink(pseudo_targetpath) or isfile(pseudo_targetpath):
                    os.remove(pseudo_targetpath)
                symlink_pseudos = self['symlink_pseudos']

                if symlink_pseudos is None:
                    symlink_pseudos = not os.name == 'nt'

                if symlink_pseudos:
                    os.symlink(pseudopotential, pseudo_targetpath)
                else:
                    shutil.copy(pseudopotential, pseudo_targetpath)
            if not spec['excess_charge'] is None:
                atomic_number += 200
                n_atoms = sum(np.array(species_numbers) == species_number)

                if spec['excess_charge'] != 0:
                    paec = float(spec['excess_charge']) / n_atoms
                    vc = get_valence_charge(pseudopotential)
                    fraction = float(vc + paec) / vc
                    pseudo_head = name[:-4]
                    fractional_command = os.environ['SIESTA_UTIL_FRACTIONAL']
                    cmd = '%s %s %.7f' % (fractional_command,
                                          pseudo_head,
                                          fraction)
                    os.system(cmd)

                    pseudo_head += '-Fraction-%.5f' % fraction
                    synth_pseudo = pseudo_head + '.psf'
                    synth_block_filename = pseudo_head + '.synth'
                    os.remove(name)
                    shutil.copyfile(synth_pseudo, name)
                    synth_block = read_vca_synth_block(
                        synth_block_filename,
                        species_number=species_number)
                    synth_blocks.append(synth_block)
                else:
                    synth_block = self.synthetic_atoms[symbol]



            if len(synth_blocks) > 0:
                fd.write(format_fdf('SyntheticAtoms', list(synth_blocks)))

            label = '.'.join(np.array(name.split('.'))[:-1])
            string = '    %d %d %s' % (species_number, atomic_number, label)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))
        fd.write((format_fdf('ChemicalSpecieslabel', chemical_labels)))
        fd.write('\n')
        fd.write((format_fdf('PAO.Basis', pao_basis)))
        fd.write((format_fdf('PAO.BasisSizes', basis_sizes)))
        fd.write('\n')


    def set_npt_elements(self):
        if len(self.npt_elems) > 0:
            npt_text = []
            for name in self.npt_elems:
                npt_text.append(
                    f"{name} non-perturbative ")
            # npt_text += "%endblock PAO.PolarizationScheme\n"
            self['fdf_arguments'].update({"PAO.PolarizationScheme": npt_text})

    def set_synthetic_atoms(self):
        nsyn=len(self.synthetic_atoms)
        if  nsyn> 0:
            syntext = []
            #syntext.append(f"{nsyn}")
            for name, content in self.synthetic_atoms.items():
                syntext.append(
                    f"{self.elem_dict[name]}")
                syntext.append(
                    " ".join([str(x) for x in content[0]]))
                syntext.append(
                    " ".join([str(x) for x in content[1]]))
            self['fdf_arguments'].update({"SyntheticAtoms": syntext})

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
        relaxed_file="relaxed.vasp"
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
        #self.read(self.prefix + '.XV')
        self.atoms=read_xv(os.path.join(self.directory, self.prefix + '.XV'))
        self.atoms.set_pbc(pbc)
        self.atoms.set_initial_magnetic_moments(initial_magnetic_moments)
        atoms = self.atoms
        self.update_fdf_arguments({
            'MD.NumCGSteps': 0,
        })
        if relaxed_file is not None:
            write(relaxed_file, atoms, vasp5=True, sort=False)
        return self.atoms

    def scf_calculation(self, atoms, dos=True, kpts=[7,7,7], **kwargs):
        if dos:
            k1, k2, k3 = kpts
            self.update_fdf_arguments({'WriteEigenvalues': '.true.', 
        		'ProjectedDensityOfStates': ['-70.00 30.0 0.015 3000 eV'],
                'PDOS.kgrid_Monkhorst_Pack': [f'{k1} 0 0 0.0',
                                              f'0 {k2} 0 0.0',
                                              f'0 0 {k3} 0.0']})
        self.calculate(atoms, **kwargs)

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
