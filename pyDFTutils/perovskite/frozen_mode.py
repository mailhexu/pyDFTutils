import numpy as np
import sys
#from phonopy.structure.cells import get_supercell
from ase import Atoms
from ase.io import write

import numpy as np
import copy
import pyDFTutils.perovskite.perovskite_mode as perovskite_mode
from pyDFTutils.perovskite.perovskite_mode import Gamma_modes

import spglib.spglib
#from phonopy.structure.atoms import PhonopyAtoms as Atoms
from pyDFTutils.perovskite.cubic_perovskite import gen_primitive
from pyDFTutils.ase_utils import vesta_view


class distorted_cell():
    def __init__(self, atoms, supercell_matrix=np.eye(3)):
        self._primitive_cell = atoms
        self._supercell_matrix = supercell_matrix
        self._supercell = get_supercell(atoms, supercell_matrix, symprec=1e-4)
        self._N = np.linalg.det(supercell_matrix)

    def _get_cell_with_modulation(self, modulation):
        """
        x=x+dx
        """
        lattice = self._supercell.get_cell()
        positions = self._supercell.get_positions()
        masses = self._supercell.get_masses()
        #magmoms = self._supercell.get_magnetic_moments()
        symbols = self._supercell.get_chemical_symbols()
        positions += modulation.real
        scaled_positions = np.dot(positions, np.linalg.inv(lattice))
        for p in scaled_positions:
            p -= np.floor(p)
            #cell = self._supercell.copy()
            cell = copy.copy(self._supercell)
            cell.set_scaled_positions(scaled_positions)
        return cell

    def _get_displacements(self,
                           eigvec,
                           q,
                           amplitude,
                           argument,
                           use_isotropy_amplitue=True):
        """
        displacements from eigvec, q, amplitude
        """
        m = self._supercell.get_masses()
        s2u_map = self._supercell.get_supercell_to_unitcell_map()
        u2u_map = self._supercell.get_unitcell_to_unitcell_map()
        s2uu_map = [u2u_map[x] for x in s2u_map]
        spos = self._supercell.get_scaled_positions()
        dim = self._supercell.get_supercell_matrix()
        coefs = np.exp(2j * np.pi * np.dot(
            np.dot(spos, dim.T), q))  # Here Do not use sqrt(m)/ np.sqrt(m)
        u = []
        for i, coef in enumerate(coefs):
            eig_index = s2uu_map[i] * 3
            u.append(eigvec[eig_index:eig_index + 3] * coef)

        #u = np.array(u) / np.sqrt(len(m))
        u = np.array(u) / np.linalg.norm(u)  # /np.sqrt(self._N)
        phase_factor = self._get_phase_factor(u, argument)

        if use_isotropy_amplitue:
            amplitude = amplitude  # *self._N
        u *= phase_factor * amplitude

        return u

    def _get_phase_factor(self, modulation, argument):
        u = np.ravel(modulation)
        index_max_elem = np.argmax(abs(u))
        max_elem = u[index_max_elem]
        phase_for_zero = max_elem / abs(max_elem)
        phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero

        return phase_factor


def get_supercell(unitcell, supercell_matrix, symprec=1e-5):
    return Supercell(unitcell, supercell_matrix, symprec=symprec)


def get_primitive(supercell, primitive_frame, symprec=1e-5):
    return Primitive(supercell, primitive_frame, symprec=symprec)


def trim_cell(relative_axes, cell, symprec):
    """
    relative_axes: relative axes to supercell axes
    Trim positions outside relative axes

    """
    positions = cell.get_scaled_positions()
    numbers = cell.get_atomic_numbers()
    masses = cell.get_masses()
    #magmoms = cell.get_magnetic_moments()
    lattice = cell.get_cell()
    trimed_lattice = np.dot(relative_axes.T, lattice)

    trimed_positions = []
    trimed_numbers = []
    if masses is None:
        trimed_masses = None
    else:
        trimed_masses = []
    # if magmoms is None:
    #    trimed_magmoms = None
    # else:
    #    trimed_magmoms = []
    extracted_atoms = []

    positions_in_new_lattice = np.dot(positions,
                                      np.linalg.inv(relative_axes).T)
    positions_in_new_lattice -= np.floor(positions_in_new_lattice)
    trimed_positions = np.zeros_like(positions_in_new_lattice)
    num_atom = 0

    mapping_table = np.arange(len(positions), dtype='intc')
    symprec2 = symprec**2
    for i, pos in enumerate(positions_in_new_lattice):
        is_overlap = False
        if num_atom > 0:
            diff = trimed_positions[:num_atom] - pos
            diff -= np.rint(diff)
            # Older numpy doesn't support axis argument.
            # distances = np.linalg.norm(np.dot(diff, trimed_lattice), axis=1)
            # overlap_indices = np.where(distances < symprec)[0]
            distances2 = np.sum(np.dot(diff, trimed_lattice)**2, axis=1)
            overlap_indices = np.where(distances2 < symprec2)[0]
            if len(overlap_indices) > 0:
                is_overlap = True
                mapping_table[i] = extracted_atoms[overlap_indices[0]]

        if not is_overlap:
            trimed_positions[num_atom] = pos
            num_atom += 1
            trimed_numbers.append(numbers[i])
            if masses is not None:
                trimed_masses.append(masses[i])
            # if magmoms is not None:
            #    trimed_magmoms.append(magmoms[i])
            extracted_atoms.append(i)

    trimed_cell = Atoms(
        numbers=trimed_numbers,
        masses=trimed_masses,
        # magmoms=trimed_magmoms,
        scaled_positions=trimed_positions[:num_atom],
        cell=trimed_lattice, )

    return trimed_cell, extracted_atoms, mapping_table


def print_cell(cell, mapping=None, stars=None):
    symbols = cell.get_chemical_symbols()
    masses = cell.get_masses()
    magmoms = cell.get_magnetic_moments()
    lattice = cell.get_cell()
    print("Lattice vectors:")
    print("  a %20.15f %20.15f %20.15f" % tuple(lattice[0]))
    print("  b %20.15f %20.15f %20.15f" % tuple(lattice[1]))
    print("  c %20.15f %20.15f %20.15f" % tuple(lattice[2]))
    print("Atomic positions (fractional):")
    for i, v in enumerate(cell.get_scaled_positions()):
        num = " "
        if stars is not None:
            if i in stars:
                num = "*"
        num += "%d" % (i + 1)
        line = ("%5s %-2s%18.14f%18.14f%18.14f" %
                (num, symbols[i], v[0], v[1], v[2]))
        if masses is not None:
            line += " %7.3f" % masses[i]
        if magmoms is not None:
            line += "  %5.3f" % magmoms[i]
        if mapping is None:
            print(line)
        else:
            print(line + " > %d" % (mapping[i] + 1))


class Supercell(Atoms):
    """Build supercell from supercell matrix
    In this function, unit cell is considered
    [1,0,0]
    [0,1,0]
    [0,0,1].
    Supercell matrix is given by relative ratio, e.g,
    [-1, 1, 1]
    [ 1,-1, 1]  is for FCC from simple cubic.
    [ 1, 1,-1]
    In this case multiplicities of surrounding simple lattice are [2,2,2].

    First, create supercell with surrounding simple lattice.
    Second, trim the surrounding supercell with the target lattice.
    """

    def __init__(self, unitcell, supercell_matrix, symprec=1e-5):
        self._s2u_map = None
        self._u2s_map = None
        self._u2u_map = None
        self._supercell_matrix = np.array(supercell_matrix, dtype='intc')
        self._create_supercell(unitcell, symprec)

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def get_supercell_to_unitcell_map(self):
        return self._s2u_map

    def get_unitcell_to_supercell_map(self):
        return self._u2s_map

    def get_unitcell_to_unitcell_map(self):
        return self._u2u_map

    def _create_supercell(self, unitcell, symprec):
        mat = self._supercell_matrix
        frame = self._get_surrounding_frame(mat)
        sur_cell, u2sur_map = self._get_simple_supercell(frame, unitcell)

        # Trim the simple supercell by the supercell matrix
        trim_frame = np.array([
            mat[0] / float(frame[0]), mat[1] / float(frame[1]),
            mat[2] / float(frame[2])
        ])
        supercell, sur2s_map, mapping_table = trim_cell(trim_frame, sur_cell,
                                                        symprec)

        multi = supercell.get_number_of_atoms(
        ) // unitcell.get_number_of_atoms()

        if multi != determinant(self._supercell_matrix):
            print("Supercell creation failed.")
            print("Probably some atoms are overwrapped. "
                  "The mapping table is give below.")
            print(mapping_table)
            Atoms.__init__(self)
        else:
            Atoms.__init__(
                self,
                numbers=supercell.get_atomic_numbers(),
                masses=supercell.get_masses(),
                # magmoms=supercell.get_magnetic_moments(),
                scaled_positions=supercell.get_scaled_positions(),
                cell=supercell.get_cell(),
                # pbc=True)
            )
            self._u2s_map = np.arange(unitcell.get_number_of_atoms()) * multi
            self._u2u_map = dict([(j, i) for i, j in enumerate(self._u2s_map)])
            self._s2u_map = np.array(u2sur_map)[sur2s_map] * multi

    def _get_surrounding_frame(self, supercell_matrix):
        # Build a frame surrounding supercell lattice
        # For example,
        #  [2,0,0]
        #  [0,2,0] is the frame for FCC from simple cubic.
        #  [0,0,2]
        m = np.array(supercell_matrix)
        axes = np.array([[0, 0, 0], m[:, 0], m[:, 1], m[:, 2],
                         m[:, 1] + m[:, 2], m[:, 2] + m[:, 0],
                         m[:, 0] + m[:, 1], m[:, 0] + m[:, 1] + m[:, 2]])
        frame = [max(axes[:, i]) - min(axes[:, i]) for i in (0, 1, 2)]
        return frame

    def _get_simple_supercell(self, multi, unitcell):
        # Scaled positions within the frame, i.e., create a supercell that
        # is made simply to multiply the input cell.
        positions = unitcell.get_scaled_positions()
        numbers = unitcell.get_atomic_numbers()
        masses = unitcell.get_masses()
        #magmoms = unitcell.get_magnetic_moments()
        lattice = unitcell.get_cell()

        atom_map = []
        positions_multi = []
        numbers_multi = []
        if masses is None:
            masses_multi = None
        else:
            masses_multi = []
        # if magmoms is None:
        #    magmoms_multi = None
        # else:
        #    magmoms_multi = []
        for l, pos in enumerate(positions):
            for i in range(multi[2]):
                for j in range(multi[1]):
                    for k in range(multi[0]):
                        positions_multi.append([(pos[0] + k) / multi[0],
                                                (pos[1] + j) / multi[1],
                                                (pos[2] + i) / multi[2]])
                        numbers_multi.append(numbers[l])
                        if masses is not None:
                            masses_multi.append(masses[l])
                        atom_map.append(l)
                        # if magmoms is not None:
                        #    magmoms_multi.append(magmoms[l])

        simple_supercell = Atoms(
            numbers=numbers_multi,
            masses=masses_multi,
            # magmoms=magmoms_multi,
            scaled_positions=positions_multi,
            cell=np.dot(np.diag(multi), lattice),
            # pbc=True
        )

        return simple_supercell, atom_map


class Primitive(Atoms):
    def __init__(self, supercell, primitive_matrix, symprec=1e-5):
        """
        primitive_matrix (3x3 matrix):
        Primitive lattice is given with respect to supercell by
           np.dot(primitive_matrix.T, supercell.get_cell())
        """
        self._primitive_matrix = np.array(primitive_matrix)
        self._symprec = symprec
        self._p2s_map = None
        self._s2p_map = None
        self._p2p_map = None
        self._primitive_cell(supercell)
        self._supercell_to_primitive_map(supercell.get_scaled_positions())
        self._primitive_to_primitive_map()

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def get_primitive_to_supercell_map(self):
        return self._p2s_map

    def get_supercell_to_primitive_map(self):
        return self._s2p_map

    def get_primitive_to_primitive_map(self):
        return self._p2p_map

    def _primitive_cell(self, supercell):
        trimed_cell, p2s_map, mapping_table = trim_cell(
            self._primitive_matrix, supercell, self._symprec)
        Atoms.__init__(
            self,
            numbers=trimed_cell.get_atomic_numbers(),
            masses=trimed_cell.get_masses(),
            # magmoms=trimed_cell.get_magnetic_moments(),
            scaled_positions=trimed_cell.get_scaled_positions(),
            cell=trimed_cell.get_cell(),
            pbc=True)

        self._p2s_map = np.array(p2s_map, dtype='intc')

    def _supercell_to_primitive_map(self, pos):
        inv_F = np.linalg.inv(self._primitive_matrix)
        s2p_map = []
        for i in range(pos.shape[0]):
            s_pos = np.dot(pos[i], inv_F.T)
            for j in self._p2s_map:
                p_pos = np.dot(pos[j], inv_F.T)
                diff = p_pos - s_pos
                diff -= np.rint(diff)
                if (abs(diff) < self._symprec).all():
                    s2p_map.append(j)
                    break
        self._s2p_map = np.array(s2p_map, dtype='intc')

    def _primitive_to_primitive_map(self):
        """
        Mapping table from supercell index to primitive index
        in primitive cell
        """
        self._p2p_map = dict([(j, i) for i, j in enumerate(self._p2s_map)])


def determinant(m):
    return (m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] + m[0][1]
            * m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[0][2] *
            m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0])


def gen_P21c_perovskite(
        name,
        cell=[3.9, 3.9, 3.9],
        supercell_matrix=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
        modes=dict(
        R2_m_O1=0.0,  # R2-[O1:c:dsp]A2u(a), O, breathing
        # R3-[O1:c:dsp]A2u(a), O JT inplane-stagger, out-of-plane antiphase
        R3_m_O1=0.0,
        # R3-[O1:c:dsp]A2u(b), O, out-of-plane-stagger, inplane antiphase, Unusual.
        R3_m_O2=0.0,
        R4_m_A1=0.0,  # R4-[Nd1:a:dsp]T1u(a), A , Unusual
        R4_m_A2=0.0,  # R4-[Nd1:a:dsp]T1u(b), A, Unusual
        R4_m_A3=0.0,  # R4-[Nd1:a:dsp]T1u(c), A, Unusual
        R4_m_O1=0.0,  # R4-[O1:c:dsp]Eu(a), O, Unusual
        R4_m_O2=0.0,  # R4-[O1:c:dsp]Eu(b), O, Unusual
        R4_m_O3=0.0,  # R4-[O1:c:dsp]Eu(c), O, Unusual
        R5_m_O1=0.0,  # R5-[O1:c:dsp]Eu(a), O  a-
        R5_m_O2=0.0,  # R5-[O1:c:dsp]Eu(b), O  b-
        R5_m_O3=0.0,  # R5-[O1:c:dsp]Eu(c), O  c-
        X3_m_A1=0.0,  # X3-[Nd1:a:dsp]T1u(a), What's this..
        X3_m_O1=0.0,  # X3-[O1:c:dsp]A2u(a)

        # X5_m_A1=0.0,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode
        # X5_m_A2=0.0,  # [Nd1:a:dsp]T1u(b), A , save as above
        # X5_m_O1=0.0,  # [Nd1:a:dsp]T1u(a), O , Antiferro mode
        # X5_m_O2=0.0,  # [Nd1:a:dsp]T1u(b), O , same as above
        # M2_p_O1=0.0,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation c+

        Z5_m_A1=0.0,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode
        Z5_m_A2=0.0,  # [Nd1:a:dsp]T1u(b), A , save as above
        Z5_m_O1=0.0,  # [Nd1:a:dsp]T1u(a), O , Antiferro mode
        Z5_m_O2=0.0,  # [Nd1:a:dsp]T1u(b), O , same as above
        M2_p_O1=0.0,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation

        M3_p_O1=0.0,  # M3+[O1:c:dsp]A2u(a), O, D-type JT inplane stagger
        M5_p_O1=0.0,  # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting
        M5_p_O2=0.0,  # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting
        # M4+[O1:c:dsp]A2u(a), O, in-plane-breathing (not in P21/c)
        M4_p_O1=0.0,

        G_Ax=0.0,
        G_Ay=0.0,
        G_Az=0.0,
        G_Sx=0.0,
        G_Sy=0.0,
        G_Sz=0.0,
        G_Axex=0.0,
        G_Axey=0.0,
        G_Axez=0.0,
        G_Lx=0.0,
        G_Ly=0.0,
        G_Lz=0.0,
        G_G4x=0.0,
        G_G4y=0.0,
        G_G4z=0.0,
        )
):
    atoms = gen_primitive(name=name, mag_order='PM', latticeconstant=cell[0])
    spos = atoms.get_scaled_positions()
    atoms.set_cell(cell)
    atoms.set_scaled_positions(spos)
    dcell = distorted_cell(atoms, supercell_matrix=supercell_matrix)
    eigvec = np.zeros(15)

    mode_dict = {
        'R2_m_O1': perovskite_mode.R2p,
        # R3-[O1:c:dsp]A2u(a), O JT inplane-stagger, out-of-plane antiphase
        'R3_m_O1': perovskite_mode.R12p_1,
        # R3-[O1:c:dsp]A2u(b), O, out-of-plane-stagger, inplane antiphase
        'R3_m_O2': perovskite_mode.R12p_2,
        'R4_m_A1': perovskite_mode.R15_1,  # R4-[Nd1:a:dsp]T1u(a), A
        'R4_m_A2': perovskite_mode.R15_2,  # R4-[Nd1:a:dsp]T1u(b), A
        'R4_m_A3': perovskite_mode.R15_3,  # R4-[Nd1:a:dsp]T1u(c), A
        'R4_m_O1': perovskite_mode.R15_4,  # R4-[O1:c:dsp]Eu(a), O
        'R4_m_O2': perovskite_mode.R15_5,  # R4-[O1:c:dsp]Eu(b), O
        'R4_m_O3': perovskite_mode.R15_6,  # R4-[O1:c:dsp]Eu(c), O
        # R5-[O1:c:dsp]Eu(a), O, out-of-phase rotation a-
        'R5_m_O1': perovskite_mode.R25_1,
        'R5_m_O2': perovskite_mode.R25_2,  # R5-[O1:c:dsp]Eu(b), O, b-
        # R5-[O1:c:dsp]Eu(c), O, c-. For Pnma. Do not use.
        'R5_m_O3': perovskite_mode.R25_3,

        # 'X3_m_A1':perovskite_mode., # X3-[Nd1:a:dsp]T1u(a), What's this..
        # 'X3_m_O1':perovskite_mode., # X3-[O1:c:dsp]A2u(a)
        'Z5_m_A1':
        perovskite_mode.Z5p_1,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode
        'Z5_m_A2':
        perovskite_mode.Z5p_2,  # [Nd1:a:dsp]T1u(b), A , save as above
        'Z5_m_O1':
        perovskite_mode.Z5p_3,  # [Nd1:a:dsp]T1u(b), O , same as above
        'Z5_m_O2':
        perovskite_mode.Z5p_4,  # [Nd1:a:dsp]T1u(b), O , same as above

        'M2_p_O1':
        perovskite_mode.M3,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation
        # M3+[O1:c:dsp]A2u(a), O, D-type JT inplane stagger
        'M3_p_O1': perovskite_mode.M2,
        # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting
        'M5_p_O1': perovskite_mode.M5_1,
        # M5+[O1:c:dsp]Eu(b), O, Out of phase tilting, -above
        'M5_p_O2': perovskite_mode.M5_2,
        # M4+[O1:c:dsp]A2u(a), O, in-plane-breathing (not in P21/c)
        'M4_p_O1': perovskite_mode.M4,
    }

    # add Gamma modes to mode_dict
    Gamma_mode_dict = Gamma_modes(atoms.get_chemical_symbols())
    mode_dict.update(Gamma_mode_dict)

    mode_disps = {}
    qdict = {'G': [0, 0, 0],
             # 'X':[0,0.0,0.5],
             'M': [0.5, 0.5, 0],
             'R': [0.5, 0.5, 0.5],
             'Z': [0.0, 0.0, 0.5]
             }
    disps = 0.0  # np.zeros(3,dtype='complex128')
    for name, amp in modes.items():
        eigvec = np.array(mode_dict[name])
        disp = dcell._get_displacements(
            eigvec=eigvec,
            q=qdict[name[0]],
            amplitude=amp,
            argument=0)
        disps += disp

    newcell = dcell._get_cell_with_modulation(disps)
    newcell = Atoms(newcell)
    print(spglib.get_spacegroup(newcell))
    # vesta_view(newcell)
    return newcell


def gen_distorted_perovskite(
        name,
        cell=[3.9, 3.9, 3.9],
        supercell_matrix=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
        out_of_phase_rotation=0.0,
        in_phase_rotation=0.0,
        in_phase_tilting=0.0,
        breathing=0.0,
        JT_d=0.0,
        JT_a=0.0):
    atoms = gen_primitive(name=name, mag_order='PM', latticeconstant=3.9)
    spos = atoms.get_scaled_positions()
    # atoms.set_cell([3.5,3.5,3.9,90,90,90])
    atoms.set_cell(cell)
    atoms.set_scaled_positions(spos)
    #from ase.io import write
    # write('cubic_LaMnO3.cif',atoms)
    dcell = distorted_cell(atoms, supercell_matrix=supercell_matrix)
    eigvec = np.zeros(15)

    eig_breathing = np.array(perovskite_mode.R2p)

    eig_JT_d = np.array(perovskite_mode.M2)

    eig_in_phase_tilting = np.array(perovskite_mode.X5_3)

    eig_out_of_phase_rotation_x = np.array(perovskite_mode.R25_1)
    eig_out_of_phase_rotation_y = np.array(perovskite_mode.R25_2)

    eig_in_phase_rotation_z = np.array(perovskite_mode.M3)

    disp_br = dcell._get_displacements(
        eigvec=eig_breathing,
        q=[0.5, 0.5, 0.5],
        amplitude=breathing,
        argument=0)
    # disp2=dcell._get_displacements(eigvec=out_of_phase_rotation,q=[0.5,0.5,0.5],amplitude=0.55,argument=0)
    disp_jt = dcell._get_displacements(
        eigvec=eig_JT_d, q=[0.5, 0.5, 0.0], amplitude=JT_d, argument=0)

    disp_tilting = dcell._get_displacements(
        eigvec=eig_in_phase_tilting,
        q=[0.0, 0.0, 0.5],
        amplitude=in_phase_tilting,
        argument=0)

    disp_rotx = dcell._get_displacements(
        eigvec=eig_out_of_phase_rotation_x,
        q=[0.5, 0.5, 0.5],
        amplitude=out_of_phase_rotation,
        argument=0)
    disp_roty = dcell._get_displacements(
        eigvec=eig_out_of_phase_rotation_y,
        q=[0.5, 0.5, 0.5],
        amplitude=out_of_phase_rotation,
        argument=0)
    disp_rotz = dcell._get_displacements(
        eigvec=eig_in_phase_rotation_z,
        q=[0.5, 0.5, 0.0],
        amplitude=in_phase_rotation,
        argument=0)

    # print(out_of_phase_rotation)
    # print(disp_rotx)
    # print(disp_jt)

    # print disp.shape
    newcell = dcell._get_cell_with_modulation(
        disp_jt + disp_rotx + disp_roty + disp_rotz + disp_br + disp_tilting)
    newcell = Atoms(newcell)
    print(spglib.get_spacegroup(newcell))
    # vesta_view(newcell)
    return newcell


def isotropy_normfactor(scell, sc_mat, disps):
    """
    pcell: primitive cell parameter. 3*3 matrix
    sc_mat: primitive->supercell transform matrix. 3*3 matrix
    disps: list of vectors defining displacements.
    """
    # Bs supercell
    # Bs=np.dot(pcell, sc_mat)
    # scell: supercell
    sum = 0.0
    for disp in disps:
        sum += (np.linalg.norm(np.dot(scell, disp)))**2
    norm_factor = 1.0 / np.sqrt(sum)
    return norm_factor


def test():
    atoms = gen_P21c_perovskite(name='YNiO3', cell=[3.7, 3.7, 3.7],

                                supercell_matrix=[
                                    [1, -1, 0], [1, 1, 0], [0, 0, 2]],
                                modes=dict(
        # R2_m_O1=0.8, #breathing
        # R3_m_O1=1.0,
        # R3_m_O2=1.0,  # R3-[O1:c:dsp]A2u(b), O, out-of-plane-stagger, inplane antiphase

        # R5_m_O1=1.2,  # R5-[O1:c:dsp]Eu(a), O a-
        # R5_m_O2=1.2,  # R5-[O1:c:dsp]Eu(a), O b-
        # R5_m_O3=1.0,  # R5-[O1:c:dsp]Eu(c), O  c-
        # X5_m_A1=1.0,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode


        # R4_m_A1=0.5,  # R4-[Nd1:a:dsp]T1u(a), A , Antipolar mode in Pnma
        # R4_m_A2=0.5,  # R4-[Nd1:a:dsp]T1u(b), A,  Antipolar mode in Pnma
        # R4_m_A3=0.0,  # R4-[Nd1:a:dsp]T1u(c), A, Unusual
        # R4_m_A3=0.0,  # R4-[Nd1:a:dsp]T1u(c), A, Unusual
        # R4_m_O1=0.0,  # R4-[O1:c:dsp]Eu(a), O, Unusual
        # R4_m_O2=0.0,  # R4-[O1:c:dsp]Eu(b), O, Unusual
        # R4_m_O3=0.0,  # R4-[O1:c:dsp]Eu(c), O, Unusual


        # M2_p_O1=1.2,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation c+

        # M3_p_O1=0.1,  # M3+[O1:c:dsp]A2u(a), O, D-type JT inplane stagger

        # M5_p_O1=1.0,  # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting

        # M4_p_O1=1.0 , # M4+[O1:c:dsp]A2u(a), O, in-plane-breathing (not in P21/c)
        G_Ax=0.0,
        G_Ay=0.0,
        G_Az=0.0,
        G_Sx=0.0,
        G_Sy=0.0,
        G_Sz=0.0,
        G_Axex=0.0,
        G_Axey=0.0,
        G_Axez=0.0,
        G_Lx=0.0,
        G_Ly=0.0,
        G_Lz=0.0,
        # G_G4x=0.1,
        # G_G4y=0.1,
        # G_G4z=0.1,
    )
    )
    vesta_view(atoms)


if __name__ == '__main__':
    test()
