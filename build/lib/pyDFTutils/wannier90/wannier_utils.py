#!/usr/bin/env python
from pyDFTutils.ase_utils import symbol_number, symnum_to_sym
import numpy as np
import re
import os
import sys
from xyz_read import projections_to_basis, projection_dict_by_site_to_basis


def read_nbands(filename='OUTCAR'):
    """
    read the number of bands form OUTCAR.
    """
    text = open(filename, 'r').read()
    m = re.search('NBANDS\s*=\s*(\d*)', text)
    if m:
        t = m.group(1)
    else:
        return None
    return int(t)


def read_value(key, filename='wannier90.up.win'):
    """
    read the value from file with line "key = val"
    """
    lines = open(filename, 'r').readlines()
    for i, line in enumerate(lines):
        m = re.search('^\s*%s\s*[=|:]\s*(.*)' % key, line)
        if m is not None:
            return m.group(1)
    return None

def find_value_line(lines, key):
    for i, line in enumerate(lines):
        m = re.search('^\s*%s\s*[=|:]\s*.*' % key, line)
        if m is not None:
            return i
    return None


def sub_lines(text, key, value):
    i = find_value_line(text, key)
    if i is not None:
        text[i] = "%s = %s\n" % (key, value)
    else:
        text.append("%s = %s\n" % (key, value))
    return text


def sub_text(filename, **kwargs):
    """
    write value to the wannier input file.

    :param filename: the name of the input file.
    usage: e.g. sub_text('wannier90.win',bands_plot='true')
    """
    with open(filename) as infile:
        lines = infile.readlines()
    for key, value in kwargs.items():
        lines = sub_lines(lines, key, value)
    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line)


def run_wannier(command=None, spin=None, copy_win=True):
    """
    run wannier90.
    """
    if command is None:
        command = 'wannier90.x'
    if spin is None:
        name = 'wannier90'
    elif spin == 'up':
        name = 'wannier90.up'
    elif spin == 'dn' or spin == 'down':
        name = 'wannier90.dn'
    if spin is not None and copy_win:
        os.system('cp wannier90.win %s.win' % name)
    os.system("%s %s" % (command, name))
    if spin is not None:
        if not os.path.exists(name):
            os.mkdir(name)


int_keys = ("num_bands", "num_wann", 'bands_num_points',
            "fermi_surface_num_points", "num_iter", "num_cg_steps",
            "conv_window", "conv_noise_num", "num_dump_cycles",
            "search_shells", "wannier_plot_supercell")

float_keys = (
    "dis_win_min",
    "dis_win_max",
    "dis_froz_min",
    "dis_froz_max",
    "wannier_plot_radius",
    "hr_cutoff",
    "dist_cutoff",
    "fermi_energy",
    "conv_tol",
    "conv_noise_amp",
    "kmesh_tol", )

bool_keys = (
    "guiding_centres",
    "use_bloch_phases",
    "write_xyz",
    "write_hr_diag",
    "wannier_plot",
    "bands_plot",
    "fermi_surface_plot",
    "hr_plot", )

string_keys = (
    "wannier_plot_format",
    "wannier_plot_mode",
    "bands_plot_format", )

list_keys = (
    "wannier_plot_list",
    "bands_plot_project", )

all_keys = float_keys + bool_keys + string_keys + list_keys

s_orbs = ['s']
eg_orbs = ['dxz', 'dyz', 'dxy']
t2g_orbs = ['dz2', 'dx2-y2']
d_orbs = ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
p_orbs = ['pz', 'px', 'py']

# angular momentum dict, which is NOT that used in wannier90
orb_dict = {
    's': (0, 0),
    'pz': (0, 1),
    'px': (1, 1),
    'py': (-1, 1),
    'dz2': (0, 2),
    'dxz': (1, 2),
    'dyz': (-1, 2),
    'dxy': (2, 2),
    'dx2-y2': (-2, 2),
    'fz3': (0, 3),
    'fxz2': (1, 3),
    'fyz2': (-1, 3),
    'fxyz': (2, 3),
    'fz(x2-y2)': (-2, 3),
    'fx(x2-3y2)': (3, 3),
    'fx(3x2-y2)': (-3, 3)
}

w90_orb_dict = {
    's': (0, 0),
    'pz': (1, 1),
    'px': (2, 1),
    'py': (3, 1),
    'dz2': (1, 2),
    'dxz': (2, 2),
    'dyz': (3, 2),
    'dxy': (5, 2),
    'dx2-y2': (4, 2),
    'fz3': (1, 3),
    'fxz2': (2, 3),
    'fyz2': (3, 3),
    'fxyz': (5, 3),
    'fz(x2-y2)': (4, 3),
    'fx(x2-3y2)': (6, 3),
    'fx(3x2-y2)': (7, 3),
    'p': (None, 1),
    'd': (None, 2),
    'f': (None, 3),
}

revesed_w90_orb_dict = dict(zip(w90_orb_dict.values(), w90_orb_dict.keys()))


class wannier_interface():
    def __init__(self, atoms, seed=None, bands=None, spin=0, **kwargs):
        """
        The wannier.win generator.
        """
        self.atoms=atoms
        if seed is None:
            seed = 'wannier90'
        self.seed = seed
        self.bands = bands
        self.spin = spin

        self.projection_dict = None
        self.unit_cell = None
        self.kpoints = None
        self.kpath = None
        self.mp_grid = None
        self.float_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_params = {}
        for key in float_keys:
            self.float_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_keys:
            self.list_params[key] = None

        self.basis = []

        self.set(**kwargs)
        self.projection_dict_by_site = {}
        self.initial_basis = []

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.unit_cell = atoms.get_cell()

    def add_basis(self, atom_name, orb=None, m=None, l=None, r=None,
                  spin=None):
        """
        atoms_name: name of specy or a atoms. Eg. 'Fe' or 'Fe1'.
        """
        if orb is not None:
            if m is not None or l is not None:
                raise ValueError(
                    "the projection can be either given by name (like 'dxy') or by (m,l) pair (4,2)"
                )
            m, l = w90_orb_dict[orb]

        if m is None:
            mlist = [i + 1 for i in range(l * 2 + 1)]
        else:
            mlist = [m]

        for mi in mlist:
            if '0' <= atom_name[-1] <= '9':  # single atoms
                sdict = symbol_number(self.atoms)
                self.initial_basis.append([atom_name, mi, l, r, spin])
            else:
                sdict = symbol_number(self.atoms)
                for aname in sdict:
                    if symnum_to_sym(aname) == atom_name:
                        self.initial_basis.append([aname, mi, l, r, spin])

    def set_projections(self, projection_dict=None):
        """
        set the projection

        :param projection_dict: eg. {'Ti':['dxy','dyz','dxz']}
        or {'Ba':{'r=1':['l=0','l=1'], 'r=2':['l=0']}}
        """
        self.projection_dict = projection_dict

    def add_projections_by_site(self, projections, atoms=None):
        """
        set projections by site. eg. {('Fe1','Fe3'):['dxy','dyz','dxz']}
        """
        for key, val in projections.items():
            self.projection_dict_by_site[key] = val
        if atoms is not None:
            self.atoms = atoms

    def get_basis(self, atoms=None):
        """
        get the basis.
        """
        if self.atoms is None:
            if atoms is None:
                raise ValueError("atoms should be specified.")
            else:
                self.atoms = atoms
        self.basis = []
        ### To Be removed. Only kept for backward compatibility.
        if self.projection_dict is not None:
            self.basis += projections_to_basis(self.atoms,
                                               self.projection_dict)
        if self.projection_dict_by_site is not None:
            self.basis += projection_dict_by_site_to_basis(
                self.atoms, self.projection_dict_by_site)
        ### End of to be moved.
        for b in self.initial_basis:
            symnum, m, l, r, spin = b
            bname = '|'.join(
                ['symnum', w90_orb_dict[(m, l)], str(r), str(spin)])
            self.basis.append(bname)
        return self.basis

    def write_basis(self, fname='basis.txt'):
        """
        write basis to a file. default fname is basis.txt.
        """
        self.get_basis()
        with open(fname, 'w') as myfile:
            for b in self.basis:
                myfile.write(str(b) + '\n')

    def set(self, **kwargs):
        """
        set the parameters.
        """
        for key in kwargs:
            if key in self.float_params:
                self.float_params[key] = kwargs[key]
            elif key in self.string_params:
                self.string_params[key] = kwargs[key]
            elif key in self.int_params:
                self.int_params[key] = kwargs[key]
            elif key in self.bool_params:
                self.bool_params[key] = kwargs[key]
            elif key in self.list_params:
                self.list_params[key] = kwargs[key]
            else:
                raise ValueError("%s is not a valid key" % key)

    def set_energy_window(self, win, froz_win, shift_efermi=0):
        """
		set the energy window for entanglement.

        :param win: the disentangle window. [min,max]
        :param froz_win: the frozen window. [min,max]
		:param shift_efermi: shift the energies. eg. if shift_efermi =3, the energies will be added by 3.
        """
        assert froz_win[0] >= win[0]
        assert froz_win[1] <= win[1]
        self.set(
            dis_win_min=win[0] + shift_efermi,
            dis_win_max=win[1] + shift_efermi,
            dis_froz_min=froz_win[0] + shift_efermi,
            dis_froz_max=froz_win[1] + shift_efermi, )

    def set_kpath(self, kpoints, labels, npoints):
        """
        set the kpoints path to draw band diagram.

        :param kpoints: the spectial kpoint list.
        :param  labels: the name of the kpoints.
        :param npoints: the number of kpoints in between.
        """
        self.kpath = zip(labels, np.array(kpoints, dtype=float))
        self.set(bands_num_points=npoints)

    def gen_input(self):
        if self.int_params['num_wann'] is None:
            self.int_params['num_wann']=len(self.initial_basis)
        input_text = ""
        for key, value in self.float_params.items():
            if value is not None:
                input_text += "{0} = {1}\n".format(key, value)
        for key, value in self.string_params.items():
            if value is not None:
                input_text += "{0} = {1}\n".format(key, value)
        for key, value in self.int_params.items():
            if value is not None:
                input_text += "{0} = {1}\n".format(key, value)
        for key, value in self.bool_params.items():
            if value is not None:
                input_text += "{0} = {1}\n".format(key, 'true'
                                                   if True else 'false')
        for key, value in self.list_params.items():
            if value is not None:
                input_text += "{0} = {1}\n".format(key,
                                                   ','.join(map(str, value)))

        # projection block
        if self.projection_dict is not None or self.projection_dict_by_site is not None:
            input_text += '\nbegin projections\n'
            if self.projection_dict is not None:
                for key, value in self.projection_dict.items():
                    if isinstance(value, list):
                        input_text += "{0}: {1}\n".format(key, ','.join(value))
                    elif isinstance(value, dict):
                        for lr, t in value.items():
                            input_text += "{0}:{1}:{2}\n".format(
                                key, ','.join(value), lr)
            for key, value in self.projection_dict_by_site.items():
                for a in key:
                    position = self.atoms.get_positions()[symbol_number(
                        self.atoms)[a]]
                    input_text += "c={0}: {1}\n".format(
                        ','.join(map(str, position)), ','.join(value))
        poses=self.atoms.get_positions()
        sdict=symbol_number(self.atoms)
        for b in self.initial_basis:
            symnum, m, l, r, spin = b
            pos= poses[sdict[symnum]]
            pos_text=','.join(map(str,pos))
            input_text += "c=%s: l=%s, m=%s "%(pos_text, l, m)
            if r is not None:
                input_text += ":r=%s"%(r)
            if spin is not None:
                input_text += "(%s)"%(spin[0])
            input_text += "\t\t# %s|%s|%s|%s\n"%(symnum,revesed_w90_orb_dict[(m,l)],r,spin)
        input_text += 'end projections\n\n'

        #unit cell block
        if self.unit_cell is not None:
            input_text += 'begin unit_cell_cart\n'
            for vec in self.unit_cell:
                input_text += '\t' + '\t'.join(map(str, vec)) + '\n'
            input_text += 'end unit_cell_cart\n\n'

        # atom cordinates
        if self.atoms is not None:
            input_text += '\nbegin atoms_cart\n'
            for sym, pos in zip(self.atoms.get_chemical_symbols(),
                                self.atoms.get_positions()):
                input_text += '{0}\t{1}\n'.format(sym,
                                                  '\t'.join(map(str, pos)))
            input_text += 'end atoms_cart\n\n'

        # kpoints

        if self.mp_grid is not None:
            input_text += 'mp_grid = \t{0}\n'.format('\t'.join(
                map(str, self.mp_grid)))
        if self.kpoints is not None:
            input_text += 'begin kpoints\n'
            for kpt in self.kpoints:
                input_text += '\t' + '\t'.join(map(str, kpt)) + '\n'
            input_text += 'end kpoints\n\n'

        # k path
        if self.kpath is not None:
            input_text += 'begin kpoint_path\n'
            for k_from, k_to in zip(self.kpath[:-1], self.kpath[1:]):
                input_text += "{0} {1}\t{2} {3}\n".format(
                    k_from[0], ' '.join([str(x) for x in k_from[1]]), k_to[0],
                    ' '.join([str(x) for x in k_to[1]]))
            input_text += 'end kpoint_path\n\n'

        self.input_text = input_text
        return self.input_text

    def write_input(self, fname="wannier90.win", basis_fname='basis.txt'):
        """
        write wannier input file.
        """
        #if not self.input_text:
        self.gen_input()
        with open(fname, 'w') as infile:
            infile.write(self.input_text)
        self.write_basis(fname=basis_fname)


def test():
    from ase import Atoms
    atoms = Atoms('MnO', positions=[(0, 0, 0), (1, 1, 1)], cell=np.eye(3))
    wa = wannier_interface(atoms)
    print(atoms)
    wa.set(num_bands=9,  dis_win_min=-3.0, dis_win_max=12.5)
    wa.add_basis('Mn','dxy')
    wa.add_basis('Mn','p')
    wa.add_basis('O',orb='s',spin='up')
    wa.add_basis('O',orb='p',r=2,spin='up')
    print (wa.gen_input())
    #wa.set_projections(projection_dict={'Mn':d_orbs,'O':p_orbs})
    #wa.add_projections_by_site(projections={('Mn1','O1'):eg_orbs},atoms=atoms)
    #wa.atoms=atoms
    #wa.unit_cell=np.eye(3)
    #wa.set_kpath([(0,0,0),[0,0,0.5],[0,0.5,0.5]],['G','L','X'],40)
    #print(wa.gen_input())
    #wa.write_input()



if __name__ == '__main__':
    test()
