#!/usr/bin/env python
from pyDFTutils.ase_utils import symbol_number, symnum_to_sym
import numpy as np
import re
import os
import sys
import pickle
from scipy.integrate import trapz
from collections import OrderedDict

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
    "write_hr",
    "hr_plot"
)

string_keys = (
    "spin",
    "wannier_plot_format",
    "wannier_plot_mode",
    "bands_plot_format", )

list_keys = ("wannier_plot_list", "bands_plot_project", "mp_grid",
             'exclude_bands')

all_keys = float_keys + bool_keys + string_keys + list_keys

s_orbs = ['s']
t2g_orbs = ['dxz', 'dyz', 'dxy']
eg_orbs = ['dz2', 'dx2-y2']
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
    's': (1, 0),
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

reversed_w90_orb_dict = dict(
    list(zip(w90_orb_dict.values(), w90_orb_dict.keys())))


class WannierInput(object):
    def __init__(self,
                 atoms,
                 seed=None,
                 bands=None,
                 spin=0,
                 kpoints=None,
                 spinor=False,
                 write_info=True,
                 **kwargs):
        """
        The wannier.win generator.
        """
        self.unit_cell = None
        self.set_atoms(atoms)
        if seed is None:
            seed = 'wannier90'
        self.seed = seed
        self.bands = bands
        self.spin = spin
        self.spinor = spinor
        self.write_info = write_info
        if self.spinor:
            self.nspinor = 2
        else:
            self.nspinor = 1

        self.projection_dict = None
        self.kpoints = kpoints
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
        self.axis = {}

    def set_kpoints(self, kpoints):
        self.kpoints = kpoints

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.unit_cell = atoms.get_cell()

    def add_basis(self,
                  atom_name,
                  orb=None,
                  m=None,
                  l=None,
                  r=None,
                  spin=None,
                  axis=None):
        """
        set the initial projections for wannier functions.
        Parameters:
        ---------------------
        atoms_name: string
           name of specy or a atoms. Eg. 'Fe' or 'Fe1'.
        orb: string
           name of orbitals, 'dxy', which has the same meaning of a l,m pair.
        m, l, r: int
          quantum numbers. m and r can be none. If m is None, it means all possible m for a given l. r=None means r is default 1.Note r is not the radial quantum number. It labels 1, 2, 3. Eg. if there're two 4p and 5p orbitals, they can be labled as r=1 and r=2 respectively.
        spin: string or None
           "up"|"down"|None
        Results:
        ---------
        [aname, mi, l, r, spin] added to self.initial_basis (list)
        TODO: Is nwann correct for spin polarized structure? Check this.
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

        if '0' <= atom_name[-1] <= '9':  # single atoms
            sdict = symbol_number(self.atoms)
            for mi in mlist:
                self.initial_basis.append([atom_name, mi, l, r, spin])
                if axis is not None:
                    self.axis[atom_name] = axis
                    #print("Axis added")
        else:
            sdict = symbol_number(self.atoms)
            for aname in sdict:
                if symnum_to_sym(aname) == atom_name:
                    for mi in mlist:
                        self.initial_basis.append([aname, mi, l, r, spin])
                        if axis is not None:
                            self.axis[aname] = axis
                            print("Axis added")

    def add_basis_from_dict(self, atom_map=None, conf_dict=None, band='v+c'):
        """
        Parameters:
        -----------
        atom_map: dict
          a dict map chemical symbols or chemical_symbols+number to  label pairs. eg. {'Bi':3 , 'Fe1':'3_up', 'Fe2':'3_dn','O':-2}. If the ion is not magnetic, usually use its valence as the label. The label does not have to be physically meaningful.
        conf_dict: dict
          a dict map the (elem, label) tuple -> electron configuration. eg. {(Mn,'3_up'): [(dxy, 3, 1, 'up'), (dyz,),}
        band: string
          'v'|'v+c'
        """
        # add spin=None to conf_dict if spin is not given.
        for key in conf_dict:
            val = conf_dict[key]
            conf_dict[key] = [
                list(v) + [None]
                if not (v[-1] == 'up' or v[-1] == 'dn' or v[-1] is None) else
                list(v) for v in val
            ]
        for s in atom_map:
            confs = conf_dict[(s, atom_map[s])]
            min_r = min([o[2] for o in confs])
            for conf in confs:
                if isinstance(conf[0], int) or conf[0] is None:  # (m, l ,...)
                    if len(conf) == 4:
                        m, l, r, occ = conf
                        spin = None
                    elif len(conf) == 5:
                        m, l, r, occ, spin = conf
                    if occ > 0 or band == 'v+c':
                        self.add_basis(
                            atom_name=s, m=m, l=l, r=r - min_r + 1, spin=spin)
                else:  # ('dxy'...)
                    if len(conf) == 3:
                        orb_name, r, occ = conf
                        spin = None
                    elif len(conf) == 4:
                        orb_name, r, occ, spin = conf
                    print(conf)
                    if occ > 0 or band == 'v+c':
                        self.add_basis(
                            atom_name=s,
                            orb=orb_name,
                            r=r - min_r + 1,
                            spin=spin)

    def get_basis(self, atoms=None):
        """
        get the name of the basis.
        """
        if self.atoms is None:
            if atoms is None:
                raise ValueError("atoms should be specified.")
            else:
                self.atoms = atoms
        self.basis = []
        for b in self.initial_basis:
            symnum, m, l, r, spin = b
            bname = '|'.join(
                [symnum, reversed_w90_orb_dict[(m, l)], str(r), str(spin)])
            self.basis.append(bname)
        return self.basis

    def write_basis(self, fname='basis.txt'):
        """
        write basis to a file. default fname is basis.txt.
        """
        self.get_basis()
        with open(fname, 'w') as myfile:
            for i, b in enumerate(self.basis):
                myfile.write(str(b) + '\t' + str(i + 1) + '\n')

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
        self.kpath = list(zip(labels, np.array(kpoints, dtype=float)))
        self.set(bands_num_points=npoints)

    def gen_input(self):
        if self.int_params['num_wann'] is None:
            self.int_params['num_wann'] = len(self.initial_basis)*self.nspinor
        print(self.int_params)
        print(self.float_params)
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
            if value is not None and key not in ['mp_grid']:
                input_text += "{0} = {1}\n".format(key,
                                                   ','.join(map(str, value)))
        print(input_text)

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
        poses = self.atoms.get_positions()
        sdict = symbol_number(self.atoms)
        for b in self.initial_basis:
            symnum, m, l, r, spin = b
            pos = poses[sdict[symnum]]
            pos_text = ','.join(map(str, pos))
            input_text += "c=%s: l=%s, mr=%s " % (pos_text, l, m)
            if r is not None:
                input_text += ":r=%s" % (r)
            if spin is not None:
                input_text += "(%s)" % (spin[0])
            if symnum in self.axis:
                #input_text += ":z=%s"%(','.join(map(str, self.axis[symnum])))
                input_text += "%s" % (self.axis[symnum])
            input_text += "\t\t# %s|%s|%s|%s\n" % (
                symnum, reversed_w90_orb_dict[(m, l)], r, spin)
        input_text += 'end projections\n\n'

        if self.write_info:
            # unit cell block
            if self.unit_cell is not None:
                input_text += 'begin unit_cell_cart\n'
                for vec in self.unit_cell:
                    input_text += '\t' + '\t'.join(map(str, vec)) + '\n'
                input_text += 'end unit_cell_cart\n\n'

            # atom cordinates
            if self.atoms is not None:
                input_text += '\nbegin atoms_cart\n'
                for sym, pos in list(
                        zip(self.atoms.get_chemical_symbols(),
                            self.atoms.get_positions())):
                    input_text += '{0}\t{1}\n'.format(sym,
                                                      '\t'.join(map(str, pos)))
                input_text += 'end atoms_cart\n\n'

            # kpoints

            if 'mp_grid' in self.list_params:
                input_text += 'mp_grid = \t{0}\n'.format(
                    '\t'.join(map(str, self.list_params['mp_grid'])))
            if self.kpoints is not None:
                input_text += 'begin kpoints\n'
                for kpt in self.kpoints:
                    input_text += '\t' + '\t'.join(map(str, kpt)) + '\n'
                input_text += 'end kpoints\n\n'

        # k path
        if self.kpath is not None:
            input_text += 'begin kpoint_path\n'
            for k_from, k_to in list(zip(self.kpath[:-1], self.kpath[1:])):
                input_text += "{0} {1}\t{2} {3}\n".format(
                    k_from[0], ' '.join([str(x) for x in k_from[1]]), k_to[0],
                    ' '.join([str(x) for x in k_to[1]]))
            input_text += 'end kpoint_path\n\n'

        self.input_text = input_text
        return self.input_text

    def write_input(self,
                    prefix="wannier90",
                    basis_fname='basis.txt',
                    save_dict=True,
                    spin=False,
                    dft_code="vasp",
                    ):
        """
        write wannier input file.
        """
        # if not self.input_text:
        self.gen_input()
        if spin:
            if dft_code.lower() == 'vasp':
                fname_up = "wannier90_up.win"
                fname_dn = "wannier90_dn.win"
            elif dft_code.lower() == "abinit":
                fname_up = prefix+'o_w90_up.win'
                fname_dn = prefix+'o_w90_down.win'
            with open("wannier90.win", 'w') as infile:
                infile.write(self.input_text)
                self.write_basis(fname=basis_fname)
            with open(fname_up, 'w') as infile:
                infile.write(self.input_text)
                self.write_basis(fname=basis_fname)
            with open(fname_dn, 'w') as infile:
                infile.write(self.input_text)
                self.write_basis(fname=basis_fname)
        else:
            fname = prefix+'.win'
            with open(fname, 'w') as infile:
                infile.write(self.input_text)
                self.write_basis(fname=basis_fname)
        if save_dict:
            fname_pickle = prefix+'_wannier'
            with open('%s.pickle' % fname_pickle, 'wb') as pfile:
                pickle.dump(self, pfile)

    def get_nwann(self):
        """
        get number of wannier functions
        """
        if self.int_params['num_wann'] is None:
            self.int_params['num_wann'] = len(self.initial_basis)*self.nspinor
        return self.int_params['num_wann']


def pdos_band(input_fname, iwann, dos=True, band=True, restart='plot'):
    """
    calculate projected dos and projected band.
    """
    os.system('cp %s %s.bak' % (input_fname, input_fname))
    pre = os.path.splitext(input_fname)[0]
    valdict = {}
    if restart:
        valdict['restart'] = restart
    if band:
        valdict['bands_plot_project'] = iwann
        valdict['bands_plot'] = 'true'
        text = replace_value_file(input_fname, valdict)
        with open(input_fname, 'w') as myfile:
            myfile.write(text)
        os.system('bash -c "wannier90.x %s"' % input_fname)
        os.system('cp %s_band.dat %s_band_%s.dat' % (pre, pre, iwann))
    os.system('cp %s %s.band' % (input_fname, input_fname))
    os.system('cp %s.bak %s' % (input_fname, input_fname))
    if dos:
        valdict['dos'] = 'true'
        valdict['dos_kmesh'] = 10
        valdict['dos_project'] = iwann
        text = replace_value_file(input_fname, valdict)
        with open(input_fname, 'w') as myfile:
            myfile.write(text)
        os.system('bash -c "postw90.x %s"' % input_fname)
        os.system('cp %s-dos.dat %s_dos_%s.dat' % (pre, pre, iwann))
    os.system('cp %s %s.dos' % (input_fname, input_fname))
    os.system('cp %s.bak %s' % (input_fname, input_fname))


def find_value_line(lines, key):
    """
    """
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


def run_wannier(command=None, spin=None, copy_win=True, zenobe=False):
    """
    run wannier90.
    """
    if command is None:
        command = 'wannier90.x'
    if spin is None:
        name = 'wannier90'
    elif spin == 'up':
        name = 'wannier90.up'
        spinline = 'spin = up\n'
    elif spin == 'dn' or spin == 'down':
        name = 'wannier90.dn'
        spinline = 'spin = down\n'
    if spin is not None and copy_win:
        with open("%s.win" % name, 'w') as myfile:
            myfile.write(spinline)
            str = open('wannier90.win').read()
            myfile.write(str)
        #os.system('cp wannier90.win %s.win' % name)
    if not zenobe:
        os.system("%s %s" % (command, name))
    else:
        from pyDFTutils.queue.commander import zenobe_run_wannier90
        zenobe_run_wannier90(spin=spin)
    if spin is not None:
        if not os.path.exists(name):
            os.mkdir(name)


def test():
    from ase import Atoms
    atoms = Atoms('MnO', positions=[(0, 0, 0), (1, 1, 1)], cell=np.eye(3))
    wa = wannier_input(atoms)
    print(atoms)
    wa.set(num_bands=9, dis_win_min=-3.0, dis_win_max=12.5)
    wa.add_basis('Mn', 'dxy')
    wa.add_basis('Mn', 'p')
    wa.add_basis('O', orb='s', spin='up')
    wa.add_basis('O', orb='p', r=2, spin='up')
    print(wa.gen_input())
    # wa.atoms=atoms
    # wa.unit_cell=np.eye(3)
    # wa.set_kpath([(-1,0,0),[0,0,0.5],[0,0.5,0.5]],['G','L','X'],40)
    # print(wa.gen_input())
    # wa.write_input()


def wannier_default(name='BaTiO3'):
    from ase_utils.cubic_perovskite import gen_primitive
    z_db = {'Ba': 10, 'Ti': 12, 'O': 6}
    # orb_db is (symbol, label), label can be valence, but also others,like 'high_spin'
    orb_db = {  # (symbol, valence): [(m, l, r, occ),... ]
        ('Ba', 2): [(None, 0, 5, 2), (None, 1, 5, 6), (None, 0, 6, 0)],
        ('Ti', 4):
        [(None, 0, 3, 2), (None, 1, 3, 6), (None, 2, 3, 0), (None, 0, 4, 0)],
        ('O', -2): [(None, 0, 2, 2), (None, 1, 2, 6)],
    }
    atoms = gen_primitive('BaTiO3', latticeconstant=3.946, mag_order='PM')
    vals = {'Ba': 2, 'Ti': 4, 'O': -2}
    wa = wannier_input(atoms)

    syms = atoms.get_chemical_symbols()
    nwann = sum((z_db[s] for s in syms))

    print(wa.gen_input())
    return wa


def wannier_closeshell(atoms, val_dict=None, band='v+c'):
    from ase_utils.cubic_perovskite import gen_primitive
    from data.ONCV_PBEsol_conf import ONCV_PBEsol_conf
    from psp import gen_ion_conf_dict
    econf = ONCV_PBEsol_conf
    vals = {
        'Ba': 2,
        'Ti': 4,
        'O': -2,
        'Ca': 2,
        'Sr': 2,
        'Pb': 2,
        'Sn': 2,
        'Zr': 2,
        'Li': 1,
        'Nb': 5
    }
    if val_dict is not None:
        for key in val_dict:
            vals[key] = val_dict[key]
    econf = gen_ion_conf_dict(vals, econf)
    wa = wannier_input(atoms)
    wa.add_basis_from_dict(atom_map=vals, conf_dict=econf, band=band)
    syms = atoms.get_chemical_symbols()
    print(wa.gen_input())
    # print wa.get_nwann()
    return wa


def replace_value(text, valdict, position='start'):
    """
    Replace text line a=b with a=c (if a=... exist, else add line a=c)
    Params:
    -------------
    text: string
    valdict: dictionary of a:c. Note that c will be written as str(c).
       If the default is not the same as python format
       e.g. you may like to write [1,2,3] as 1 2 3,
       then please use "1 2 3", rather than using [1,2,3]
    position: 'start'| 'end'. If the key do not exist already, write at the beginning or end of file.
    """
    lines = text.split('\n')
    newlines = []
    for line in lines:
        indict = False
        for key in valdict:
            if re.findall(r'\s*%s\s*=' % key, line) != []:
                indict = True
                found_key = key
                newlines.append('%s = %s\n' % (key, str(valdict[key])))
        if not indict:
            newlines.append(line + '\n')
        else:
            valdict.pop(found_key)

    for key in valdict:
        if position == 'start':
            newlines.insert(0, '%s = %s\n' % (key, str(valdict[key])))
        else:
            newlines.append('%s = %s\n' % (key, str(valdict[key])))

    return ''.join(newlines)


def replace_value_file(fname, valdict, position='start', rewrite=True):
    """
    same as replace_value, but replace text from file instead of replacing a text
    """
    with open(fname) as myfile:
        text = myfile.read()
    new_text = replace_value(text, valdict, position=position)
    if rewrite:
        with open(fname, 'w') as myfile:
            myfile.write(new_text)
    return new_text


def occupation(fname, efermi):
    data = np.loadtxt(fname)
    # plt.plot(data[:,0]-efermi,data[:,1])
    return trapz(data[:, 1][data[:, 0] - efermi < 0],
                 data[:, 0][data[:, 0] - efermi < 0]) / 2


def read_basis(fname):
    """
    return basis names from file (often named as basis.txt). Return a dict. key: basis name. value: basis index, from 0
    """
    bdict = OrderedDict()
    if fname.endswith('.win'):
        with open(fname) as myfile:
            inside = False
            iline = 0
            for line in myfile.readlines():
                if line.strip().startswith('end projections'):
                    inside = False
                if inside:
                    a = line.strip().split('#')
                    assert len(
                        a) == 2, "The format should be .... # label_of_basis"
                    bdict[a[-1].strip()] = iline
                    iline += 1
                if line.strip().startswith('begin projections'):
                    inside = True
    else:
        with open(fname) as myfile:
            for iline, line in enumerate(myfile.readlines()):
                a = line.strip().split()
                if len(a) != 0:
                    bdict[a[0]] = iline
    return bdict


# wannier_default()
# wannier_closeshell()
if __name__ == '__main__':
    wannier_closeshell()
