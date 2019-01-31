#! /usr/bin/env python

from pyDFTutils.vasp.vasp_utils import read_poscar_and_unsort
from ase.calculators.vasp import Vasp
from ase.dft.kpoints import get_bandpath
import matplotlib.pyplot as plt
import os
from os.path import join
import sys
from ase.utils import devnull, basestring
import numpy as np
import tempfile
import ase.io
from pyDFTutils.ase_utils.symbol import symbol_number
import socket
from shutil import copyfile, move

default_pps_1 = {
    'Pr': 'Pr_3',
    'Ni': 'Ni',
    'Yb': 'Yb_3',
    'Pd': 'Pd',
    'Pt': 'Pt',
    'Ru': 'Ru_pv',
    'S': 'S',
    'Na': 'Na_pv',
    'Nb': 'Nb_sv',
    'Nd': 'Nd_3',
    'C': 'C',
    'Li': 'Li_sv',
    'Pb': 'Pb_d',
    'Y': 'Y_sv',
    'Tl': 'Tl_d',
    'Lu': 'Lu_3',
    'Rb': 'Rb_sv',
    'Ti': 'Ti_sv',
    'Te': 'Te',
    'Rh': 'Rh_pv',
    'Tc': 'Tc_pv',
    'Ta': 'Ta_pv',
    'Be': 'Be',
    'Sm': 'Sm_3',
    'Ba': 'Ba_sv',
    'Bi': 'Bi_d',
    'La': 'La',
    'Ge': 'Ge_d',
    'Po': 'Po_d',
    'Fe': 'Fe',
    'Br': 'Br',
    'Sr': 'Sr_sv',
    'Pm': 'Pm_3',
    'Hf': 'Hf_pv',
    'Mo': 'Mo_sv',
    'At': 'At_d',
    'Tb': 'Tb_3',
    'Cl': 'Cl',
    'Mg': 'Mg',
    'B': 'B',
    'F': 'F',
    'I': 'I',
    'H': 'H',
    'K': 'K_sv',
    'Mn': 'Mn_pv',
    'O': 'O',
    'N': 'N',
    'P': 'P',
    'Si': 'Si',
    'Sn': 'Sn_d',
    'W': 'W_sv',
    'V': 'V_sv',
    'Sc': 'Sc_sv',
    'Sb': 'Sb',
    'Os': 'Os',
    'Dy': 'Dy_3',
    'Se': 'Se',
    'Hg': 'Hg',
    'Zn': 'Zn',
    'Co': 'Co',
    'Ag': 'Ag',
    'Re': 'Re',
    'Ca': 'Ca_sv',
    'Ir': 'Ir',
    'Eu': 'Eu_3',
    'Al': 'Al',
    'Ce': 'Ce_3',
    'Cd': 'Cd',
    'Ho': 'Ho_3',
    'As': 'As',
    'Gd': 'Gd_3',
    'Au': 'Au',
    'Zr': 'Zr_sv',
    'Ga': 'Ga_d',
    'In': 'In_d',
    'Cs': 'Cs_sv',
    'Cr': 'Cr_pv',
    'Tm': 'Tm_3',
    'Cu': 'Cu',
    'Er': 'Er_3'
}

default_pps = {}
for p in default_pps_1:
    v = default_pps_1[p]
    default_pps[p] = v[len(p):]


class myvasp(Vasp):
    def __init__(self,
                 restart=None,
                 output_template='vasp',
                 track_output=False,
                 **kwargs):

        self.force_no_calc = False
        self.vca=None

        self.tempdir = None

        Vasp.__init__(
            self,
            restart=None,
            output_template='vasp',
            track_output=False,
            **kwargs)
        self.commander = None

    def set_commander(self, commander):
        self.commander = commander
    def set_vca(self, vca):
        self.vca=vca
    def run(self):
        """Method which explicitely runs VASP."""

        if self.track_output:
            self.out = self.output_template + str(self.run_counts) + '.out'
            self.run_counts += 1
        else:
            self.out = self.output_template + '.out'
        stderr = sys.stderr
        p = self.input_params
        if p['txt'] is None:
            sys.stderr = devnull
        elif p['txt'] == '-':
            pass
        elif isinstance(p['txt'], basestring):
            sys.stderr = open(p['txt'], 'w')

        if self.commander is not None:
            exitcode = self.commander.run()
        elif 'VASP_COMMAND' in os.environ:
            vasp = os.environ['VASP_COMMAND']
            exitcode = os.system('%s > %s' % (vasp, self.out))
        elif 'VASP_SCRIPT' in os.environ:
            vasp = os.environ['VASP_SCRIPT']
            locals = {}
            exec(compile(open(vasp).read(), vasp, 'exec'), {}, locals)
            exitcode = locals['exitcode']
        else:
            raise RuntimeError('Please set either VASP_COMMAND'
                               ' or VASP_SCRIPT environment variable')
        sys.stderr = stderr
        if exitcode != 0:
            raise RuntimeError('Vasp exited with exit code: %d.  ' % exitcode)

    def magnetic_calculation(self, do_nospin=True):
        self.set(ispin=1, istart=0)
        self.calculate(self.atoms)
        self.set(
            ispin=2,
            istart=0,
            icharg=1,
            nelm=150,
            amix=0.2,
            amix_mag=0.8,
            bmix=0.0001,
            bmix_mag=0.0001,
            maxmix=20)
        self.calculate(self.atoms)

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = [
            'CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR', 'DOSCAR',
            'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT',
            'POTCAR', 'vasprun.xml', 'WAVECAR', 'XDATCAR', 'PROCAR',
            'ase-sort.dat', 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2'
        ]
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        if self.tempdir is not None:
            for f in files:
                try:
                    os.remove(os.path.join(self.tempdir, f))
                except OSError:
                    pass

    def myrelax_calculation(self, do_nospin=False, pre_relax=True, pre_relax_method='dampedmd'):
        """
        a optimized stratigey to do the relax.
        do_nospin: if do_nospin is True, a non-spin-polarized relaxation is done first.
        """
        ispin = self.int_params['ispin']
        nelmdl = self.int_params['nelmdl']
        ibrion = self.int_params['ibrion']
        sigma = self.float_params['sigma']
        smass = self.float_params['smass']
        #potim = self.float_params['potim']
        if sigma is None:
            sigma = 0.1
        ediff = self.exp_params['ediff']
        if ediff is None:
            ediff = 1e-4
        ediffg = self.exp_params['ediffg']
        if ediffg is None:
            ediffg = -0.01

        ldipol = self.bool_params['ldipol']
        if ldipol is None:
            ldipol = False
        nsw = self.int_params['nsw']

        #first do this
        if pre_relax and pre_relax_method=='cg':
            self.set(
                nelmdl=6,
                nelmin=-9,
                ediff=3e-3,
                ediffg=-0.3,
                nsw=30,
                ibrion=1,
                sigma=sigma * 3,
                ldipol=False,
                maxmix=-20)

        if pre_relax and pre_relax_method=='dampedmd':
            self.set(
                nelmdl=6,
                nelmin=-9,
                ediff=3e-3,
                ediffg=-0.3,
                nsw=30,
                ibrion=3,
                sigma=sigma * 3,
                ldipol=False,
                potim=0.1,
                smass=1.1,
                maxmix=-20)
            if do_nospin:
                print("----------------------------------------------------")
                self.set(ispin=1)
            self.calculate(self.atoms)
            self.read_contcar(filename='CONTCAR')

        # if do_nospin
        if do_nospin:
            self.set(
                ispin=1,
                nelmdl=nelmdl,
                nelmin=5,
                ediff=ediff,
                ediffg=ediffg,
                ibrion=ibrion,
                sigma=sigma,
                ldipol=ldipol,
                nsw=30,
                maxmix=40)
            #self.read_contcar(filename='CONTCAR')
            self.set(istart=1)
            self.calculate(self.atoms)
            self.read_contcar(filename='CONTCAR')

        # then increase the accuracy.
        self.set(
            ispin=ispin,
            nelmdl=nelmdl,
            nelmin=5,
            ediff=ediff,
            ediffg=ediffg,
            ibrion=ibrion,
            sigma=sigma,
            ldipol=ldipol,
            nsw=nsw,
            maxmix=40,
            smass=smass,
            nfree=15)

        #self.read_contcar(filename='CONTCAR')
        self.set(istart=1)
        self.calculate(self.atoms)
        self.read_contcar(filename='CONTCAR')

        if not os.path.exists('RELAX'):
            os.mkdir('RELAX')
        for f in ['POSCAR', 'OUTCAR', 'EIGENVAL', 'CONTCAR', 'INCAR', 'log']:
            copyfile(f, os.path.join('RELAX', f))
        return self.atoms

    def band_calculation(self, special_kpoints, names, npoints=60):
        """
        calculate the band structure.
        """
        self.set(icharg=11, nsw=0, ibrion=-1, ismear=0)
        cell = self.atoms.get_cell()
        kpts, xcords, sp_xcords = get_bandpath(
            special_kpoints, cell, npoints=npoints)

        self.band_xs = xcords
        self.band_special_xs = sp_xcords
        self.special_kpts_names = names
        self.set(kpts=kpts, reciprocal=True)
        self.calculate(self.atoms)
        if not os.path.exists('BAND'):
            os.mkdir('BAND')
        for f in ['POSCAR', 'OUTCAR', 'EIGENVAL', 'PROCAR', 'INCAR', 'log']:
            if os.path.exists(f):
                copyfile(f, os.path.join('BAND', f))
        self.plot_bands()

    def scf_calculation(self, ismear=-5, sigma=0.1, istart=1):
        """
        scf calculation
        """
        self.set(
            nsw=0, ibrion=1, ismear=ismear, nedos=501, sigma=sigma, nelmdl=-10, istart=istart)

        self.calculate(self.atoms)

        if not os.path.exists('SCF'):
            os.mkdir('SCF')
        for f in ['POSCAR', 'OUTCAR', 'EIGENVAL', 'DOSCAR', 'INCAR', 'log']:
            copyfile(f, os.path.join('SCF', f))
        myfile = open('SCF/result.txt', 'w')

        try:
            eg = self.get_bandgap()
        except Exception:
            eg = np.NaN
        try:
            efermi = self.get_fermi_level()
        except Exception:
            efermi = np.NaN
        try:
            e_free, e_zero = self.read_energy()
        except Exception:
            e_free, ezero = np.NaN, np.NaN

        myfile.write(" energy=%s\n eg=%s\n efermi=%s\n" % (e_zero, eg, efermi))
        myfile.close()

    def dos_calculation(self, ismear=-5, sigma=0.05):
        """
        calculate the total density of state.
        """
        self.set(
            icharg=11,
            nsw=0,
            ibrion=-1,
            ismear=ismear,
            nedos=1001,
            sigma=sigma,
            nelmdl=1)

        self.calculate(self.atoms)

        if not os.path.exists('DOS'):
            os.mkdir('DOS')
        for f in [
                'POSCAR', 'OUTCAR', 'EIGENVAL', 'PROCAR', 'DOSCAR', 'INCAR',
                'log'
        ]:
            copyfile(f, os.path.join('DOS', f))
        plot_dos(output='DOS/DOS.png')

    def ldos_calculation(self, ismear=-5, sigma=0.05):
        """
        calculate the local density of state of all the ions.
        """
        self.set(
            icharg=10,
            nsw=0,
            ismear=ismear,
            nedos=1001,
            lorbit=10,
            sigma=sigma)

        self.calculate(self.atoms)
        if not os.path.exists('LDOS'):
            os.mkdir('LDOS')
        for f in [
                'POSCAR', 'OUTCAR', 'EIGENVAL', 'PROCAR', 'DOSCAR', 'INCAR',
                'log'
        ]:
            copyfile(f, os.path.join('LDOS', f))

    def pdos_calculation(self, ismear=-5, sigma=0.05):
        """
        calculate the local density of state of all the ions.
        """
        self.set(
            icharg=11,
            nsw=0,
            ismear=ismear,
            nedos=1001,
            lorbit=11,
            sigma=sigma)
        self.calculate(self.atoms)

        if not os.path.exists('PDOS'):
            os.mkdir('PDOS')
        for f in [
                'POSCAR', 'OUTCAR', 'EIGENVAL', 'PROCAR', 'DOSCAR', 'INCAR',
                'log'
        ]:
            copyfile(f, os.path.join('PDOS', f))

    def static_calculation(self, ismear=-5, sigma=0.05):
        self.scf_calculation(ismear=ismear, sigma=sigma)
        self.ldos_calculation(ismear=ismear, sigma=sigma)
        self.pdos_calculation(ismear=ismear, sigma=sigma)

    def set_wannier_input(self, filename='wannier90.win'):
        """
        set the wannier input file.
        """
        with open(filename, 'w') as myfile:
            lines = myfile.readlines()
            #TODO implement wannier input

    def wannier_calculation(self,
                            lwrite_unk=True,
                            projection=None,
                            frozen_window=None,
                            window=None,
                            *kwargs):
        """
        wannier calculation
        First, without projection
        Then, disentanglement
        :param lwrite_unk : True|False
        :param projection : a string
        :param frozen_window : (a,b) a and b are floats
        :param window : (a,b) a and b are floats

        """
        npar = self.int_params['npar']

        # 1. wannier without projection
        if os.path.exists('wannier90.win'):
            move('wannier90.win', 'wannier90.win.bak')

        self.set(
            lwannier90=True,
            lwrite_unk=lwrite_unk,
            lwrite_mmn_amn=True,
            npar=None)
        self.set(icharg=11, nsw=0, nedos=1001, lorbit=10)
        self.calculate(self.atoms)

        self.set(lwannier90=False, npar=npar)

        if not os.path.exists('Wannier'):
            os.mkdir('Wannier')
        for f in [
                'POSCAR', 'OUTCAR', 'EIGENVAL', 'PROCAR', 'DOSCAR', 'INCAR',
                'log'
        ]:
            copyfile(f, os.path.join('Wannier', f))
            os.system('cp ./wannier* Wannier')

    def get_eig_array(self):
        nbands = self.get_number_of_bands()

        kpts = self.get_ibz_k_points()
        nkpts = len(kpts)

        if self.get_number_of_spins() == 1:
            eigs = np.empty((nbands, nkpts), float)
            for k in range(nkpts):
                eigs[:, k] = self.get_eigenvalues(kpt=k)
        else:
            eigs = np.empty((nbands, nkpts, 2), float)
            for spin in [0, 1]:
                for k in range(nkpts):
                    eigs[:, k, spin] = self.get_eigenvalues(kpt=k, spin=spin)

        return eigs

    def read_charges(self, orbital=None):
        charges = np.zeros(len(atoms))
        n = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('total charge') > -1:
                for m in range(len(atoms)):
                    charges[m] = float(lines[n + m + 4].strip().split()[-1])
            n += 1
        return np.array(charges)[self.resort]

    def read_magnetic_moment(self):
        n = 0
        for line in open('OUTCAR', 'r'):
            if line.rfind('number of electron  ') > -1:
                if line.strip().split()[-1] == 'magnetization':
                    magnetic_moment = 0
                    # what should be done ??? if the end of the line is magnetization,seems to be a bug in vasp.
                elif line.strip().split()[-2] == 'magnetization':
                    magnetic_moment = float(line.strip().split()[-1])
                else:
                    raise Exception(
                        "Can not read magnetization, the line is %s" % line)
            n += 1
        return magnetic_moment

    def read_magnetic_moments(self, atoms, orbital=None):
        magnetic_moments = np.zeros(len(atoms))
        n = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('magnetization (x)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m] = float(
                        lines[n + m + 4].strip().split()[-1])
            n += 1
        return np.array(magnetic_moments)[self.resort]

    def read_magnetic_moments_nc(self, atoms):
        magnetic_moments = np.zeros(len(atoms), 3)
        n = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('magnetization (x)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m, 0] = float(
                        lines[n + m + 4].strip().split()[-1])
            n += 1

            if line.rfind('magnetization (y)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m, 1] = float(
                        lines[n + m + 4].strip().split()[-1])
            n += 1

            if line.rfind('magnetization (z)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m, 2] = float(
                        lines[n + m + 4].strip().split()[-1])
            n += 1

        return np.array(magnetic_moments)[self.resort]

    def plot_bands(self,
                   window=None,
                   output_filename=None,
                   show=False,
                   spin=0,
                   yrange=[-8, 5],
                   efermi=None):
        """
        plot the bands.
        window: (Emin,Emax), the range of energy to be plotted in the figure.
        speicial_kpts_name
        efermi: if efermi is not None, use the given efermi value.
        """
        if output_filename is None:
            output_filename = 'bands.png'

        eigenvalues = self.get_eig_array()
        plt.clf()
        if window is not None:
            plt.ylim(window[0], window[1])
        if self.get_number_of_spins() == 1:

            for i in range(self.nbands):
                band_i = eigenvalues[i, :] - self.get_fermi_level()
                plt.plot(self.band_xs, band_i)
        else:

            for i in range(self.nbands):
                band_i = eigenvalues[i, :, spin] - self.get_fermi_level()
                plt.plot(self.band_xs, band_i)
        plt.xlabel('K-points')
        plt.ylabel('$Energy-E_{fermi} (eV)$')
        plt.ylim(yrange)
        plt.axhline(0, color='black', linestyle='--')
        if self.special_kpts_names is not None:
            plt.xticks(self.band_special_xs, self.special_kpts_names)

        if output_filename is not None:
            plt.savefig(output_filename)
        if show:
            plt.show()

    def read_contcar(self, filename='CONTCAR'):
        """
        read from contcar and set the positions and cellpars. Note that in vasp, "sort" is True. so read and !!!!unsort!!!!.
        """
        symbols, positions, cell = read_poscar_and_unsort(filename=filename)
        self.atoms.set_cell(cell)
        self.atoms.set_positions(positions)

    def set_force_no_calc(self, nocalc=True):
        """
        set nocalc=True so that no calculation is really done.
        """
        self.force_no_calc = True

    def calculate(self, atoms):
        """Generate necessary files in the working directory and run VASP.

        The method first write VASP input files, then calls the method
        which executes VASP. When the VASP run is finished energy, forces,
        etc. are read from the VASP output.
        """

        cwd = os.getcwd()
        if os.path.exists('/data') and self.tempdir is None:
            self.tempdir = tempfile.mkdtemp(prefix='vasptmp_', dir='/data')
            with open('./syncpath.txt', 'w') as myfile:
                myfile.write('%s:%s\n' % (socket.gethostname(), self.tempdir))

        if self.tempdir is not None:
            print(('rsync -a ./* %s' % self.tempdir))
            os.system('rsync -a ./* %s' % self.tempdir)
            os.chdir(self.tempdir)
            print(("Changing path to %s" % self.tempdir))

        # Initialize calculations
        self.initialize(atoms)

        # Write input
        from ase.io.vasp import write_vasp
        write_vasp('POSCAR', self.atoms_sorted, symbol_count=self.symbol_count)
        self.write_incar(atoms)
        self.write_potcar()
        self.write_kpoints()
        self.write_sort_file()

        # Execute VASP
        if not self.force_no_calc:
            self.run()
        # Read output
        atoms_sorted = ase.io.read('CONTCAR', format='vasp')

        if (not (self.int_params['ibrion'] is None or self.int_params['nsw'] is None)) and (self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0):
            # Update atomic positions and unit cell with the ones read
            # from CONTCAR.
            atoms.positions = atoms_sorted[self.resort].positions
            atoms.cell = atoms_sorted.cell
        self.converged = self.read_convergence()
        self.set_results(atoms)
        if self.tempdir is not None:
            print("Sync back")
            print(('rsync -a %s/* %s' % (self.tempdir, cwd)))
            os.system('rsync -a %s/* %s' % (self.tempdir, cwd))
            os.chdir(cwd)

    def read_ibz_kpoints(self):
        ibz_kpts = []
        n = 0
        i = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('k-points in units of 2pi/SCALE and weight:') > -1:
                m = n + 1
                while i == 0:
                    ibz_kpts.append(
                        [float(lines[m].split()[p]) for p in range(3)])
                    m += 1
                    if lines[m] == ' \n':
                        i = 1
            if i == 1:
                continue
            n += 1
        ibz_kpts = np.array(ibz_kpts)
        return np.array(ibz_kpts)

    def get_homo_and_lumo(self):
        nspin = self.get_number_of_spins()
        nkpts = len(self.get_ibz_k_points())
        occ_eigs = []
        unocc_eigs = []
        for ispin in range(nspin):
            for ikpt in range(nkpts):
                occs = self.get_occupation_numbers(kpt=ikpt, spin=ispin)
                eigs = self.get_eigenvalues(kpt=ikpt, spin=ispin)
                for occ, eig in zip(occs, eigs):
                    if occ < 0.001:
                        unocc_eigs.append(eig)
                    else:
                        occ_eigs.append(eig)
        homo = max(occ_eigs)
        lumo = min(unocc_eigs)
        return homo, lumo

    def get_bandgap(self):
        homo, lumo = self.get_homo_and_lumo()
        if lumo > homo:
            return lumo - homo
        else:
            return 0.0

    def read_convergence(self):
        """
        A workout for a bug in vasp5.2.2 that b in outcar might be wrong.
        Method that checks whether a calculation has converged."""
        converged = None
        # First check electronic convergence
        for line in open('OUTCAR', 'r'):
            if line.rfind('EDIFF  ') > -1:
                ediff = float(line.split()[2])
            if line.rfind('total energy-change') > -1:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                try:
                    b = float(split[1].split('(')[1][0:-2])
                except ValueError:
                    b = a
                if [abs(a), abs(b)] < [ediff, ediff]:
                    converged = True
                else:
                    converged = False
                    continue
        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled
        if (self.int_params['ibrion'] in [1, 2, 3] and
                self.int_params['nsw'] not in [0]):
            if not self.read_relaxed():
                converged = False
            else:
                converged = True
        return converged

    def write_incar(self, atoms, directory='./', **kwargs):
        """Writes the INCAR file."""
        # jrk 1/23/2015 I added this flag because this function has
        # two places where magmoms get written. There is some
        # complication when restarting that often leads to magmom
        # getting written twice. this flag prevents that issue.
        magmom_written = False
        vca_written=False
        incar = open(join(directory, 'INCAR'), 'w')
        incar.write('INCAR created by Atomic Simulation Environment\n')
        for key, val in self.float_params.items():
            if val is not None:
                incar.write(' %s = %5.6f\n' % (key.upper(), val))
        for key, val in self.exp_params.items():
            if val is not None:
                incar.write(' %s = %5.2e\n' % (key.upper(), val))
        for key, val in self.string_params.items():
            if val is not None:
                incar.write(' %s = %s\n' % (key.upper(), val))
        for key, val in self.int_params.items():
            if val is not None:
                incar.write(' %s = %d\n' % (key.upper(), val))
                if key == 'ichain' and val > 0:
                    incar.write(' IBRION = 3\n POTIM = 0.0\n')
                    for key, val in self.int_params.items():
                        if key == 'iopt' and val is None:
                            print('WARNING: optimization is '
                                  'set to LFBGS (IOPT = 1)')
                            incar.write(' IOPT = 1\n')
                    for key, val in self.exp_params.items():
                        if key == 'ediffg' and val is None:
                            RuntimeError('Please set EDIFFG < 0')
        for key, val in self.list_params.items():
            if val is not None:
                if key in ('dipol', 'eint', 'ropt', 'rwigs', 'langevin_gamma'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.4f ' % x) for x in val]
                # ldau_luj is a dictionary that encodes all the
                # data. It is not a vasp keyword. An alternative to
                # the dictionary is to to use 'ldauu', 'ldauj',
                # 'ldaul', which are vasp keywords.
                elif (key in ('ldauu', 'ldauj') and
                      self.dict_params['ldau_luj'] is None):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.4f ' % x) for x in val]
                elif (key in ('ldaul') and
                      self.dict_params['ldau_luj'] is None):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%d ' % x) for x in val]
                elif key in ('ferwe', 'ferdo'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.1f ' % x) for x in val]
                elif key in ('iband', 'kpuse', 'random_seed'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%i ' % x) for x in val]
                elif key == 'magmom':
                    incar.write(' %s = ' % key.upper())
                    magmom_written = True
                    list = [[1, val[0]]]
                    for n in range(1, len(val)):
                        if val[n] == val[n - 1]:
                            list[-1][0] += 1
                        else:
                            list.append([1, val[n]])
                    [incar.write('%i*%.4f ' % (mom[0],
                                               mom[1]))
                     for mom in list]
                incar.write('\n')
        for key, val in self.bool_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if val:
                    incar.write('.TRUE.\n')
                else:
                    incar.write('.FALSE.\n')
        for key, val in self.special_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if key == 'lreal':
                    if isinstance(val, basestring):
                        incar.write(val + '\n')
                    elif isinstance(val, bool):
                        if val:
                            incar.write('.TRUE.\n')
                        else:
                            incar.write('.FALSE.\n')
        for key, val in self.dict_params.items():
            if val is not None:
                if key == 'ldau_luj':
                    llist = ulist = jlist = ''
                    for symbol in self.symbol_count:
                        #  default: No +U
                        luj = val.get(symbol[0], {'L': -1, 'U': 0.0, 'J': 0.0})
                        llist += ' %i' % luj['L']
                        ulist += ' %.3f' % luj['U']
                        jlist += ' %.3f' % luj['J']
                    incar.write(' LDAUL =%s\n' % llist)
                    incar.write(' LDAUU =%s\n' % ulist)
                    incar.write(' LDAUJ =%s\n' % jlist)

        if self.spinpol and not magmom_written:
            if not self.int_params['ispin']:
                incar.write(' ispin = 2\n'.upper())
            # Write out initial magnetic moments
            magmom = atoms.get_initial_magnetic_moments()[self.sort]
            # unpack magmom array if three components specified
            if magmom.ndim > 1:
                magmom = [item for sublist in magmom for item in sublist]
            list = [[1, magmom[0]]]
            for n in range(1, len(magmom)):
                if magmom[n] == magmom[n - 1]:
                    list[-1][0] += 1
                else:
                    list.append([1, magmom[n]])
            incar.write(' magmom = '.upper())
            [incar.write('%i*%.4f ' % (mom[0], mom[1])) for mom in list]
            incar.write('\n')

        if self.vca is not None and not vca_written:
            vcalist=''
            for symbol in self.symbol_count:
                vcalist+=' %.3f'%(self.vca.get(symbol[0], 1.0))
            incar.write('VCA = %s\n'%(vcalist))

        #if self.vca is not None and not vca_written:
        #    print(self.vca)
        #    vca=self.vca[self.sort]
        #    list = [[1, vca[0]]]
        #    for n in range(1, len(vca)):
        #        if vca[n] == vca[n - 1]:
        #            list[-1][0] += 1
        #        else:
        #            list.append([1, vca[n]])
        #    incar.write(' vca= '.upper())
        #    [incar.write('%i*%.4f ' % (v[0], v[1])) for v in list]
        #    incar.write('\n')

        incar.close()

    def restart_load(self):
        """Method which is called upon restart."""
        import ase.io
        # Try to read sorting file
        if os.path.isfile('ase-sort.dat'):
            self.sort = []
            self.resort = []
            file = open('ase-sort.dat', 'r')
            lines = file.readlines()
            file.close()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))
            atoms = ase.io.read('CONTCAR', format='vasp')[self.resort]
        else:
            atoms = ase.io.read('CONTCAR', format='vasp')
            self.sort = list(range(len(atoms)))
            self.resort = list(range(len(atoms)))
        self.atoms = atoms.copy()
        self.read_incar()
        self.read_outcar()
        self.set_results(atoms)
        try:
            self.read_kpoints()
        except NotImplementedError:
            print(
                "*****\n  WARING :Only Monkhorst-Pack and gamma centered grid supported for restart. Thus set Kpoints as gamma only. "
            )
            self.set(kpts=[1, 1, 1])
        self.read_potcar()
        #        self.old_incar_params = self.incar_params.copy()
        self.old_input_params = self.input_params.copy()
        self.converged = self.read_convergence()


def read_OUTCAR_energy(outcar='./OUTCAR'):
    """
    read energy with out entropy from OUTCAR
    """
    [energy_free, energy_zero] = [0, 0]
    if all:
        energy_free = []
        energy_zero = []
    for line in open(outcar, 'r'):
        # Free energy
        if line.lower().startswith('  free  energy   toten'):
            if all:
                energy_free.append(float(line.split()[-2]))
            else:
                energy_free = float(line.split()[-2])
                # Extrapolated zero point energy
        if line.startswith('  energy  without entropy'):
            if all:
                energy_zero.append(float(line.split()[-1]))
            else:
                energy_zero = float(line.split()[-1])
    return energy_free


def read_OUTCAR_charges(poscar=None,
                        outcar='./OUTCAR',
                        orbital=None,
                        symnum=None):
    poscar = os.path.join(os.path.split(outcar)[0], 'POSCAR')
    atoms = read(poscar)
    sdict = symbol_number(atoms)
    charges = np.zeros(len(atoms))
    orb_charges = []
    n = 0
    lines = open(outcar, 'r').readlines()
    for line in lines:
        if line.rfind('total charge   ') > -1:
            for m in range(len(atoms)):
                charges[m] = float(lines[n + m + 4].strip().split()[-1])
                orb_charges.append(
                    [float(x) for x in lines[n + m + 4].strip().split()[1:-1]])
        n += 1
    if orbital is None and symnum is None:
        return np.array(charges)
    elif orbital is None and symnum is not None:
        return np.array(charges)[sdict[symnum]]
    elif orbital is not None and symnum is not None:
        return np.array(orb_charges)[sdict[symnum]][orbital]
    else:
        return np.array(orb_charges)[:, orbital]


def read_magnetic_moments(poscar=None,
                          outcar='./OUTCAR',
                          orbital=None,
                          symnum=None):
    poscar = os.path.join(os.path.split(outcar)[0], 'POSCAR')
    atoms = read(poscar)
    sdict = symbol_number(atoms)
    magmoms = np.zeros(len(atoms))
    orb_magmoms = []
    n = 0
    lines = open(outcar, 'r').readlines()
    for line in lines:
        if line.rfind('magnetization (x)') > -1:
            for m in range(len(atoms)):
                magmoms[m] = float(lines[n + m + 4].strip().split()[-1])
                orb_magmoms.append(
                    [float(x) for x in lines[n + m + 4].strip().split()[1:-1]])
        n += 1
    if orbital is None and symnum is None:
        return np.array(magmoms)
    elif orbital is None and symnum is not None:
        return np.array(magmoms)[sdict[symnum]]
    elif orbital is not None and symnum is not None:
        return np.array(orb_magmoms)[sdict[symnum]][orbital]
    else:
        return np.array(orb_magmoms)[:, orbital]
