"""This module defines an ASE interface to ABINIT.
http://www.abinit.org/
"""

import os
from glob import glob
from os.path import join, isfile, islink
import shutil

from ase.atoms import Atoms
import numpy as np

from ase.data import atomic_numbers
from ase.units import Bohr, Hartree, fs, Ha, eV
from ase.data import chemical_symbols
from ase.io.abinit import read_abinit
from ase.io.vasp import write_vasp
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
    ReadError,all_changes,Calculator
import subprocess
from pyDFTutils.ase_utils.kpoints import get_ir_kpts, cubic_kpath
from string import Template

keys_with_units = {
    'toldfe': 'eV',
    'tsmear': 'eV',
    'paoenergyshift': 'eV',
    'zmunitslength': 'Bohr',
    'zmunitsangle': 'rad',
    'zmforcetollength': 'eV/Ang',
    'zmforcetolangle': 'eV/rad',
    'zmmaxdispllength': 'Ang',
    'zmmaxdisplangle': 'rad',
    'ecut': 'eV',
    'pawecutdg': 'eV',
    'dmenergytolerance': 'eV',
    'electronictemperature': 'eV',
    'oneta': 'eV',
    'onetaalpha': 'eV',
    'onetabeta': 'eV',
    'onrclwf': 'Ang',
    'onchemicalpotentialrc': 'Ang',
    'onchemicalpotentialtemperature': 'eV',
    'mdmaxcgdispl': 'Ang',
    'mdmaxforcetol': 'eV/Ang',
    'mdmaxstresstol': 'eV/Ang**3',
    'mdlengthtimestep': 'fs',
    'mdinitialtemperature': 'eV',
    'mdtargettemperature': 'eV',
    'mdtargetpressure': 'eV/Ang**3',
    'mdnosemass': 'eV*fs**2',
    'mdparrinellorahmanmass': 'eV*fs**2',
    'mdtaurelax': 'fs',
    'mdbulkmodulus': 'eV/Ang**3',
    'mdfcdispl': 'Ang',
    'warningminimumatomicdistance': 'Ang',
    'rcspatial': 'Ang',
    'kgridcutoff': 'Ang',
    'latticeconstant': 'Ang'
}


class Abinit(FileIOCalculator):
    """Class for doing ABINIT calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(label='abinit', xc='LDA', ecut=400, toldfe=1e-5)
    """

    implemented_properties = ['energy', 'forces', 'stress', 'magmom']
    #command = 'mpirun -np 6 abinit < PREFIX.files |tee PREFIX.log'
    if 'ASE_ABINIT_SCRIPT' in os.environ:
        command = os.environ['ASE_ABINIT_SCRIPT']
    else:
        print("Please set environment variable $ASE_ABINIT_SCRIPT")
        raise ValueError('env $ASE_ABINIT_SCRIPT not set.')
    default_parameters = dict(
        xc='LDA',
        smearing=None,
        kpts=None,
        gamma=True,
        charge=0.0,
        raw=None,
        pps='fhi')

    def __init__(self,
                 restart=None,
                 ignore_bad_restart_file=False,
                 label='abinit',
                 atoms=None,
                 scratch=None,
                 pppaths=None,
                 **kwargs):
        """Construct ABINIT-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'abinit'.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=Abinit(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        self.scratch = scratch
        self.U_dict = {}
        self.species = None
        self.ppp_list = None
        self.pspdict = {}
        self.pppaths = pppaths
        self.ndtset = 0  #num of data sets
        self.atoms = atoms
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        self.commander = None

    def set_command(self, command=None, commander=None):
        """
        set command. will override command from $ASE_ABINIT_SCRIPT.
        Parameters:
        ==============
        command: a single command to be excuted.
        commander: A object which has commander.run()
        Only one of command and commander should be set.
        """
        if command is not None:
            self.command = command
        if commander is not None:
            self.command = None
            self.commander = commander

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore boundary conditions:
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        return system_changes

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def set_Hubbard_U(self, U_dict, type=1):
        """
        set Hubbard_U parameters.

        :param U_dict: A dict like this {'Fe':{'L':,2,'U':4,'J',0.3}, 'O':{'L':1,'U':1,'J':0.3} }. 'L' is orbital angular momentem. 0 1 2 3 -> s p d f. 'U' is the Hubbard_U parameter, Hubbard_J is the J parameter.
        :param type: type of LDA+U correction. If type==0, only U should be specified.

        Usage::

            calc.set_Hubbard_U({'Fe':{'L':,2,'U':4,'J',0.3}, 'O':{'L':1,'U':1,'J':0.3} },type=1)
        """
        self.U_type = type
        self.U_dict = U_dict
        self.set(usepawu=type)

    def add_dataset(self, **kwargs):
        self.ndtset += 1
        self.set(ndtset=self.ndtset)
        kw = {}
        for key in kwargs:
            keyn = "%s%s" % (key, self.ndtset)
            kw[keyn] = kwargs[key]
        self.set(**kw)

    def ldos_calculation(self, atoms, pdos=True):
        if os.path.exists('abinito_DEN'):
            if os.path.exists('abiniti_DEN'):
                os.remove('abiniti_DEN')
            os.symlink('./abinito_DEN', './abiniti_DEN')
        # pawprtdos=2 and prtdosm should be used together
        self.set(
            autoparal=0,
            iscf=-2,
            prtdos=2,
            pawprtdos=0,
            prtdosm=0,
            tolwfr=1e-18,
            toldfe=0)
        if pdos:
            self.set(
                autoparal=0,
                iscf=-2,
                prtdos=3,
                pawprtdos=2,
                prtdosm=2,
                tolwfr=1e-18,
                toldfe=0)
        self.calculate(atoms, properties=[])
        if not os.path.exists('LDOS'):
            os.mkdir('LDOS')
        for fname in [
                'abinit.in', 'abinit.txt', 'abinit.log', 'abinit.ase',
                'abinito_DOS'
        ]:
            if os.path.exists(fname):
                shutil.copy(fname, 'LDOS')
        self.set(iscf=17, prtdos=0, prtdosm=0)

    def band_calculation(self, kpts):
        # TODO finish this
        self.set(iscf=-3)
        pass

    def relax_calculation(self, atoms, pre_relax=False, **kwargs):
        self.set(ionmov=2, ecutsm=0.5, dilatmx=1.1, optcell=2, ntime=50)
        if kwargs:
            self.set(**kwargs)
        self.calculate(atoms=atoms, properties=[])
        self.set(ntime=0, toldfe=0.0)
        newatoms = read_output(fname='abinit.txt', afterend=True)
        write_vasp('CONTCAR', newatoms, vasp5=True)
        print(newatoms)
        scaled_positions = newatoms.get_scaled_positions()
        cell = newatoms.get_cell()
        atoms.set_cell(cell)
        atoms.set_scaled_positions(scaled_positions)
        if not os.path.exists('RELAX'):
            os.mkdir('RELAX')
        for fname in [
                'abinit.in', 'abinit.txt', 'abinit.log', 'abinit.ase',
                'CONTCAR'
        ]:
            if os.path.exists(fname):
                shutil.copy(fname, 'RELAX')

        return atoms

    def scf_calculation(self, atoms, dos=True, **kwargs):
        self.set(ntime=0, toldfe=0.0)
        if dos:
            self.set(prtdos=2)
        self.calculate(atoms=atoms, )
        if not os.path.exists('SCF'):
            os.mkdir('SCF')
        for fname in ['abinit.in', 'abinit.txt', 'abinit.log', 'abinit.ase']:
            if os.path.exists(fname):
                shutil.copy(fname, 'SCF')
        if dos:
            if not os.path.exists('LDOS'):
                os.mkdir('LDOS')
        for fname in [
                'abinit.in', 'abinit.txt', 'abinit.log', 'abinit.ase',
                'abinito_DOS'
        ]:
            if os.path.exists(fname):
                shutil.copy(fname, 'LDOS')
        return atoms

    def efield_calculation(self,
                           atoms,
                           target_efield,
                           target_dfield,
                           nsteps,
                           skip_step=0,
                           workdir='./',
                           relax_params={''}):
        target_dfield = np.array(target_dfield)
        target_efield = np.array(target_efield)
        small_dfield = target_dfield * 1e-3
        small_efield = target_efield * 1e-3
        dfield = [small_dfield] + [
            i * 1.0 * target_dfield / nsteps for i in range(1, nsteps + 1)
        ]
        efield = [small_efield] + [
            i * 1.0 * target_efield / nsteps for i in range(1, nsteps + 1)
        ]
        print(dfield)

        workdir = os.path.abspath(workdir)
        wdir = [join(workdir, 'w%s' % i) for i in range(nsteps + 1)]

        cwd = os.getcwd()

        if skip_step == 0:
            if not os.path.exists(wdir[0]):
                os.makedirs(wdir[0])
            os.chdir(wdir[0])
            self.set(**relax_params)
            #self.set(toldff=5e-6, tolmxf=5e-5)
            self.relax_calculation(atoms)
            os.chdir(cwd)

        for i in range(skip_step, nsteps + 1):
            if not os.path.exists(wdir[i]):
                os.makedirs(wdir[i])
            os.chdir(wdir[i])

            self.set(ntime=0, berryopt=0,irdwfk=0)
            self.scf_calculation(atoms)

            src = os.path.join(wdir[i], 'abinito_WFK')
            des = os.path.join(wdir[i], 'abiniti_WFK')
            if os.path.exists(des):
                os.remove(des)
            os.symlink(src, des)

            self.scf_calculation(atoms, dos=False)
            self.set(
                ntime=0,
                irdwfk=1,
                berryopt=16,
                red_dfield=' '.join(map(str,dfield[i])),
                red_efield=' '.join(map(str,efield[i])))

            self.scf_calculation(atoms, dos=False)

            src = os.path.join(wdir[i], 'abinito_WFK')
            des = os.path.join(wdir[i], 'abiniti_WFK')
            if os.path.exists(des):
                os.remove(des)
            os.symlink(src, des)

            self.set(**relax_params)
            self.set(
                irdwfk=1,
                berryopt=16,
                red_dfield=' '.join(map(str,dfield[i])),
                red_efield=' '.join(map(str,efield[i])))

            atoms = self.relax_calculation(atoms)

            os.chdir(cwd)

        return atoms


    def wannier_calculation(self, atoms, wannier_input, **kwargs):
        nkpts = np.product(self.parameters['kpts'])
        self.set(
            ntime=0,
            toldfe=0.0,
            w90iniprj=2,
            prtwant=2,
            paral_kgb=0,
            kptopt=3,
            istwfk="%s*1" % nkpts)
        self.set(**kwargs)
        wannier_input.write_input()
        self.calculate(atoms=atoms, )
        if not os.path.exists('Wannier'):
            os.mkdir('Wannier')
        for fname in [
                'abinit.in', 'abinit.txt', 'abinit.log', 'abinit.ase',
                'abinito_DOS'
        ]:
            if os.path.exists(fname):
                shutil.copy(fname, 'Wannier')
        return atoms

    def calculate_phonon(self,
                         atoms,
                         ddk=True,
                         efield=True,
                         strain=True,
                         qpts=[2, 2, 2],
                         tolwfr=1e-23,
                         tolvrs=1e-12,
                         prtwf=0,
                         postproc=True,
                         ifcout=10240,
                         rfasr=1,
                         plot_band=True,
                         kpath=cubic_kpath()[0],
                         prtbbb=0,
                         split_dataset=True):
        self.ndtset = 0
        self.add_dataset(
            getwfk=0,
            kptopt=1,
            nqpt=0,
            rfphon=0,
            tolvrs=0,
            tolwfr=tolwfr,
            iscf=17)
        # d/dk
        # efield / strain Then dataset2: ddk, dataset3: efield/strain and Gamma phonon
        # else: dataset2: Gamma phonon
        if efield or strain:
            # DDK
            self.add_dataset(
                iscf=-3,
                getwfk=1,
                kptopt=2,
                rfelfd=2,
                nqpt=1,
                rfphon=0,
                tolvrs=0,
                tolwfr=tolwfr * 10,
                qpt='0 0 0')
            # Gamma, efield and/or strain and phonon
            self.add_dataset(
                getwfk=1,
                getddk=self.ndtset,
                kptopt=2,
                rfelfd=3 * int(efield),
                rfstrs=3 * strain,
                nqpt=1,
                iscf=7,
                qpt='0 0 0',
                rfphon=1,
                rfatpol="%s %s" % (1, len(atoms)),
                rfdir='1 1 1',
                tolvrs=tolvrs,
                toldfe=0,
                prtwf=prtwf,
                prtbbb=prtbbb)
        # phonon
        if isinstance(qpts[0], int):
            qpoints = get_ir_kpts(atoms, qpts)
        else:
            qpoints = qpts
        list2str = lambda l: " ".join(map(str, l))
        for q in qpoints:

            # phonon at Gamma
            if np.isclose(np.array(q), np.array([0, 0, 0])).all():
                if not (efield or strain):
                    self.add_dataset(
                        getwfk=1,
                        kptopt=2,
                        rfphon=1,
                        rfatpol="%s %s" % (1, len(atoms)),
                        rfdir="1 1 1",
                        toldfe=0,
                        tolvrs=tolvrs,
                        nqpt=1,
                        qpt=list2str(q),
                        iscf=7,
                        rfasr=rfasr,
                        prtwf=prtwf)
            # phonon
            else:
                self.add_dataset(
                    getwfk=1,
                    kptopt=3,
                    rfphon=1,
                    rfatpol="%s %s" % (1, len(atoms)),
                    rfdir="1 1 1",
                    toldfe=0,
                    tolvrs=tolvrs,
                    nqpt=1,
                    qpt=list2str(q),
                    iscf=7,
                    rfasr=rfasr,
                    prtwf=prtwf)
        if not split_dataset:
            self.calculate(atoms, properties=[])
        else:
            if True:  #efield or strain:
                # 1. ground state should be first.
                # 2. ddk & efield/strain & gamma
                # 3. qpoints.
                # 2 and 3 can be parallelized. Will be implemented later.
                for j in range(self.ndtset):
                    jdtset = j + 1
                    self.set(ndtset=1, jdtset=jdtset)
                    self.calculate(atoms, properties=[])
                    os.system('cp abinit.in abinit_DS%s.in' % jdtset)
                    os.system('cp abinit.txt abinit_DS%s.txt' % jdtset)
                    os.system('cp abinit.log abinit_DS%s.log' % jdtset)
            else:
                raise Exception("Shouldn't be here.")

        if postproc:
            if efield or strain:
                text = gen_mrgddb_input(self.prefix,
                                        list(range(3, self.ndtset + 1)))
            else:
                text = gen_mrgddb_input(self.prefix,
                                        list(range(2, self.ndtset + 1)))
            with open('%s.mrgddb.in' % self.prefix, 'w') as myfile:
                myfile.write(text)
            os.system("mrgddb <%s.mrgddb.in |tee %s.mrgddb.log" %
                      (self.prefix, self.prefix))

            with open('%s_ifc.files' % self.prefix, 'w') as myfile:
                myfile.write(gen_ifc_files(prefix=self.prefix))
            if efield:
                dipdip = 1
            else:
                dipdip = 0
            with open('%s_ifc.in' % self.prefix, 'w') as myfile:
                myfile.write(
                    gen_ifc_in(
                        ifcflag=1,
                        brav=1,
                        qpts=qpts,
                        nqshft=1,
                        ifcana=1,
                        ifcout=ifcout,
                        natifc=len(atoms),
                        atifc=" ".join(
                            map(str, list(range(1, len(atoms) + 1)))),
                        chneut=1,
                        dipdip=dipdip,
                        kpath=kpath))
            os.system("anaddb < %s_ifc.files" % self.prefix)

    def postproc_phonon(self,
                        atoms,
                        prefix='abinit',
                        ndtset=None,
                        efield=True,
                        qpts=[2, 2, 2],
                        tolwfr=1e-23,
                        tolvrs=1e-9,
                        postproc=True,
                        ifcout=1024,
                        rfasr=1):
        if True:
            ndtset = self.ndtset
            if efield:
                text = gen_mrgddb_input(prefix, list(range(3, ndtset + 1)))
            else:
                text = gen_mrgddb_input(prefix, list(range(2, ndtset + 1)))
            with open('%s.mrgddb.in' % prefix, 'w') as myfile:
                myfile.write(text)
            os.system("mrgddb <%s.mrgddb.in |tee %s.mrgddb.log" % (prefix,
                                                                   prefix))

            with open('%s_ifc.files' % prefix, 'w') as myfile:
                myfile.write(gen_ifc_files(prefix=prefix))
            if efield:
                dipdip = 1
            else:
                dipdip = 0
            with open('%s_ifc.in' % prefix, 'w') as myfile:
                myfile.write(
                    gen_ifc_in(
                        ifcflag=1,
                        brav=1,
                        qpts=qpts,
                        nqshft=1,
                        ifcana=1,
                        ifcout=ifcout,
                        natifc=len(atoms),
                        atifc=" ".join(
                            map(str, list(range(1, len(atoms) + 1)))),
                        chneut=1,
                        dipdip=dipdip))

            os.system("anaddb < %s_ifc.files" % prefix)

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input parameters to files-file."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        if ('numbers' in system_changes
                or 'initial_magmoms' in system_changes):
            self.initialize(atoms)

        fh = open(self.label + '.files', 'w')

        fh.write('%s\n' % (self.prefix + '.in'))  # input
        fh.write('%s\n' % (self.prefix + '.txt'))  # output
        fh.write('%s\n' % (self.prefix + 'i'))  # input
        fh.write('%s\n' % (self.prefix + 'o'))  # output

        # XXX:
        # scratch files
        #scratch = self.scratch
        #if scratch is None:
        #    scratch = dir
        #if not os.path.exists(scratch):
        #    os.makedirs(scratch)
        #fh.write('%s\n' % (os.path.join(scratch, prefix + '.abinit')))
        fh.write('%s\n' % (self.prefix + '.abinit'))
        # Provide the psp files
        for ppp in self.ppp_list:
            if not os.path.exists('psp'):
                os.makedirs('psp')
            newppp = os.path.join('psp', os.path.split(ppp)[-1])
            shutil.copyfile(ppp, newppp)
            fh.write('%s\n' % (newppp))  # psp file path
        fh.close()

        # write psp info
        with open('pspinfo.txt', 'w') as myfile:
            myfile.write(str(self.pspdict))

        # Abinit will write to label.txtA if label.txt already exists,
        # so we remove it if it's there:
        filename = self.label + '.txt'
        if os.path.isfile(filename):
            os.remove(filename)

        param = self.parameters
        param.write(self.label + '.ase')

        fh = open(self.label + '.in', 'w')
        inp = {}
        inp.update(param)
        for key in ['xc', 'smearing', 'kpts', 'pps', 'raw', 'gamma']:
            del inp[key]

        smearing = param.get('smearing')
        if 'tsmear' in param or 'occopt' in param:
            assert smearing is None

        if smearing is not None:
            inp['occopt'] = {
                'fermi-dirac': 3,
                'gaussian': 7
            }[smearing[0].lower()]
            inp['tsmear'] = smearing[1]

        inp['natom'] = len(atoms)

        if 'nbands' in param:
            inp['nband'] = param.nbands
            del inp['nbands']

        if 'ixc' not in param:
            inp['ixc'] = {
                'LDA': 1,
                'PBE': 11,
                'revPBE': 14,
                'RPBE': 15,
                'WC': 23,
                'PBEsol': -116133
            }[param.xc]

        magmoms = atoms.get_initial_magnetic_moments()
        if magmoms.any():
            #inp['nsppol'] = 2
            fh.write('spinat\n')
            for n, M in enumerate(magmoms):
                fh.write('%.14f %.14f %.14f\n' % (0, 0, M))
        else:
            inp['nsppol'] = 1

        #### Mod by Hexu , add Hubbard U########################
        if self.U_dict != {}:
            syms = atoms.get_chemical_symbols()
            elems = []
            for s in syms:
                if not s in elems:
                    elems.append(s)
            for s in elems:
                if s not in self.U_dict:
                    self.U_dict[s] = {'L': -1, 'U': 0, 'J': 0}
            if self.U_dict != {}:
                fh.write('# DFT+U\n')
                fh.write('lpawu %s\n' %
                         (' '.join([str(self.U_dict[s]['L']) for s in elems])))
                fh.write('upawu %s eV\n' %
                         (' '.join([str(self.U_dict[s]['U']) for s in elems])))
                fh.write('jpawu %s eV\n' %
                         (' '.join([str(self.U_dict[s]['J']) for s in elems])))

        for key in sorted(inp.keys()):
            value = inp[key]
            unit = keys_with_units.get(key)
            if unit is None:
                fh.write('%s %s\n' % (key, value))
            else:
                if 'fs**2' in unit:
                    value /= fs**2
                elif 'fs' in unit:
                    value /= fs
                fh.write('%s %e %s\n' % (key, value, unit))

        if param.raw is not None:
            for line in param.raw:
                if isinstance(line, tuple):
                    fh.write(' '.join(['%s' % x for x in line]) + '\n')
                else:
                    fh.write('%s\n' % line)

        fh.write('#Definition of the unit cell\n')
        fh.write('acell\n')
        fh.write('%.14f %.14f %.14f Angstrom\n' % (1.0, 1.0, 1.0))
        fh.write('rprim\n')
        for v in atoms.cell:
            fh.write('%.14f %.14f %.14f\n' % tuple(v))

        fh.write('chkprim 0 # Allow non-primitive cells\n')

        fh.write('#Definition of the atom types\n')
        fh.write('ntypat %d\n' % (len(self.species)))
        fh.write('znucl')
        for n, Z in enumerate(self.species):
            fh.write(' %d' % (Z))
        fh.write('\n')
        fh.write('#Enumerate different atomic species\n')
        fh.write('typat')
        fh.write('\n')
        self.types = []
        for Z in atoms.numbers:
            for n, Zs in enumerate(self.species):
                if Z == Zs:
                    self.types.append(n + 1)
        n_entries_int = 20  # integer entries per line
        for n, type in enumerate(self.types):
            fh.write(' %d' % (type))
            if n > 1 and ((n % n_entries_int) == 1):
                fh.write('\n')
        fh.write('\n')

        fh.write('#Definition of the atoms\n')
        fh.write('xangst\n')
        for pos in atoms.positions:
            fh.write('%.14f %.14f %.14f\n' % tuple(pos))
        gamma = self.parameters.gamma
        if 'kptopt' not in param:
            mp = kpts2mp(atoms, param.kpts)
            fh.write('kptopt 1\n')
            fh.write('ngkpt %d %d %d\n' % tuple(mp))
            fh.write('nshiftk 1\n')
            fh.write('shiftk\n')
            if gamma:
                fh.write('%.1f %.1f %.1f\n' % tuple(
                    (np.array(mp) + 1) % 2 * 0.0))
            else:
                fh.write('%.1f %.1f %.1f\n' % tuple(
                    (np.array(mp) + 1) % 2 * 0.5))
        #----Modified by hexu-----#
        elif param['kptopt'] in [1, 2, 3, 4]:
            mp = kpts2mp(atoms, param.kpts)
            #fh.write('kptopt %s\n'%(param['kptopt']))
            fh.write('ngkpt %d %d %d\n' % tuple(mp))
            fh.write('nshiftk 1\n')
            fh.write('shiftk\n')
            if gamma:
                fh.write('%.1f %.1f %.1f\n' % tuple(
                    (np.array(mp) + 1) % 2 * 0.0))
            else:
                fh.write('%.1f %.1f %.1f\n' % tuple(
                    (np.array(mp) + 1) % 2 * 0.5))

        fh.write(
            'chkexit 1 # abinit.exit file in the running directory terminates after the current SCF\n'
        )

        fh.close()

    def calculate(self,
                  atoms,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None and self.commander is None:
            raise RuntimeError('Please set $%s environment variable ' % (
                'ASE_' + self.name.upper() + '_COMMAND'
            ) + 'or supply the command keyword or supply a commander with run() method (calc.set_command(...))'
                               )
        if self.commander is None:
            command = self.command.replace('PREFIX', self.prefix)
        olddir = os.getcwd()
        try:
            os.chdir(self.directory)
            if self.commander is None:
                errorcode = subprocess.call(command, shell=True)
            else:
                errorcode = self.commander.run()
            #os.system("bash %s"%command)
        finally:
            os.chdir(olddir)

        if errorcode:
            #raise RuntimeError('%s returned an error: %d' %
            #                   (self.name, errorcode))
            print(('%s returned an error: %d' % (self.name, errorcode)))

        self.read_results()

    def read(self, label):
        """Read results from ABINIT's text-output file."""
        FileIOCalculator.read(self, label)
        filename = self.label + '.txt'
        if not os.path.isfile(filename):
            raise ReadError

        self.atoms = read_abinit(self.label + '.in')
        self.parameters = Parameters.read(self.label + '.ase')

        self.initialize(self.atoms)
        self.read_results()

    def read_results(self):
        filename = self.label + '.txt'
        text = open(filename).read().lower()

        if ('error' in text
                or 'was not enough scf cycles to converge' in text):
            #raise ReadError("not enoubh scf cycles to converge")
            print("not enoubh scf cycles to converge")

        for line in iter(text.split('\n')):
            if line.rfind('natom  ') > -1:
                natoms = int(line.split()[-1])

        lines = iter(text.split('\n'))
        # Stress:
        # Printed in the output in the following format [Hartree/Bohr^3]:
        # sigma(1 1)=  4.02063464E-04  sigma(3 2)=  0.00000000E+00
        # sigma(2 2)=  4.02063464E-04  sigma(3 1)=  0.00000000E+00
        # sigma(3 3)=  4.02063464E-04  sigma(2 1)=  0.00000000E+00
        for line in lines:
            if line.rfind(
                    'cartesian components of stress tensor (hartree/bohr^3)'
            ) > -1:
                stress = np.empty(6)
                for i in range(3):
                    entries = lines.next().split()
                    stress[i] = float(entries[2])
                    stress[i + 3] = float(entries[5])
                self.results['stress'] = stress * Hartree / Bohr**3
                break
        else:
            #raise RuntimeError('Stress not found')
            print("Stress not found")

        # Energy [Hartree]:
        # Warning: Etotal could mean both electronic energy and free energy!
        etotal = None
        efree = None
        if 'PAW method is used'.lower(
        ) in text:  # read DC energy according to M. Torrent
            for line in iter(text.split('\n')):
                if line.rfind('>>>>> internal e=') > -1:
                    etotal = float(
                        line.split('=')[-1]) * Hartree  # second occurence!
            for line in iter(text.split('\n')):
                if line.rfind('>>>> etotal (dc)=') > -1:
                    efree = float(line.split('=')[-1]) * Hartree
        else:
            for line in iter(text.split('\n')):
                if line.rfind('>>>>> internal e=') > -1:
                    etotal = float(
                        line.split('=')[-1]) * Hartree  # first occurence!
                    break
            for line in iter(text.split('\n')):
                if line.rfind('>>>>>>>>> etotal=') > -1:
                    efree = float(line.split('=')[-1]) * Hartree
        if efree is None:
            print('Total energy not found')
            efree = 0
        #raise RuntimeError('Total energy not found')
        if etotal is None:
            etotal = efree

        # Energy extrapolated to zero Kelvin:
        self.results['energy'] = (etotal + efree) / 2
        self.results['free_energy'] = efree

        # Forces:
        for line in lines:
            if line.lower().rfind(
                    'cartesian forces (ev/angstrom) at end:') > -1:
                forces = []
                for i in range(natoms):
                    forces.append(
                        np.array([float(f) for f in lines.next().split()[1:]]))
                self.results['forces'] = np.array(forces)
                break
        else:
            #raise RuntimeError('Forces not found')
            print("Forces not found!")
        #
        self.width = self.read_electronic_temperature()
        self.nband = self.read_number_of_bands()
        self.niter = self.read_number_of_iterations()
        self.nelect = self.read_number_of_electrons()
        self.results['magmom'] = self.read_magnetic_moment()

    def initialize(self, atoms):
        numbers = atoms.get_atomic_numbers().copy()
        self.species = []
        for a, Z in enumerate(numbers):
            if Z not in self.species:
                self.species.append(Z)

        self.spinpol = atoms.get_initial_magnetic_moments().any()

        if self.pppaths is not None:
            pppaths = self.pppaths
        elif 'ABINIT_PP_PATH' in os.environ:
            pppaths = os.environ['ABINIT_PP_PATH'].split(':')
        else:
            pppaths = []

        self.ppp_list = []
        if self.parameters.xc != 'LDA' and self.parameters.xc != 'PBEsol':
            xcname = 'PBE'
        elif self.parameters.xc == 'PBEsol':
            xcname = 'PBEsol'
        else:
            xcname = 'LDA'

        pps = self.parameters.pps
        self.pspdict['psptype'] = pps
        if pps not in [
                'fhi', 'hgh', 'hgh.sc', 'hgh.k', 'tm', 'paw', 'jth', 'gbrv',
                'eric-gen', 'ONCV', 'jth_sp', 'GPAW'
        ]:
            raise ValueError('Unexpected PP identifier %s' % pps)

        for Z in self.species:
            symbol = chemical_symbols[abs(Z)]
            number = atomic_numbers[symbol]

            if pps == 'fhi':
                name = '%02d-%s.%s.fhi' % (number, symbol, xcname)
            elif pps in ['paw']:
                hghtemplate = '%s-%s-%s.paw'  # E.g. "H-GGA-hard-uspp.paw"
                name = hghtemplate % (symbol, xcname, '*')
            elif pps in ['hgh.k']:
                hghtemplate = '%s-q%s.hgh.k'  # E.g. "Co-q17.hgh.k"
                name = hghtemplate % (symbol, '*')
            elif pps in ['tm']:
                hghtemplate = '%d%s%s.pspnc'  # E.g. "44ru.pspnc"
                name = hghtemplate % (number, symbol.lower(), '*')
            elif pps in ['hgh', 'hgh.sc']:
                hghtemplate = '%d%s.%s.hgh'  # E.g. "42mo.6.hgh"
                # There might be multiple files with different valence
                # electron counts, so we must choose between
                # the ordinary and the semicore versions for some elements.
                #
                # Therefore we first use glob to get all relevant files,
                # then pick the correct one afterwards.
                name = hghtemplate % (number, symbol.lower(), '*')
            elif pps in ['jth', 'jth_sp', 'jth_fincore']:
                xcdict = {'LDA': 'LDA', 'PBE': 'GGA_PBE'}
                #hghtemplate = '%s.%s-JTH.xml'
                #name = hghtemplate%(symbol, xcdict[xcname.upper()])
                print((os.path.join(pppaths[-1], '*/%s.%s*JTH*.xml' %
                                    (symbol, xcdict[xcname]))))
                if pps == 'jth':
                    name = glob(
                        os.path.join(pppaths[-1], '*/%s.%s*JTH.xml' % (
                            symbol, xcdict[xcname])))[0]
                elif pps == 'jth_sp' or pps == 'jth_fincore':
                    names = None
                    names = glob(
                        os.path.join(pppaths[-1], '*/%s.%s*JTH*.xml' % (
                            symbol, xcdict[xcname])))
                    name = None
                    for n in names:
                        if n.find('_sp') != -1 or n.find('_fincore') != -1:
                            name = n
                    if name is None:
                        name = names[-1]
                print(name)
            elif pps in ['gbrv']:
                xcdict = {'LDA': 'lda', 'PBE': 'pbe'}
                name = glob(
                    os.path.join(pppaths[-1], '*/%s_%s*' % (
                        symbol.lower(), xcname.lower())))[-1]
            elif pps in ['eric-gen']:
                xcdict = {'LDA': 'LDA', 'PBE': 'PBE', 'PBEsol': 'PBEsol'}
                print('*/%s_%s_%s*' % (number, symbol, xcdict[xcname]))
                name = glob(
                    os.path.join(pppaths[-1], '*/%02d_%s_%s*' % (
                        number, symbol, xcdict[xcname])))[-1]
            elif pps in ['ONCV']:
                xcdict = {'LDA': 'PW', 'PBE': 'PBE', 'PBEsol': 'PBEsol'}
                #print '*/%s_%s_%s*'%(number,symbol,xcdict[xcname])
                name = glob(
                    os.path.join(pppaths[-1], 'ONCVPSP-%s-PDv0.3/%s/%s*.psp8' %
                                 (xcdict[xcname], symbol, symbol)))[-1]
            elif pps in ['GPAW']:
                xcdict = {'LDA': 'LDA', 'PBE': 'PBE', 'PBEsol': 'PBEsol'}
                print('*/%s_%s_%s*' % (number, symbol, xcdict[xcname]))
                name = glob(
                    os.path.join(pppaths[-1], 'GPAW-%s/%s.%s.xml' % (xcdict[
                        xcname], symbol, xcdict[xcname])))[-1]

            found = False
            print(pps, name, pppaths)

            for path in pppaths:
                if (pps.startswith('paw') or pps.startswith('hgh')
                        or pps.startswith('tm')):
                    #pps.startswith('jth')):
                    filenames = glob(join(path, name))
                    print("fname", filenames)
                    if not filenames:
                        continue
                    assert len(filenames) in [0, 1, 2]
                    if pps == 'paw':
                        selector = max  # Semicore or hard
                        # warning: see download.sh in
                        # abinit-pseudopotentials*tar.gz for additional
                        # information!
                        S = selector([
                            str(os.path.split(name)[1].split('-')[2][:-4])
                            for name in filenames
                        ])
                        name = hghtemplate % (symbol, xcname, S)
                    elif pps == 'hgh':
                        selector = min  # Lowest valence electron count
                        Z = selector([
                            int(os.path.split(name)[1].split('.')[1])
                            for name in filenames
                        ])
                        name = hghtemplate % (number, symbol.lower(), str(Z))
                    elif pps == 'hgh.k':
                        selector = min  # Semicore - highest electron count
                        Z = selector([
                            int(os.path.split(name)[1].split('-')[1][:-6][1:])
                            for name in filenames
                        ])
                        name = hghtemplate % (symbol, Z)
                    elif pps == 'tm':
                        selector = max  # Semicore - highest electron count
                        # currently only one version of psp per atom
                        name = hghtemplate % (number, symbol.lower(), '')
                    elif pps == 'jth':
                        selector = max  # Semicore - highest electron count
                        # currently only one version of psp per atom
                        # name = hghtemplate % (symbol, xcname.upper())
                    elif pps == 'gbrv':
                        selector = max
                    elif pps == 'eric-gen':
                        selector = max
                    elif pps == 'ONCV':
                        selector = max
                    elif pps == 'GPAW':
                        selector = max
                    else:
                        assert pps == 'hgh.sc'
                        selector = max  # Semicore - highest electron count
                        Z = selector([
                            int(os.path.split(name)[1].split('.')[1])
                            for name in filenames
                        ])
                        name = hghtemplate % (number, symbol.lower(), str(Z))
                filename = join(path, name)
                print(filename)
                if isfile(filename) or islink(filename):
                    found = True
                    self.ppp_list.append(filename)
                    self.pspdict[symbol] = filename
                    break
            if not found:
                raise RuntimeError('No pseudopotential for %s!' % symbol)

    def get_number_of_iterations(self):
        return self.niter

    def read_number_of_iterations(self):
        niter = None
        for line in open(self.label + '.txt'):
            if line.find(
                    ' At SCF step') != -1:  # find the last iteration number
                niter = int(line.split(',')[0].split()[3].strip())
        return niter

    def get_electronic_temperature(self):
        return self.width * Hartree

    def read_electronic_temperature(self):
        width = None
        # only in log file!
        for line in open(self.label + '.log'):  # find last one
            if line.find('tsmear') != -1:
                width = float(line.split()[1].strip())
        return width

    def get_number_of_electrons(self):
        return self.nelect

    def read_number_of_electrons(self):
        nelect = None
        # only in log file!
        for line in open(self.label + '.log'):  # find last one
            if line.find('with nelect') != -1:
                nelect = float(line.split('=')[1].strip())
        return nelect

    def get_number_of_bands(self):
        return self.nband

    def read_number_of_bands(self):
        nband = None
        for line in open(self.label + '.txt'):  # find last one
            if line.find('     nband') != -1:  # nband, or nband1, nband*
                nband = int(line.split()[-1].strip())
        return nband

    def get_kpts_info(self, kpt=0, spin=0, mode='eigenvalues'):
        return self.read_kpts_info(kpt, spin, mode)

    def get_k_point_weights(self):
        return self.get_kpts_info(kpt=0, spin=0, mode='k_point_weights')

    def get_bz_k_points(self):
        raise NotImplementedError

    def get_ibz_k_points(self):
        return self.get_kpts_info(kpt=0, spin=0, mode='ibz_k_points')

    def get_spin_polarized(self):
        return self.spinpol

    def get_number_of_spins(self):
        return 1 + int(self.spinpol)

    def read_magnetic_moment(self):
        magmom = None
        if not self.get_spin_polarized():
            magmom = 0.0
        else:  # only for spinpolarized system Magnetisation is printed
            for line in open(self.label + '.txt'):
                if line.find('Magnetisation') != -1:  # last one
                    magmom = float(line.split('=')[-1].strip())
        return magmom

    def get_fermi_level(self):
        return self.read_fermi()

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.get_kpts_info(kpt, spin, 'eigenvalues')

    def get_occupations(self, kpt=0, spin=0):
        return self.get_kpts_info(kpt, spin, 'occupations')

    def read_fermi(self):
        """Method that reads Fermi energy in Hartree from the output file
        and returns it in eV"""
        E_f = None
        filename = self.label + '.txt'
        text = open(filename).read().lower()
        assert 'error' not in text
        for line in iter(text.split('\n')):
            if line.rfind('fermi (or homo) energy (hartree) =') > -1:
                E_f = float(line.split('=')[1].strip().split()[0])
        return E_f * Hartree

    def read_nkpt(self):
        nkpt = None
        fname = self.label + '.txt'
        with open(fname) as myfile:
            for line in myfile:
                result = re.findall('nkpt\s*=\s*(\d*)', line)
                if len(result) != 0:
                    nkpt = int(result[0])
                    return nkpt
        #if nkpt is None:
        #    raise ValueError("Warning: nkpt not found!")
        return nkpt

    def read_nband(self):
        nband = None
        fname = self.label + '.txt'
        with open(fname) as myfile:
            for line in myfile:
                result = re.findall('mband\s*=\s*(\d*)', line)
                if len(result) != 0:
                    nband = int(result[0])
                    return nband
        return nband

    def read_kpts_info(self, kpt=0, spin=0, mode='eigenvalues'):
        """ Returns list of last eigenvalues, occupations, kpts weights, or
        kpts coordinates for given kpt and spin.
        Due to the way of reading output the spins are exchanged in spin-polarized case.  """
        # output may look like this (or without occupation entries); 8 entries per line:
        #
        #  Eigenvalues (hartree) for nkpt=  20  k points:
        # kpt#   1, nband=  3, wtk=  0.01563, kpt=  0.0625  0.0625  0.0625 (reduced coord)
        #  -0.09911   0.15393   0.15393
        #      occupation numbers for kpt#   1
        #   2.00000   0.00000   0.00000
        # kpt#   2, nband=  3, wtk=  0.04688, kpt=  0.1875  0.0625  0.0625 (reduced coord)
        # ...
        #
        assert mode in [
            'eigenvalues', 'occupations', 'ibz_k_points', 'k_point_weights'
        ], mode
        if self.get_spin_polarized():
            spin = {0: 1, 1: 0}[spin]
        if spin == 0:
            spinname = ''
        else:
            spinname = 'SPIN UP'.lower()
        # number of lines of eigenvalues/occupations for a kpt
        nband = self.get_number_of_bands()
        n_entries_float = 8  # float entries per line
        n_entry_lines = max(1, int((nband - 0.1) / n_entries_float) + 1)

        filename = self.label + '.txt'
        text = open(filename).read().lower()
        assert 'error' not in text
        lines = text.split('\n')
        text_list = []
        # find the begining line of last eigenvalues
        contains_eigenvalues = 0
        for n, line in enumerate(lines):
            if spin == 0:
                if line.rfind('eigenvalues (hartree) for nkpt') > -1:
                    #if line.rfind('eigenvalues (   ev  ) for nkpt') > -1: #MDTMP
                    contains_eigenvalues = n
            else:
                if (line.rfind('eigenvalues (hartree) for nkpt') > -1 and
                        line.rfind(spinname) > -1):  # find the last 'SPIN UP'
                    contains_eigenvalues = n
        # find the end line of eigenvalues starting from contains_eigenvalues
        text_list = [lines[contains_eigenvalues]]
        for line in lines[contains_eigenvalues + 1:]:
            text_list.append(line)
            # find a blank line or eigenvalues of second spin
            if (not line.strip()
                    or line.rfind('eigenvalues (hartree) for nkpt') > -1):
                break
        # remove last (blank) line
        text_list = text_list[:-1]

        assert contains_eigenvalues, 'No eigenvalues found in the output'

        n_kpts = int(text_list[0].split('nkpt=')[1].strip().split()[0])

        # get rid of the "eigenvalues line"
        text_list = text_list[1:]

        # join text eigenvalues description with eigenvalues
        # or occupation numbers for kpt# with occupations
        contains_occupations = False
        for line in text_list:
            if line.rfind('occupation numbers') > -1:
                contains_occupations = True
                break
        if mode == 'occupations':
            assert contains_occupations, 'No occupations found in the output'

        if contains_occupations:
            range_kpts = 2 * n_kpts
        else:
            range_kpts = n_kpts

        values_list = []
        offset = 0
        for kpt_entry in range(range_kpts):
            full_line = ''
            for entry_line in range(n_entry_lines + 1):
                full_line = full_line + str(text_list[offset + entry_line])
            first_line = text_list[offset]
            if mode == 'occupations':
                if first_line.rfind('occupation numbers') > -1:
                    # extract numbers
                    full_line = [
                        float(v)
                        for v in full_line.split('#')[1].strip().split()[1:]
                    ]
                    values_list.append(full_line)
                    full_line = [
                        Hartree * float(v)
                        for v in full_line.split(')')[1].strip().split()[:]
                    ]
                    #full_line = [float(v) for v in full_line.split(')')[1].strip().split()[:]] #MDTMP
                elif mode == 'ibz_k_points':
                    full_line = [
                        float(v)
                        for v in full_line.split('kpt=')[1].strip().split('(')[
                            0].split()
                    ]
                else:
                    full_line = float(
                        full_line.split('wtk=')[1].strip().split(',')[0]
                        .split()[0])
                    values_list.append(full_line)
            offset = offset + n_entry_lines + 1

        if mode in ['occupations', 'eigenvalues']:
            return np.array(values_list[kpt])
        else:
            return np.array(values_list)


def gen_mrgddb_input(prefix, idatasets):
    """
    prefix_mrgddb.out
    Generated by mrgddb.
    number of datasets
    prefixo_DS[i=3,4,...]_DDB
    """
    text = ''
    text += '%s_mrgddb.out\n' % prefix
    text += 'Unnamed\n'
    text += '%s\n' % len(idatasets)
    for i in idatasets:
        text += '%so_DS%s_DDB\n' % (prefix, i)
    return text


def gen_ifc_files(prefix='abinit'):
    text = """${prefix}_ifc.in
${prefix}_ifc.out
${prefix}_mrgddb.out
${prefix}_band2eps
${prefix}_dummy1
${prefix}_dummy2
${prefix}_dummy3
"""
    template = Template(text)
    t = template.substitute({"prefix": prefix})
    return t


def gen_ifc_in(**kwargs):
    print(kwargs)
    if 'qpts' in kwargs:
        kwargs['qpts'] = ' '.join(map(str, kwargs['qpts']))

    text = """
!Input file for anaddb to generate ifc.
    alphon 1
    eivec=2
    ifcflag $ifcflag
    brav $brav
    ngqpt $qpts
    nqshft $nqshft
    q1shft 3*0.0

    ifcana $ifcana
    ifcout $ifcout
    natifc $natifc
    atifc $atifc

    chneut $chneut

    dipdip $dipdip\n"""
    template = Template(text)
    t = template.substitute(kwargs)
    if 'kpath' in kwargs:
        kpath = kwargs['kpath']
        nk = len(kpath)
        t += "!====Phonon band kpoints===\n"
        t += "    nph1l %s\n" % nk
        t += " qph1l %s 1.0\n" % (' '.join([str(x) for x in kpath[0]]))
        for kpt in kpath[1:]:
            t += "       %s 1.0\n" % (' '.join([str(x) for x in kpt]))
    return t


def gen_ifc(prefix='abinit'):
    with open('%s_ifc.files' % prefix, 'w') as myfile:
        myfile.write(gen_ifc_files(prefix=prefix))

    with open('%s_ifc.in' % prefix, 'w') as myfile:
        myfile.write(
            gen_ifc_in(
                ifcflag=1,
                brav=1,
                nk=2,
                qpts=[2, 2, 2],
                nqshft=1,
                ifcana=1,
                ifcout=20,
                natifc=5,
                atifc="1 2 3 4 5",
                chneut=1,
                dipdip=1,
                kpath=cubic_kpath()[0]))


    #os.system("anaddb < %s_ifc.files"%prefix)
def read_nband(fname='abinit.txt'):
    inside_outvar = False
    with open(fname) as myfile:
        for line in myfile:
            if line.strip().find('END DATASET') != -1:
                inside_outvar = True
            if inside_outvar:
                #read nband
                if line.strip().startswith('nband'):
                    nband = int(line.strip().split()[-1])
                    return nband
    return None


def read_output(fname='abinit.txt', afterend=True):
    inside_outvar = False
    cell = []
    with open(fname) as myfile:
        for line in myfile:
            if line.strip().find('END DATASET') != -1:
                inside_outvar = True
            if inside_outvar or afterend == False:
                # read acell
                if line.strip().startswith('acell'):
                    #print line
                    t = line.strip().split()[1:4]
                    acell = list(map(float, t))
                    acell = np.asarray(acell) * Bohr
                    #print "acell",acell
                #read rprim:
                if line.strip().startswith('rprim'):
                    cell.append([
                        float(x) * acell[0] for x in line.strip().split()[1:]
                    ])
                    for i in range(1, 3):
                        line = next(myfile)
                        cell.append([
                            float(x) * acell[i] for x in line.strip().split()
                        ])
                    #print cell

                    #read natom
                if line.strip().startswith('natom'):
                    natom = int(line.strip().split()[-1])
                    #print natom
                #read ntypat
                if line.strip().startswith('ntypat'):
                    ntypat = int(line.strip().split()[-1])
                    print(ntypat)

                if line.strip().startswith('typat'):
                    t = line.strip().split()[1:]
                    typat = list(map(int, t))
                    #print typat

                #read positions:
                if line.strip().startswith('xred'):
                    poses = []
                    poses.append([float(x) for x in line.strip().split()[1:]])
                    for i in range(1, natom):
                        line = next(myfile)
                        poses.append([float(x) for x in line.strip().split()])
                    #print poses

                    #read znucl
                if line.strip().startswith('znucl'):
                    t = line.strip().split()[1:4]
                    znucl = list(map(int, list(map(float, t))))
                    #print znucl
    if cell == []:
        cell = np.diag(acell)
    numbers = [znucl[i - 1] for i in typat]
    atoms = Atoms(numbers=numbers, scaled_positions=poses, cell=cell, pbc=True)
    return atoms


def default_abinit_calculator(ecut=35 * Ha,
                              xc='LDA',
                              nk=8,
                              mag_order='PM',
                              is_metal=False,
                              pps='ONCV',
                              **kwargs):
    """
    default abinit calculator.

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
        ecut=ecut * eV,  # warning - used to speedup the test
        kpts=[nk, nk, nk],  # warning - used to speedup the test
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
                        print(line)
                        spos[i] = [
                            float(s.replace('D', 'E'))
                            for s in line.strip().split()[-3:]
                        ]
            numbers = [znucl[i - 1] for i in typat]
            self.symbols = [chemical_symbols[i] for i in numbers]
            self.masses = [atomic_masses[i] for i in numbers]
            self.cell = [rprim0, rprim1, rprim2]
            print(self.symbols)
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
                               idir2] = self.dynamic_matrix_dict[(idir1 + 1,
                                                                  ipert1 + 1,
                                                                  idir2 + 1,
                                                                  ipert2 + 1)]
        return dynmat


#myreader = DDB_reader("../test/BaTiO3_bak/abinito_DS2_DDB")
#print myreader.read_atoms().get_positions()
#myreader.read_2DE_DDB()
#print myreader.get_dynamic_matrix()
