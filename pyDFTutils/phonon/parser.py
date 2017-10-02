#!/usr/bin/env python
import os
import numpy as np
from ase.data import chemical_symbols
import matplotlib.pyplot as plt
from abipy.abilab import abiopen
from pyDFTutils.perovskite.perovskite_mode import label_zone_boundary, label_Gamma
from ase.units import Ha
from spglib import spglib


def displacement_cart_to_evec(displ_cart,
                              masses,
                              scaled_positions,
                              qpoint=None,
                              add_phase=True):
    """
    displ_cart: cartisien displacement. (atom1_x, atom1_y, atom1_z, atom2_x, ...)
    masses: masses of atoms.
    scaled_postions: scaled postions of atoms.
    qpoint: if phase needs to be added, qpoint must be given.
    add_phase: whether to add phase to the eigenvectors.
    """
    if add_phase and qpoint is None:
        raise ValueError('qpoint must be given if adding phase is needed')
    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    evec = displ_cart * m
    if add_phase:
        phase = [
            np.exp(-2j * np.pi * np.dot(pos, qpoint))
            for pos in scaled_positions
        ]
        phase = np.kron(phase, [1, 1, 1])
        evec *= phase
        evec /= np.linalg.norm(evec)
    return evec


def ixc_to_xc(ixc):
    """
    translate ixc (positive: abinit. negative: libxc) to XC.
    """
    xcdict = {
        0: 'NO-XC',
        1: 'LDA',
        2: 'LDA-PZCA',
        3: 'LDA-CA',
        4: 'LDA-Winger',
        5: 'LDA-Hedin-Lundqvist',
        6: 'LDA-X-alpha',
        7: 'LDA-PW92',
        8: 'LDA-PW92-xonly',
        9: 'LDA-PW92-xRPA',
        11: 'GGA-PBE',
        12: 'GGA-PBE-xonly',
        14: 'GGA-revPBE',
        15: 'GGA-RPBE',
        16: 'GGA-HTCH93',
        17: 'GGA-HTCH120',
        23: 'GGA-WC',
        40: 'Hartree-Fock',
        41: 'GGA-PBE0',
        42: 'GGA-PBE0-1/3',
        -1009: 'LDA-PZCA',
        -101130: 'GGA-PBE',
        -106131: 'GGA-BLYP',
        -106132: 'GGA-BP86',
        -116133: 'GGA-PBEsol',
        -118130: 'GGA-WC',
    }
    if ixc in xcdict:
        return xcdict[ixc]
    else:
        return 'libxc_%s' % ixc


class mat_data():
    def __init__(self,
                 name,
                 mag='PM',
                 description="None",
                 author='High Throughput Bot',
                 email='x.he@ulg.ac.be',
                 is_verified=False,
                 verification_info="",
                 tags=[]
                 ):
        self._already_in_db = False
        self.name = name
        self.db_directory = None
        self.all_data_directory = None
        self.mag = mag
        self.insert_time = None
        self.update_time = None
        self.log = ""
        self.description = description
        self.author = author
        self.email = email
        self.tags=tags

        self.is_verified = is_verified
        self.verification_info = verification_info

        # properties in database. should be band | phonon
        self.has_ebands = False
        self.has_phonon = False

        self.is_cubic_perovskite = True

        self.cellpar = [0] * 6
        self.cell = [0]*9
        self.natoms = 0
        self.chemical_symbols = []
        self.masses = []
        self.scaled_positions = []
        self.ispin = 0
        self.spinat = []
        self.spgroup = 1
        self.spgroup_name = 'P1'
        self.ixc = 1
        self.XC = 'PBEsol'
        self.pp_type = 'ONCV'
        self.pp_info = 'Not implemented yet.'

        self.U_type=0
        self.species=[]
        self.zion=[]
        self.U_l=[]
        self.U_u=[]
        self.U_j=[]
        self.GSR_parameters = {}
        self.energy = 0
        self.efermi = 0
        self.bandgap = 0
        self.ebands = {}

        self.kptrlatt=[]
        self.usepaw=0
        self.pawecutdg=0.0
        self.nsppol=1
        self.nspden=1

        self.emacro = [0.0] * 9
        self.becs = {}
        self.elastic = []
        self.nqpts = [1, 1, 1]
        self.special_qpts = {}

        self.phonon_mode_freqs = {}
        self.phonon_mode_names = {}
        self.phonon_mode_evecs = {}
        self.phonon_mode_phdispl = {}

        self.phonon_mode_freqs_LOTO = {}
        self.phonon_mode_names_LOTO = {}
        self.phonon_mode_evecs_LOTO = {}
        self.phonon_mode_phdispl_LOTO = {}

    def read_BAND_nc(self, fname, outputfile='Ebands.png', plot_ebands=True):
        try:
            band_file = abiopen(fname)
            self.has_ebands = True
        except Exception:
            raise IOError("can't read %s" % fname)
        self.efermi = band_file.energy_terms.e_fermie

        gap = band_file.ebands.fundamental_gaps
        if len(gap) != 0:
            for g in gap:
                self.gap = g.energy
                self.is_direct_gap = g.is_direct
        self.bandgap = self.gap
        if plot_ebands:
            fig, ax = plt.subplots()
            fig = band_file.ebands.plot(ax=ax, show=False, ylims=[-7, 5])
            fig.savefig(outputfile)

    def read_OUT_nc(self, fname):
        f = abiopen(fname)
        self.invars = f.get_allvars()
        for key in self.invars:
            if isinstance(self.invars[key], np.ndarray):
                self.invars[key] = tuple(self.invars[key])
        self.spgroup = f.spgroup[0]
        self.ixc = f.ixc[0]
        self.XC = ixc_to_xc(self.ixc)
        self.ecut = f.ecut[0]
        self.species = [chemical_symbols[int(i)] for i in f.znucl]
        if 'usepawu' in self.invars:
            self.U_type= f.usepawu[0]
        else:
            self.U_type= 0
        if self.U_type:
            self.U_l = f.lpawu
            self.U_u= [ x * Ha for x in f.upawu] 
            self.U_j= [ x* Ha for x in f.jpawu ]
        #self.nband = f.nband[0]
        self.kptrlatt = tuple(f.kptrlatt)

    def print_scf_info(self):
        for key, val in self.invars:
            print("%s : %s\n" % (key, val))

    def read_GSR_nc(self, fname):
        f = abiopen(fname)
        self.energy = f.energy
        self.stress_tensor = f.cart_stress_tensor  # unit ?
        self.forces = np.array(f.cart_forces)  # unit eV/ang

    def read_DDB(self,
                 fname=None,
                 do_label=True,
                 workdir=None,
                 phonon_output_dipdip='phonon_band_dipdip.png',
                 phonon_output_nodipdip='phonon_band_nodipdip.png'):
        """
        read phonon related properties from DDB file.
        """
        self.has_phonon = True

        ddb = abiopen(fname)

        self.ddb_header = ddb.header

        self.atoms = ddb.structure.to_ase_atoms()
        self.natoms = len(self.atoms)
        self.cellpar = self.atoms.get_cell_lengths_and_angles()
        self.cell=self.atoms.get_cell().flatten()
        self.masses = self.atoms.get_masses()
        self.scaled_positions = self.atoms.get_scaled_positions()
        self.chemical_symbols = self.atoms.get_chemical_symbols()
        self.spgroup_name = spglib.get_spacegroup(self.atoms,symprec=1e-4)

        self.ixc = self.ddb_header['ixc']
        self.XC = ixc_to_xc( self.ixc)
        self.ispin = self.ddb_header['nsppol']
        self.spinat = self.ddb_header['spinat']
        self.nband = self.ddb_header['nband']
        self.ecut = self.ddb_header['ecut'] 
        self.tsmear =self.ddb_header['tsmear'] 
        self.usepaw =self.ddb_header['usepaw']
        self.pawecutdg = self.ddb_header['tsmear'] 
        self.nsppol = self.ddb_header['nsppol']
        self.nspden= self.ddb_header['nspden']

        self.species = [chemical_symbols[int(i)] for i in self.ddb_header['znucl']]
        self.zion = [int(x) for x in self.ddb_header['zion']]
        self.znucl = [int(x) for x in self.ddb_header['znucl']]
        emacror, becsr = ddb.anaget_emacro_and_becs()
        emacro = emacror[0].cartesian_tensor
        becs_array = becsr.values
        becs = {}
        for i, bec in enumerate(becs_array):
            becs[str(i)] = bec
        nqpts = ddb._guess_ngqpt()
        qpts = tuple(ddb.qpoints.frac_coords)

        self.emacro = emacro
        self.becs = becs
        self.nqpts = nqpts
        self.qpts = qpts
        for qpt in qpts:
            qpt = tuple(qpt)
            m = ddb.anaget_phmodes_at_qpoint(qpt)
            #self.results['phonon'][qpt]['frequencies'] = m.phfreqs
            #self.results['phonon'][qpt][
            #    'eigen_displacements'] = m.phdispl_cart

        qpoints, evals, evecs, edisps = self.phonon_band(
            ddb,
            lo_to_splitting=False,
            phonon_output_dipdip=phonon_output_dipdip,
            phonon_output_nodipdip=phonon_output_nodipdip)

        #for i in range(15):
        #    print(evecs[0, :, i])
        self.special_qpts = {
            'X': (0, 0.5, 0.0),
            'M': (0.5, 0.5, 0),
            'R': (0.5, 0.5, 0.5)
        }

        zb_modes = self.label_zone_boundary_all(
            qpoints, evals, evecs, label=do_label)
        for qname in self.special_qpts:
            self.phonon_mode_freqs[qname] = zb_modes[qname][0]
            self.phonon_mode_names[qname] = zb_modes[qname][1]
            self.phonon_mode_evecs[qname] = zb_modes[qname][2]
        Gmodes = self.label_Gamma_all(qpoints, evals, evecs, label=do_label)
        self.phonon_mode_freqs['Gamma'] = Gmodes[0]
        self.phonon_mode_names['Gamma'] = Gmodes[1]
        self.phonon_mode_evecs['Gamma'] = Gmodes[2]

    def get_zb_mode(self, qname, mode_name):
        """
        return the frequencies of mode name.
        """
        ibranches = []
        freqs = []
        for imode, mode in enumerate(
                self.results['phonon']['boundary_modes'][qname]):
            freq, mname = mode
            if mname == mode_name:
                ibranches.append(imode)
                freqs.append(freq)
        return ibranches, freqs

    def get_gamma_modes(self):
        """
        return (Freqs, names, evecs)
        """
        return self.phonon_mode_freqs['Gamma'], self.phonon_mode_names['Gamma'], self.phonon_mode_evecs['Gamma'], 


    def get_gamma_mode(self, mode_name):
        """
        return the frequencies of mode name.
        """
        ibranches = []
        freqs = []
        for imode, mode in enumerate(zip(self.phonon_mode_freqs['Gamma'], self.phonon_mode_names['Gamma'])):
            freq, mname = mode
            if mname == mode_name:
                ibranches.append(imode)
                freqs.append(freq)
        return ibranches, freqs

    def label_Gamma_all(self, qpoints, evals, evecs, label=True):
        Gamma_mode_freqs = []
        Gamma_mode_names = []
        Gamma_mode_evecs = []
        for i, qpt in enumerate(qpoints):
            if np.isclose(qpt, [0, 0, 0], rtol=1e-5, atol=1e-3).all():
                evecq = evecs[i]
                for j, evec in enumerate(evecq.T):
                    freq = evals[i][j]
                    if label:
                        mode = label_Gamma(
                            evec=evec, masses=self.atoms.get_masses())
                        Gamma_mode_names.append(mode)
                    else:
                        Gamma_mode_names.append('')
                    Gamma_mode_freqs.append(freq)
                    Gamma_mode_evecs.append(np.real(evec))
            return Gamma_mode_freqs, Gamma_mode_names, Gamma_mode_evecs
        if Gamma_mode_names == []:
            print("Warning: No Gamma point found in qpoints.\n")
            return Gamma_mode_freqs, Gamma_mode_names, Gamma_mode_evecs

    def label_zone_boundary_all(self, qpoints, evals, evecs, label=True):
        mode_dict = {}
        qdict = {'X': (0, 0.5, 0.0), 'M': (0.5, 0.5, 0), 'R': (0.5, 0.5, 0.5)}
        for i, qpt in enumerate(qpoints):
            for qname in qdict:
                if np.isclose(qpt, qdict[qname], rtol=1e-5, atol=1e-3).all():
                    mode_freqs = []
                    mode_names = []
                    mode_evecs = []
                    #print "===================================="
                    #print qname
                    evecq = evecs[i]
                    for j, evec in enumerate(evecq.T):
                        freq = evals[i][j]
                        mode_freqs.append(freq)
                        if label:
                            mode = label_zone_boundary(qname, evec=evec)
                            mode_names.append(mode)
                        else:
                            mode_names.append('')
                        mode_evecs.append(np.real(evec))
                    mode_dict[qname] = (mode_freqs, mode_names, mode_evecs)
        return mode_dict

    def phonon_band(self,
                    ddb,
                    lo_to_splitting=False,
                    workdir=None,
                    phonon_output_dipdip='phonon_band_dipdip.png',
                    phonon_output_nodipdip='phonon_band_nodipdip.png',
                    show=False):
        atoms = ddb.structure.to_ase_atoms()

        if workdir is not None:
            workdir_dip = os.path.join(workdir, '/phbst_dipdip')
            #if os.path.exists(workdir_dip):
            #    os.system('rm -r %s' % workdir_dip)
        else:
            workdir_dip = None
        phbst, phdos = ddb.anaget_phbst_and_phdos_files(
            nqsmall=10,
            asr=1,
            chneut=1,
            dipdip=1,
            verbose=1,
            lo_to_splitting=True,
            anaddb_kwargs={'alphon': 1},
            workdir=workdir_dip,
            #qptbounds=kpath_bounds,
        )
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #plt.tight_layout(pad=2.19)
        #plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
        ax.axhline(0, linestyle='--', color='black')

        ax.set_title(self.name)
        ticks, labels = phbst.phbands._make_ticks_and_labels(qlabels=None)
        fig.axes[0].set_xlim([ticks[0],ticks[-1]])
        fig = phbst.phbands.plot(
            ax=ax,
            units='cm-1',
            match_bands=False,
            linewidth=1.7,
            color='blue',
            show=False)
        fig.axes[0].grid(False)

        if show:
            plt.show()
        if phonon_output_dipdip:
            fig.savefig(phonon_output_dipdip)
        plt.close()

        if workdir is not None:
            workdir_nodip = os.path.join(workdir, 'phbst_nodipdip')
            #if os.path.exists(workdir_dip):
            #    os.system('rm -r %s' % workdir_nodip)
        else:
            workdir_nodip = None
        phbst, phdos = ddb.anaget_phbst_and_phdos_files(
            nqsmall=5,
            asr=1,
            chneut=1,
            dipdip=0,
            verbose=1,
            lo_to_splitting=False,
            anaddb_kwargs={'alphon': 1},
            workdir=workdir_nodip
            #qptbounds=kpath_bounds,
        )

        fig, ax = plt.subplots(nrows=1, ncols=1)
        #plt.tight_layout(pad=2.19)
        #plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
        ax.axhline(0, linestyle='--', color='black')
        ax.set_title(self.name)


        ax.set_title(self.name)
        ticks, labels = phbst.phbands._make_ticks_and_labels(qlabels=None)
        fig.axes[0].set_xlim([ticks[0],ticks[-1]])
        fig = phbst.phbands.plot(
            ax=ax,
            units='cm-1',
            match_bands=False,
            linewidth=1.4,
            color='blue',
            show=False)
        fig.axes[0].grid(False)

        if show:
            plt.show()
        if phonon_output_dipdip:
            fig.savefig(phonon_output_nodipdip)
        plt.close()

        qpoints = phbst.qpoints.frac_coords
        nqpts = len(qpoints)
        nbranch = 3 * len(atoms)
        evals = np.zeros([nqpts, nbranch])
        evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')
        edisps = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

        masses = atoms.get_masses()
        scaled_positions = atoms.get_scaled_positions()

        for iqpt, qpt in enumerate(qpoints):
            for ibranch in range(nbranch):
                phmode = phbst.get_phmode(qpt, ibranch)
                evals[iqpt, ibranch] = phmode.freq * 8065.6
                evec = displacement_cart_to_evec(
                    phmode.displ_cart,
                    masses,
                    scaled_positions,
                    qpoint=qpt,
                    add_phase=False)
                evecs[iqpt, :, ibranch] = evec / np.linalg.norm(evec)
                edisps[iqpt, :, ibranch] = phmode.displ_cart
        return qpoints, evals, evecs, edisps


def test():
    m = mat_data()
    m.read_BAND_nc('./BAND_GSR.nc')
    m.read_OUT_nc('./OUT.nc')
    m.read_DDB('out_DDB')


#test()
