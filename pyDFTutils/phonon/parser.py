#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from abipy.abilab import abiopen
from pyDFTutils.perovskite.perovskite_mode import label_zone_boundary, label_Gamma


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


class mat_data():
    def __init__(self,
                 name,
                 mag='PM',
                 description="None",
                 author='High Throughput Bot',
                 email='x.he@ulg.ac.be',
                 is_verified=False,
                 verification_info=""):
        self._already_in_db = False
        self.name = name
        self.db_directory = None
        self.all_data_directory = None
        self.mag = 'PM'
        self.insert_time = None
        self.update_time = None
        self.log = ""
        self.description = ""
        self.author = 'High Throughput Bot'
        self.email = 'x.he@ulg.ac.be'

        self.is_verified = is_verified
        self.verification_info = verification_info

        # properties in database. should be band | phonon
        self.has_ebands = False
        self.has_phonon = False

        self.is_cubic_perovskite = True

        self.cellpar = [0] * 6
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

        self.U = {'ldau_type': 1, 'ldau_luj': {}}
        self.GSR_parameters = {}
        self.energy = 0
        self.efermi = 0
        self.bandgap = 0
        self.ebands = {}

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

    def read_BAND_nc(self, fname, outputfile=None):
        try:
            band_file = abiopen(fname)
            self.has_eband = True
        except Exception:
            raise IOError("can't read %s" % fname)
        self.efermi = self.band_file.energy_terms.e_fermie

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
        f= abiopen(fname)
        self.invars = f.get_allvars()
        for key in self.invars:
            if isinstance(self.invars[key], np.ndarray):
                self.invars[key] = tuple(self.invars[key])
        self.spgroup = f.spgroup[0]
        self.ixc = f.ixc[0]
        self.ecut = f.ecut[0]
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

    def read_DDB(self, fname=None, do_label=True):
        """
        read phonon related properties from DDB file.
        """
        self.has_phonon = True

        ddb = abiopen(fname)

        self.ddb_header = ddb.header

        self.atoms = ddb.structure.to_ase_atoms()
        self.natoms = len(self.atoms)
        self.cellpar = self.atoms.get_cell_lengths_and_angles()
        self.masses = self.atoms.get_masses()
        self.scaled_positions = self.atoms.get_scaled_positions()
        self.chemical_symbols = self.atoms.get_chemical_symbols()

        self.ixc = self.ddb_header['ixc']
        self.ispin = self.ddb_header['nsppol']
        self.spinat = self.ddb_header['spinat']
        self.nband = self.ddb_header['nband']

        emacror, becsr = ddb.anaget_emacro_and_becs()
        emacro = emacror[0].cartesian_tensor
        becs_array = becsr.values
        becs = {}
        for i, bec in enumerate(becs_array):
            becs[str(i)] = becs_array
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
            ddb, lo_to_splitting=False)

        #for i in range(15):
        #    print(evecs[0, :, i])
        self.special_qpts = {
            'X': (0, 0.5, 0.0),
            'M': (0.5, 0.5, 0),
            'R': (0.5, 0.5, 0.5)
        }

        if do_label:
            zb_modes = self.label_zone_boundary(qpoints, evals, evecs)
            for qname in self.special_qpts:
                self.phonon_mode_freqs[qname] = zb_modes[qname][0]
                self.phonon_mode_names[qname] = zb_modes[qname][1]
                self.phonon_mode_evecs[qname] = zb_modes[qname][2]
            Gmodes = self.label_Gamma(qpoints, evals, evecs)
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
        return self.results['phonon']['Gamma_modes']

    def get_gamma_mode(self, mode_name):
        """
        return the frequencies of mode name.
        """
        ibranches = []
        freqs = []
        for imode, mode in enumerate(self.results['phonon']['Gamma_modes']):
            freq, mname = mode
            if mname == mode_name:
                ibranches.append(imode)
                freqs.append(freq)
        return ibranches, freqs

    def label_Gamma(self, qpoints, evals, evecs):
        Gamma_mode_freqs = []
        Gamma_mode_names = []
        Gamma_mode_evecs= []
        for i, qpt in enumerate(qpoints):
            if np.isclose(qpt, [0, 0, 0], rtol=1e-5, atol=1e-3).all():
                evecq = evecs[i]
                for j, evec in enumerate(evecq.T):
                    mode = label_Gamma(
                        evec=evec, masses=self.atoms.get_masses())
                    freq = evals[i][j]
                    Gamma_mode_names.append(mode)
                    Gamma_mode_freqs.append(freq)
                    Gamma_mode_evecs.append(np.real(evec))
            return Gamma_mode_freqs, Gamma_mode_names, Gamma_mode_evecs
        if Gamma_mode_names == []:
            print("Warning: No Gamma point found in qpoints.\n")
            return Gamma_mode_freqs, Gamma_mode_names, Gamma_mode_evecs

    def label_zone_boundary(self, qpoints, evals, evecs):
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
                        mode = label_zone_boundary(qname, evec=evec)
                        freq = evals[i][j]
                        mode_freqs.append(freq)
                        mode_names.append(mode)
                        mode_evecs.append(np.real(evec))
                    mode_dict[qname] = (mode_freqs, mode_names,mode_evecs)
        return mode_dict

    def phonon_band(self, ddb, lo_to_splitting=False):
        atoms = ddb.structure.to_ase_atoms()
        phbst, phdos = ddb.anaget_phbst_and_phdos_files(
            nqsmall=5,
            asr=1,
            chneut=1,
            dipdip=0,
            verbose=1,
            lo_to_splitting=False,
            #qptbounds=kpath_bounds,
        )

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
