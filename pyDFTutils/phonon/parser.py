#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from abipy.abilab import abiopen
from perovskite_mode import label_zone_boundary, label_Gamma

property_categories = ['scf', 'phonon', 'relax']


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
    def __init__(self):
        self._id = None
        self._uid = None
        self._name = None
        self._db_directory = None
        self._all_data_directory = None

        self._insert_time = None
        self._update_time = None
        self._log = ""
        self._note = ""
        self._author = None
        self._email = None

        self._checked = False
        self._check_info = ""

        # properties in database. should be band | phonon
        self._properties = []

        self._results = {}
        for p in property_categories:
            self._results[p] = {}

    def read_OUT_nc(self, fname):
        pass

    def read_DDB(self, fname=None, do_label=True):
        """
        read phonon related properties from DDB file.
        """
        if 'phonon' not in self._properties:
            self._properties.append('phonon')

        ddb = abiopen(fname)

        self.atoms = ddb.structure.to_ase_atoms()

        emacror, becsr = ddb.anaget_emacro_and_becs()
        emacro = emacror[0].cartesian_tensor
        becs = becsr.values
        nqpts = ddb._guess_ngqpt()
        qpts = tuple(ddb.qpoints.frac_coords)

        self._results['phonon']['emacro'] = emacro
        self._results['phonon']['becs'] = becs
        self._results['phonon']['nqpts'] = nqpts
        self._results['phonon']['qpoints'] = qpts
        for qpt in qpts:
            qpt = tuple(qpt)
            m = ddb.anaget_phmodes_at_qpoint(qpt)
            #self._results['phonon'][qpt]['frequencies'] = m.phfreqs
            #self._results['phonon'][qpt][
            #    'eigen_displacements'] = m.phdispl_cart

        qpoints, evals, evecs, edisps = self.phonon_band(
            ddb, lo_to_splitting=False)

        #for i in range(15):
        #    print(evecs[0, :, i])

        if do_label:
            zb_modes = self.label_zone_boundary(qpoints, evals, evecs)
            Gmodes = self.label_Gamma(qpoints, evals, evecs)
            self._results['phonon']['boundary_modes'] = zb_modes
            self._results['phonon']['Gamma_modes'] = Gmodes

    def get_zb_mode(self, qname, mode_name):
        """
        return the frequencies of mode name.
        """
        ibranches = []
        freqs = []
        for imode, mode in enumerate(
                self._results['phonon']['boundary_modes'][qname]):
            freq, mname = mode
            if mname == mode_name:
                ibranches.append(imode)
                freqs.append(freq)
        return ibranches, freqs

    def get_gamma_mode(self,  mode_name):
        """
        return the frequencies of mode name.
        """
        ibranches = []
        freqs = []
        for imode, mode in enumerate(self._results['phonon']['Gamma_modes']):
            freq, mname = mode
            if mname == mode_name:
                ibranches.append(imode)
                freqs.append(freq)
        return ibranches, freqs

    def label_Gamma(self, qpoints, evals, evecs):
        Gamma_modes = []
        for i, qpt in enumerate(qpoints):
            if np.isclose(qpt, [0, 0, 0], rtol=1e-5, atol=1e-3).all():
                evecq = evecs[i]
                for j, evec in enumerate(evecq.T):
                    mode = label_Gamma(
                        evec=evec, masses=self.atoms.get_masses())
                    freq = evals[i][j]
                    Gamma_modes.append([freq, mode])
            return Gamma_modes
        if Gamma_modes == []:
            print("Warning: No Gamma point found in qpoints.\n")
            return Gamma_modes

    def label_zone_boundary(self, qpoints, evals, evecs):
        mode_dict = {}
        qdict = {'X': (0, 0.5, 0.0), 'M': (0.5, 0.5, 0), 'R': (0.5, 0.5, 0.5)}
        for i, qpt in enumerate(qpoints):
            for qname in qdict:
                if np.isclose(qpt, qdict[qname], rtol=1e-5, atol=1e-3).all():
                    mode_dict[qname] = []
                    print "===================================="
                    print qname
                    evecq = evecs[i]
                    for j, evec in enumerate(evecq.T):
                        mode = label_zone_boundary(qname, evec=evec)
                        freq = evals[i][j]
                        mode_dict[qname].append([freq, mode])
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
                evals[iqpt, ibranch] = phmode.freq
                evec = displacement_cart_to_evec(
                    phmode.displ_cart,
                    masses,
                    scaled_positions,
                    qpoint=qpt,
                    add_phase=False)
                evecs[iqpt, :, ibranch] = evec / np.linalg.norm(evec)
                edisps[iqpt, :, ibranch] = phmode.displ_cart
        return qpoints, evals, evecs, edisps
