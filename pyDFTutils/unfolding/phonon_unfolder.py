"""
Phonon unfolding: Reciprocal space method. The method is described in
P. B. Allen et al. Phys Rev B 87, 085322 (2013).
This method should be also applicable to other bloch waves on discrete grid, eg. electrons wave function in wannier basis set, magnons, etc. Now only phonon istested.
"""
from ase.build import make_supercell
from ase.atoms import Atoms
import numpy as np


class phonon_unfolder:
    """ phonon unfolding class"""

    def __init__(self,
                 atoms,
                 supercell_matrix,
                 eigenvectors,
                 qpoints,
                 tol_r=0.03,
                 ndim=3,
                 labels=None,
                 compare=None,
                 phase=True,
                 ghost_atoms=None):
        """
        Params:
        ===================
        atoms: The structure of supercell.
        supercell matrix: The matrix that convert the primitive cell to supercell.
        eigenvectors: The phonon eigenvectors. format np.array() index=[ikpts, ifreq, 3*iatoms+j]. j=0..2
        qpoints: list of q-points, note the q points are in the BZ of the supercell.
        tol_r: tolerance. If abs(a-b) <r, they are seen as the same atom.
        ndim: number of dimensions. For 3D phonons, use ndim=3. For electrons(no spin), ndim=1. For spinors, use ndim=2 (TODO: spinor not tested. is it correct?).
        labels: labels of the basis. for 3D phonons, ndim can be set to 1 alternately, with labels set to ['x','y','z']*natoms. The labels are used to decide if two basis are identical by translation. (Not used for phonon)
        compare: how to decide the basis are identical (Not used for phonon)
        ghost_atoms: scaled positions of ghost atoms in case of vacancies.
        """
        self._atoms = atoms
        self._scmat = supercell_matrix
        self._evecs = eigenvectors
        self._qpts = qpoints
        self._tol_r = tol_r
        self._ndim = ndim
        self._labels = labels
        self._trans_rs = None
        self._trans_indices = None

        if ghost_atoms is not None:
            print("adding ghosts")
            symbols = atoms.get_chemical_symbols()
            positions = list(atoms.get_scaled_positions())
            cell = atoms.get_cell()
            for gatom in ghost_atoms:
                # sorry Rn, you are the chosen one to be the ghost. unlikely to appear in a phonon calculation.
                symbols.append('C')
                positions.append(gatom)
            self._atoms = Atoms(symbols, scaled_positions=positions, cell=cell)
            # set the eigenvectors to zeros for the ghosts.
            nkpt, nbranch, neigdim = self._evecs.shape
            evecs = np.zeros((nkpt, nbranch+ 3 * len(ghost_atoms), neigdim + 3 * len(ghost_atoms)), dtype=self._evecs.dtype)
            evecs[:,:neigdim, :neigdim]=self._evecs
            self._evecs=evecs
        self._make_translate_maps()
        self._phase = phase

    def _translate(self, evec, r):
        """
        T(r) psi: r is integer numbers of primitive cell lattice matrix.
        Params:
        =================
        evec: an eigen vector of supercell
        r: The translate vector

        Returns:
        ================
         tevec: translated vector.
        """
        pass

    def _make_translate_maps(self):
        """
        find the mapping between supercell and translated cell.
        Returns:
        ===============
        A N * (ndim*natoms) array.
        index[i] is the mapping from supercell to translated supercell so that
        T(r_i) psi = psi[indices[i]].

        TODO: vacancies/add_atoms not supported. How to do it? For vacancies, a ghost atom can be added. For add_atom, maybe we can just ignore them? Will it change the energy spectrum?
        """
        a1 = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
        sc = make_supercell(a1, self._scmat)
        rs = sc.get_scaled_positions()
        positions = np.array(self._atoms.get_scaled_positions())
        indices = np.zeros(
            [len(rs), len(positions) * self._ndim], dtype='int32')
        for i, ri in enumerate(rs):
            inds = []
            Tpositions = positions + np.array(ri)
            close_to_int = lambda x: np.all(np.abs(x - np.round(x)) < self._tol_r)
            for i_atom, pos in enumerate(positions):
                jatoms=None
                d=100000
                for j_atom, Tpos in enumerate(Tpositions):
                    dpos = Tpos - pos
                    dj=np.linalg.norm(dpos - np.round(dpos))
                    if dj<d:
                        d=dj
                        jatom=j_atom
                    #if close_to_int(dpos):
                #indices[i, j_atom * self._ndim:j_atom * self._ndim +
                #                self._ndim] = np.arange(
                #                    i_atom * self._ndim,
                #                    i_atom * self._ndim + self._ndim)
                indices[i, jatom * self._ndim:jatom * self._ndim +
                                self._ndim] = np.arange(
                                    i_atom * self._ndim,
                                    i_atom * self._ndim + self._ndim)


        self._trans_rs = rs
        self._trans_indices = indices
        print(indices)
        for ind in indices:
            print(len(ind), len(set(ind)))

    def get_weight(self, evec, qpt, G=np.array([0, 0, 0])):
        """
        get the weight of a mode which has the wave vector of qpt and eigenvector of evec.
        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]
        """
        weight = 0j
        N = len(self._trans_rs)
        for r_i, ind in zip(self._trans_rs, self._trans_indices):
            if self._phase:
                #weight += np.vdot(evec, evec[ind]*np.exp(-1j * np.dot(qpt+G,r_i)) ) /N
                #r_i =np.dot(self._scmat,r_i)
                weight += np.vdot(evec, evec[ind]) * np.exp(
                    -1j * 2 * np.pi * np.dot(qpt + G, r_i)) / N
            else:
                weight += (np.vdot(evec, evec[ind]) ) / N * np.exp(
                    -1j * 2 * np.pi * np.dot(G, r_i))

        return weight.real

    def get_weights(self):
        """
        Get the weight for all the modes.
        """
        nqpts, nfreqs = self._evecs.shape[0], self._evecs.shape[1]
        weights = np.zeros([nqpts, nfreqs])
        for iqpt in range(nqpts):
            for ifreq in range(nfreqs):
                weights[iqpt, ifreq] = self.get_weight(
                    self._evecs[iqpt, :, ifreq], self._qpts[iqpt])

        self._weights = weights
        return self._weights
