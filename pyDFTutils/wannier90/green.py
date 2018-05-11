import numpy as np

class green_J(object):
    def __init__(
            self,
            Hk,  # Hk[ispin, ikpt, iorb, jorb ]
            positions,  # [iatom, xyz]
            kpts,  # [ikpt, 3]
            kweights,  # [ikpt]
            atom_index,  # e.g. [0,1,2]
            orb_labels,  # e.g. [('px','py', 'pz'), ('dxy','dyz',...)]
            efermi,  # efermi
            emin=-30,
            emax=0.05,
            height=0.5,
            nz1=150,
            nz2=300,
            nz3=150):
        self.Hk = np.array(Hk)
        self.positions = np.array(positions)
        self.nkpts = len(kpts)
        self.kpts = np.array(kpts)
        self.kweights = np.array(kweights)

        self.atom_index = atom_index
        self.natom = len(atom_index)
        self.orb_labels = orb_labels
        self.norb = sum([len(label) for label in self.orb_labels])
        self.efermi = efermi
        #self.nz = nz1 + nz2 + nz3
        self._prepared=False
        self._current_iz=None
        self.Gloc_k=None

    def set(self, emin=-30, emax=0.05, height=0.5, nz1=150, nz2=300, nz3=150):
        self.height = height
        self.emin = emin
        self.emax = emax
        self.nz1 = nz1
        self.nz2 = nz2
        self.nz3 = nz3
        self.nz = nz1 + nz2 + nz3
        self.nz = nz1 + nz2 + nz3
        self._prepared=False

    def set_species(self, species):
        self.species=species

    def set_R_range(self, R_range):
        self.R_range=R_range

    def prepare(self):
        self.Delta = np.zeros((self.norb, self.norb), dtype='complex128')
        self._current_Gloc_k=np.zeros(( 2, self.nkpts, self.norb, self.norb), dtype='complex128')
        self._get_H_index()
        self._prepare_elist()
        print("calculate Delta in real space")
        self.calc_Delta_realspace()
        self._prepared=True
        #print("calculate Gloc in k space")
        #self._calc_Gloc_k()

    def _prepare_elist(self):
        nz1, nz2, nz3 = self.nz1, self.nz2, self.nz3
        nz=self.nz
        self.elist = np.zeros(self.nz + 1, dtype='complex128')

        self.elist[:nz1]=self.emin + np.linspace(0, self.height, nz1, endpoint=False)*1j 
        self.elist[nz1:nz1+nz2]= self.emin + self.height*1j + np.linspace(0, self.emax-self.emin, nz2, endpoint=False)
        self.elist[nz1+nz2: nz]= self.emax + self.height*1j + np.linspace(0, -self.height, nz3, endpoint=False)*1j
        self.elist[-1]=self.emax # emax

        #for i in range(self.nz1): # [emin, emin+height*1i)
        #    self.elist[i] = self.emin + 1j * self.height / self.nz1 * i
        #for i in range(self.nz2):  #[emin+height*1i, emax+height*1i)
        #    self.elist[i + nz1] = self.elist[nz1 - 1] + (self.emax - self.emin
        #                                                 ) / self.nz2 *(i+1)
        #for i in range(self.nz3): # [emax+height*1i, emax)
        #    self.elist[i + nz1 + nz2] = self.elist[nz1 + nz2 - 1] + (
        #        -1j * self.height) / (self.nz3-1) * (i+1)
        print(self.elist)

    def _get_H_index(self):
        """
        generate:
        iatom_start
        iatom_end
        """
        self.iatom_start = []
        self.iatom_end = []
        i = 0
        print(self.orb_labels)
        for iatom in range(self.natom):#self.atom_index:
            m = len(self.orb_labels[iatom])
            self.iatom_start.append(i)
            self.iatom_end.append(i + m)
            i = i + m

    def get_Hk_iatom(self, spin, ik, iatom):
        return self.Hk[ispin, ik, self.iatom_start[iatom]:self.iatom_end[iatom]]

    def _calc_Gloc_k(self):
        self.Gloc_k = np.zeros(
            (self.nz, 2, self.nkpts, self.norb, self.norb), dtype='complex128')
        for iz, energy in enumerate(self.elist[:-1]):
            for ispin in [0, 1]:
                for ik in range(self.nkpts):
                    self.Gloc_k[iz, ispin, ik] = np.linalg.inv(
                        -self.Hk[ispin, ik] + np.eye(self.norb) * (
                            energy + self.efermi))

    def get_Gloc_k(self, iz, ispin, ikpt, iatom, jatom):
        if self.Gloc_k is None:
            self._calc_Gloc_k()
        istart = self.iatom_start[iatom]
        jstart = self.iatom_start[jatom]
        iend = self.iatom_end[iatom]
        jend = self.iatom_end[jatom]
        return self.Gloc_k[iz, ispin, ikpt, istart:iend, jstart:jend]

    def _get_Gloc_k(self, iz, ispin, ikpt, iatom, jatom):
        istart = self.iatom_start[iatom]
        jstart = self.iatom_start[jatom]
        iend = self.iatom_end[iatom]
        jend = self.iatom_end[jatom]
        if iz!=self._current_iz: # calculate if needed
            for iispin in [0, 1]:
                for iikpt in range(self.nkpts):
                    self._current_Gloc_k[iispin, iikpt,:,:]=np.linalg.inv(-self.Hk[iispin, iikpt, :, :] + np.eye(self.norb) * (
             self.elist[iz] + self.efermi))
        self._current_iz=iz
        return self._current_Gloc_k[ispin, ikpt, istart:iend, jstart:jend]

    def get_Gloc_realspace(self,iatom, jatom, R, iz, ispin):
        istart = self.iatom_start[iatom]
        jstart = self.iatom_start[jatom]
        iend = self.iatom_end[iatom]
        jend = self.iatom_end[jatom]
        s = np.zeros((iend - istart, jend - jstart), dtype='complex128')
        for ikpt in range(self.nkpts):
            s += self.kweights[ikpt] * np.exp(
                1j * 2.0 * np.pi * np.dot(
                    -np.array(R), # - self.positions[jatom] + self.positions[iatom],
                    self.kpts[ikpt])) *self.get_Gloc_k(iz, ispin, ikpt, iatom, jatom)
                    #self.kpts[ikpt])) *self.Gloc_k[iz, ispin, ikpt, istart:iend, jstart:jend]
        return s

    def calc_Delta_realspace(self):
        for ik in range(self.nkpts):
            self.Delta = self.Delta + self.kweights[ik] * (self.Hk[0, ik] -
                                                           self.Hk[1, ik])

    def get_Delta(self, iatom):
        istart = self.iatom_start[iatom]
        iend = self.iatom_end[iatom]
        return self.Delta[istart:iend, istart:iend]

    def get_J_e(self, iatom, jatom, R, iz):
        return np.matmul(
                    np.matmul(
                        self.get_Delta(iatom),
                        self.get_Gloc_realspace(iatom, jatom, R, iz, 1)),
                    np.matmul(
                        self.get_Delta(jatom),
                        self.get_Gloc_realspace(jatom, iatom, -np.array(R), iz, 0)))

    def get_J(self, iatom, jatom, R):
        J_ijR = 0.0
        for iz, energy in enumerate(self.elist[:-1]):
            de = self.elist[iz + 1] - self.elist[iz]
            J_ijR = J_ijR + np.imag(de*self.get_J_e(iatom, jatom, R, iz))
        Jorb=J_ijR * (-1.0 / (2.0 * np.pi))
        J=np.sum(np.diag(Jorb))
        #return J_ijR * (-1.0 / (2.0 * np.pi)), np.sum(np.diag(J_ijR))
        return Jorb, J

    def get_atom_occupations(self, iatom):
        norb_iatom=self.iatom_end[iatom]-self.iatom_start[iatom]
        occ=np.zeros((2, norb_iatom), dtype='complex128')
        for ispin in range(2):
            for iz in range(self.nz-1):
                de=self.elist[iz+1]-self.elist[iz]
                occ[ispin, :] += -1.0/np.pi * np.imag(de*np.diag(self.get_Gloc_realspace(iatom ,iatom, R=np.zeros(3), iz=iz, ispin=ispin)))
        return np.real(occ)

    def get_occupations(self):
        tot_occ=0.0
        for iatom in range(self.natom):
            print("occupation of atom %s"%iatom)
            occ_i=self.get_atom_occupations(iatom)
            tot_occ+=np.sum(occ_i)
            print(occ_i)
        print("Total_charge: ", tot_occ)

