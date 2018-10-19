# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from pythtb import tb_model,w90
from ase.calculators.interface import Calculator,DFTCalculator
from ase.dft.dos import DOS
from ase.dft.kpoints import monkhorst_pack
import numpy as np
#from tetrahedronDos import tetrahedronDosClass
from occupations import Occupations
from pyDFTutils.wannier90.band_plot import plot_band_weight
import matplotlib.pyplot as plt


class my_tb_model(tb_model):
    """
    :param dim_k: Dimensionality of reciprocal space, i.e., specifies how
      many directions are considered to be periodic.

    :param dim_r: Dimensionality of real space, i.e., specifies how many
      real space lattice vectors there are and how many coordinates are
      needed to specify the orbital coordinates.

    .. note:: Parameter *dim_r* can be larger than *dim_k*! For example,
      a polymer is a three-dimensional molecule (one needs three
      coordinates to specify orbital positions), but it is periodic
      along only one direction. For a polymer, therefore, we should
      have *dim_k* equal to 1 and *dim_r* equal to 3. See similar example
      here: :ref:`trestle-example`.

    :param lat: Array containing lattice vectors in Cartesian
      coordinates (in arbitrary units). In example the below, the first
      lattice vector has coordinates [1.0,0.5] while the second
      one has coordinates [0.0,2.0].

    :param orb: Array containing reduced coordinates of all
      tight-binding orbitals. In the example below, the first
      orbital is defined with reduced coordinates [0.2,0.3]. Its
      Cartesian coordinates are therefore 0.2 times the first
      lattice vector plus 0.3 times the second lattice vector.

    :param per: This is an optional parameter giving a list of lattice
      vectors which are considered to be periodic. In the example below,
      only the vector [0.0,2.0] is considered to be periodic (since
      per=[1]). By default, all lattice vectors are assumed to be
      periodic. If dim_k is smaller than dim_r, then by default the first
      dim_k vectors are considered to be periodic.

    :param nspin: Number of spin components assumed for each orbital
      in *orb*. Allowed values of *nspin* are *1* and *2*. If *nspin*
      is 1 then the model is spinless, if *nspin* is 2 then it is a
      spinfull model and each orbital is assumed to have two spin
      components. Default value of this parameter is *1*.

    :param nel: Number of electrons, if fix_spin , nel should be a tuple of (nel_up,nel_dn)

    :param width: smearing width

    :param verbose: whether to print some information in detail.
    :param fix_spin: whether to fix the spin polarization.
    Example usage::

       # Creates model that is two-dimensional in real space but only
       # one-dimensional in reciprocal space. Second lattice vector is
       # chosen to be periodic (since per=[1]). Three orbital
       # coordinates are specified.
       tb = tb_model(1, 2,
                   lat=[[1.0, 0.5], [0.0, 2.0]],
                   orb=[[0.2, 0.3], [0.1, 0.1], [0.2, 0.2]],
                   per=[1])


    """
    def __init__(self,dim_k,dim_r,lat,orb,per=None,nspin=1,nel=None,width=0.2,verbose=True,fix_spin=False):
        """

        :param nel: number of electrons.
        :param width: smearing width
        :param verbose: verbose
        :param fix_spin: whether to fix the spin polarization.
        """
        #super(mytb,self,dim_k,dim_r,lat,orb,per=per,nspin=nspin)
        tb_model.__init__(self,dim_k,dim_r,lat,orb,per=per,nspin=nspin)
        self._eigenvals=None
        self._eigenvecs=None
        self._nspin=nspin
        # if fix_spin: _efermi is a tuple (ef_up,ef_dn)
        self._efermi=None
        self._occupations=None
        self._kpts=None
        # _kweight is a  array [w1,w2,....].
        self._kweights=None
        self._nel=nel
        self._old_occupations=None

        self._width=width
        self._verbose=verbose
        self._eps=0.001
        self.fix_spin=fix_spin
        self._nbar=np.ndarray([len(orb),2])

    def set_kpoints(self,kpts):
        """
        set the kpoints to calculate. each kpoint can be a
        """
        if len(kpts[0])==self._dim_k:
            self._kpts=kpts
            self._kweights=np.array([1.0/len(self._kpts)]*len(self._kpts))
        elif len(kpts[0])==self.dim_k+1:
            self._kpts=kpts[:,:-1]
            self._kweights=kpts[:,-1]

    def get_number_of_bands(self):
        """
        number of bands.
        """
        return self.get_num_orbitals()

    def solve_all(self,k_list=None,eig_vectors=False):
        if eig_vectors:
            self._eigenvals,self._eigenvecs=tb_model.solve_all(self,k_list=k_list,eig_vectors=eig_vectors)
        else:
            self._eigenvals=tb_model.solve_all(self,k_list=k_list,eig_vectors=eig_vectors)
        return self._eigenvals


    def get_eigenvalues(self,kpt=0,spin=None,refresh=False):
        """
        Ak_spin. Calculate the eigenvalues and eigen vectors. the eigenvalues are returned.
        self._eigenvals are returned.
        """
        if self._eigenvals is None or refresh:
            #print self.solve_all(k_list=self._kpts,eig_vectors=True)
            self._eigenvals,self._eigenvecs=self.solve_all(k_list=self._kpts,eig_vectors=True)
        if spin is None or self._nspin==1:
            return self._eigenvals[:,kpt]
        else:
            ## seperate the spin up/ down
            ## project the evec to spin up/down basis
            eval_up=[]#np.zeros(self._norb)
            eval_dn=[]#np.zeros(self._norb)
            for ib,eval in enumerate(self._eigenvals[:,kpt]):
                vec_up=self._eigenvecs[ib,kpt,:,0]
                vec_dn=self._eigenvecs[ib,kpt,:,1]
                #if np.abs(np.abs(vec_up)).sum()>np.abs(np.abs(vec_dn)).sum():
                if np.linalg.norm(vec_up)>np.linalg.norm(vec_dn):
                    eval_up.append(eval)
                else:
                    eval_dn.append(eval)
            eval_up=np.array(eval_up)
            eval_dn=np.array(eval_dn)

            #if len(eval_up)!=4:
                #print "!=4"
                #print eval_up
                #print eval_dn
                #print self._eigenvecs[:,kpt,:,0]
                #print self._eigenvecs[:,kpt,:,1]
            if spin==0 or spin=='UP':
                #return self._eigenvals[::2,kpt]
                return eval_up
            if spin==1 or spin=='DOWN':
                return eval_dn
                #return self._eigenvals[1::2,kpt]


    def get_fermi_level(self):
        if self._efermi==None:
            print("Warning: Efermi not calculated yet. Using 0 instead.")
            return 0.0
        else:
            return self._efermi

    def get_bz_k_points(self):
        return self._kpts

    def get_ibz_k_points(self):
        raise NotImplementedError

    def get_k_point_weights(self):
        return self._kweights
    def get_number_of_spins(self):
        return self._nspin

    def get_dos(self,width=0.15,method='gaussian',npts=501):
        """
        density of states.

        :param width: smearing width
        :param method: 'gaussian'| 'tetra'
        :param npts: number of DOS energies.

        :returns:
          energies, dos. two ndarray.
        TODO: implement spin resolved DOS.
        """
        if method=='tetra':
            dos=tetrahedronDosClass(self,width,npts=npts)
        else:
            dos=DOS(self,width,window=None,npts=npts)
        return dos.get_energies(),dos.get_dos()

    #def get_pdos()



    def get_occupations(self,nel,width=0.2,refresh=False):
        """
        calculate occupations of each eigenvalue.
        the the shape of the occupation is the same as self._eigenvals.
        [eig_k1,eigk2,...], each eig_k is a column with the length=nbands.

        if nspin=2 and fix_spin, there are two fermi energies. NOTE: this conflicts with the DOS caluculation. FIXME.

        :param nel: number of electrons. if fix_spin, the nel is a tuple of (nel_up,nel_dn)

        :Returns:
          self._occupations (np.ndarray) index:[band,kpt,orb,spin] if nspin==2 else [band,kpt,orb] same as eigenvec
        """
        self._nel=nel
        self.get_eigenvalues(refresh=refresh)
        #print self._kweights
        #print self._eigenvals
        if self._nspin ==1 or not self.fix_spin:
            occ=Occupations(nel,width,self._kweights,nspin=self._nspin)
            self._occupations=occ.occupy(self._eigenvals)
            self._efermi=occ.get_mu()
        elif self._nspin==2 and self.fix_spin:
            raise NotImplementedError("current implement on fix_spin is not correct.")
            u"""FIXME: 这根本就不对，eig无法被直接区分为eig_up,eig_dn，不能这样处理"""
            nel_up,nel_dn=nel
            eig_up=self.eigenvals[::2]
            eig_dn=self.eigenvals[1::2]

            occ_up=Occupations(nel_up,width,self._kweights,nspin=1)
            occupations_up=occ_up.occupy(eig_up)
            efermi_up=occ_up.get_mu()

            occ_dn=Occupations(nel_dn,width,self._kweights,nspin=1)
            occupations_dn=occ_dn.occupy(eig_dn)
            efermi_dn=occ_dn.get_mu()

            self._occupations[::2]=occupations_up
            self._occupations[1::2]=occupations_dn

            self.efermi=(efermi_up,efermi_dn)

        return self._occupations

    def get_orbital_occupations(self,refresh=True):
        """
        self.occupations:
        if spin==1: the indexes are [orb];
        if spin==2: the indexes are [orb,spin]
        """
        A2= np.abs(self._eigenvecs)**2
        #first sum over band
        #print self._occupations.shape
        #print(A2.sum(axis=0))
        # occupations: index same as eigenval. [band, k]
        ni,nk=self._occupations.shape
        V2=np.zeros(A2.shape,dtype=float)
        if self._nspin==1:
            for i in range(ni):
                for j in range(nk):
                    V2[i,j]=self._occupations[i,j]*A2[i,j]*self._kweights[j]
            #V2=self._occupations.flatten()*A2.reshape(ni*nk,ni)/len(self._kweights)
            #V2=(self._occupations*A2).sum(axis=(0,1))#/len(self._kweights)
            V2=V2.sum(axis=(0,1))
        elif self._nspin==2:
            for i in range(ni):
                for j in range(nk):
                    V2[i,j]=self._occupations[i,j]*A2[i,j]*self._kweights[j]
            #V2=self._occupations.flatten()*A2.reshape(ni*nk,ni)/len(self._kweights)
            #V2=(self._occupations*A2).sum(axis=(0,1))#/len(self._kweights)

            self._nbar=V2.sum(axis=(0,1))
        return self._nbar

    def get_band_energy(self):
        """
        Not free energy. total energy. sum of occupied levels.
        """
        self.energy=(self._kweights*(self._occupations*self._eigenvals)).sum()

    def get_free_energy(self):
        pass
        #raise NotImplementedError

    def get_projection(self,orb,spin=0):
        """
        get the projection to nth orb.

        :param orb: the index of the orbital.
        :param spin: if spin polarized, 0 or 1

        :returns: eigenvecs[iband,ikpt]
        """
        if self._nspin==2:
            return self._eigenvecs[:,:,orb,spin]
        else:
            return self._eigenvecs[:,:,orb]

    def plot_projection(self,orb,spin=0,color='blue',axis=None):
        """
        plot the projection of the band to the basis
        """
        kslist=[list(range(len(self._kpts)))]*self._norb
        ekslist=self._eigenvals
        wkslist=np.abs(self.get_projection(orb,spin=spin))

        #fig,a = plt.subplots()
        return plot_band_weight(kslist,ekslist,wkslist=wkslist,efermi=None,yrange=None,output=None,style='alpha',color=color,axis=axis,width=10,xticks=None)


    def get_pdos(self):
        """
        get projected dos to the basis set.
        """
        raise NotImplementedError('Projected DOS is not yet implemented!')

    def get_band_gap_and_edges(self, nel=None,kpts=None):
        """
        return band gap, VBM and CBM. The VBM is defined as the highest nelth band energy.
        """
        VBM=None
        CBM=None
        if nel is None:
            nel=self._nel
        if kpts is None:
            kpts=self._kpts
        if self._eigenvals is None:
            eigvals = self.solve_all(k_list=kpts,eig_vectors=False)
        VBM=np.maximum(eigvals[nel-1,:])
        CBM=np.minimum(eigvals[nel,:])
        band_gap=CBM-VBM
        return band_gap, VBM, CBM





class myw90(w90):
    """
    wrapper of pythtb.w90, the only difference is that it use mypythtb as model class.
    """
    def model(self,zero_energy=0.0,min_hopping_norm=None,max_distance=None,ignorable_imaginary_part=None):
        """

        This function returns :class:`pythtb.tb_model` object that can
        be used to interpolate the band structure at arbitrary
        k-point, analyze the wavefunction character, etc.  

        The tight-binding basis orbitals in the returned object are
        maximally localized Wannier functions as computed by
        Wannier90.  The orbital character of these functions can be
        inferred either from the *projections* block in the
        *prefix*.win or from the *prefix*.nnkp file.  Please note that
        the character of the maximally localized Wannier functions is
        not exactly the same as that specified by the initial
        projections.  One way to ensure that the Wannier functions are
        as close to the initial projections as possible is to first
        choose a good set of initial projections (for these initial
        and final spread should not differ more than 20%) and then
        perform another Wannier90 run setting *num_iter=0* in the
        *prefix*.win file.

        Number of spin components is always set to 1, even if the
        underlying DFT calculation includes spin.  Please refer to the
        *projections* block or the *prefix*.nnkp file to see which
        orbitals correspond to which spin.

        Locations of the orbitals in the returned
        :class:`pythtb.tb_model` object are equal to the centers of
        the Wannier functions computed by Wannier90.
        
        :param zero_energy: Sets the zero of the energy in the band
          structure.  This value is typically set to the Fermi level
          computed by the density-functional code (or to the top of the
          valence band).  Units are electron-volts.

        :param min_hopping_norm: Hopping terms read from Wannier90 with
          complex norm less than *min_hopping_norm* will not be included
          in the returned tight-binding model.  This parameters is
          specified in electron-volts.  By default all terms regardless
          of their norm are included.
        
        :param max_distance: Hopping terms from site *i* to site *j+R* will
          be ignored if the distance from orbital *i* to *j+R* is larger
          than *max_distance*.  This parameter is given in Angstroms.
          By default all terms regardless of the distance are included.

        :param ignorable_imaginary_part: The hopping term will be assumed to
          be exactly real if the absolute value of the imaginary part as
          computed by Wannier90 is less than *ignorable_imaginary_part*.
          By default imaginary terms are not ignored.  Units are again
          eV.

        :returns:
           * **tb** --  The object of type :class:`pythtb.tb_model` that can be used to
               interpolate Wannier90 band structure to an arbitrary k-point as well
               as to analyze the character of the wavefunctions.  Please note 

        Example usage::

          # returns tb_model with all hopping parameters
          my_model=silicon.model()

          # simplified model that contains only hopping terms above 0.01 eV
          my_model_simple=silicon.model(min_hopping_norm=0.01)
          my_model_simple.display()
        
        """    

        # make the model object
        tb=my_tb_model(3,3,self.lat,self.red_cen)

        # remember that this model was computed from w90
        tb._assume_position_operator_diagonal=False

        # add onsite energies
        onsite=np.zeros(self.num_wan,dtype=float)
        for i in range(self.num_wan):
            tmp_ham=self.ham_r[(0,0,0)]["h"][i,i]/float(self.ham_r[(0,0,0)]["deg"])
            onsite[i]=tmp_ham.real
            if np.abs(tmp_ham.imag)>1.0E-9:
                raise Exception("Onsite terms should be real!")
        tb.set_onsite(onsite-zero_energy)

        # add hopping terms
        for R in self.ham_r:
            # avoid double counting
            use_this_R=True
            # avoid onsite terms
            if R[0]==0 and R[1]==0 and R[2]==0:
                avoid_diagonal=True
            else:
                avoid_diagonal=False
                # avoid taking both R and -R
                if R[0]!=0:
                    if R[0]<0:
                        use_this_R=False
                else:
                    if R[1]!=0:
                        if R[1]<0:
                            use_this_R=False
                    else:
                        if R[2]<0:
                            use_this_R=False
            # get R vector
            vecR=_red_to_cart((self.lat[0],self.lat[1],self.lat[2]),[R])[0]
            # scan through unique R
            if use_this_R==True:
                for i in range(self.num_wan):
                    vec_i=self.xyz_cen[i]
                    for j in range(self.num_wan):
                        vec_j=self.xyz_cen[j]
                        # get distance between orbitals
                        dist_ijR=np.sqrt(np.dot(-vec_i+vec_j+vecR,
                                                -vec_i+vec_j+vecR))
                        # to prevent double counting
                        if not (avoid_diagonal==True and j<=i):
                            
                            # only if distance between orbitals is small enough
                            if max_distance is not None:
                                if dist_ijR>max_distance:
                                    continue

                            # divide the matrix element from w90 with the degeneracy
                            tmp_ham=self.ham_r[R]["h"][i,j]/float(self.ham_r[R]["deg"])

                            # only if big enough matrix element
                            if min_hopping_norm is not None:
                                if np.abs(tmp_ham)<min_hopping_norm:
                                    continue

                            # remove imaginary part if needed
                            if ignorable_imaginary_part is not None:
                                if np.abs(tmp_ham.imag)<ignorable_imaginary_part:
                                    tmp_ham=tmp_ham.real+0.0j

                            # set the hopping term
                            tb.set_hop(tmp_ham,i,j,list(R))

        return tb

