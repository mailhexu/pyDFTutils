#!/usr/bin/env python
from read_ham import *
from xyz_read import *
import sys
import os.path
import numpy as np
from collections import OrderedDict

def read_basis(fname='basis.txt'):
    basis=[]
    for line in open(fname):
        basis.append(line.strip().split()[:2])
    return basis

class Wannier_interface():
    """
    utils for wannier90 input and output
    """
    def __init__(self,name='wannier90'):
        """
        wannier90.x input and output utils.

        :param name: the prefix of the files.
        """
        self.name=name
        self.atoms=None
        self.projections=None
        self.basis=None
        self.wannier_centers=None

    def read_win(self,filename=None,basis_file='basis.txt',poscar='POSCAR'):
        """
        read the .win file.
        """
        if filename is None:
            filename=self.name+'.win'
        atoms,projections=read_win(filename,poscar='POSCAR')
        if projections:
            self.projections=True
        if not os.path.exists(basis_file):
            self.basis=projections_to_basis(atoms,projections)
        else:
            self.basis=read_basis(basis_file)
        self.atoms=atoms


    def get_basis(self):
        """
        return the basis. which is a list of (symbol_number,orb), eg. [('Mn1','dxy'), ('O','px')]
        """
        return self.basis

    def remove_basis(self,basis_name_list):
        """
        remove basis from the basis_name_list.
        eg. wa_interface.remove_basis([('Mn1','dxy'),('Mn1','dx2')])
        """
        ids=[i for (i,bas) in enumerate(self.basis) if bas in basis_name_list]
        keep_ids=[i for (i,bas) in enumerate(self.basis) if bas not in basis_name_list]

        #id translation dict
        tdict=dict(zip(keep_ids,range(len(keep_ids))))

        ham_keys=self.hamiltonian.keys()
        for key in ham_keys:
            if key[0] in ids or key[1] in ids:
                del self.hamiltonian[key]
        self.basis=[b for b in self.basis if b not in basis_name_list]
        self.wannier_centers=[self.wannier_centers[i] for i in keep_ids]

        # reset the id of each basis.
        new_hamiltonian={}
        for key,val in self.hamiltonian.items():
            new_hamiltonian[(tdict[key[0]],tdict[key[1]],key[2])]=val
        self.hamiltonian=new_hamiltonian

    def remove_sites(self,site_list):
        """
        remove all basis on the site in the site_list
        eg. wa_interface.remove_site(['Mn1','Mn2'])
        """
        ids=[i for (i,bas) in enumerate(self.basis) if bas[0] in site_list]
        keep_ids=[i for (i,bas) in enumerate(self.basis) if bas[0] not in site_list]


        #id translation dict
        tdict=dict(zip(keep_ids,range(len(keep_ids))))

        ham_keys=self.hamiltonian.keys()
        for key in ham_keys:
            if key[0] in ids or key[1] in ids:
                if key[2]==(0,0,0):
                    print "del ",key
                del self.hamiltonian[key]
        self.basis=[b for b in self.basis if b[0] not in site_list]
        self.wannier_centers=[self.wannier_centers[i] for i in keep_ids]

        # reset the id of each basis.
        new_hamiltonian={}
        for key,val in self.hamiltonian.items():
            new_hamiltonian[(tdict[key[0]],tdict[key[1]],key[2])]=val
        self.hamiltonian=new_hamiltonian

    def print_basis_list(self,filename='basis_out.txt'):
        """
        output the basis into a txt file.
        """
        with open(filename,'w') as myfile:
            for i,bas in enumerate(self.basis):
                myfile.write("%s %s %s\n"%(bas[0],bas[1],i))

    def read_centers(self,filename=None):
        """
        read the centres.xyz file.
        """
        if filename is None:
            filename = self.name+ '_centres.xyz'
        wannier_centers,atoms=read_centers(filename=filename)
        self.wannier_centers=wannier_centers
        print("%s wannier centers read."%(len(self.wannier_centers)))

    def check_center(self,max_distance=0.5):
        """
        check whether the wannier_centers are near the projection basis
        from the .win file.
        """
        return check_center(self.wannier_centers,self.basis,self.atoms,max_distance=max_distance)


    def read_hamiltonian(self, filename=None, cutoff=0.3,sort=True):
        """
        read the hamiltonian. The hamiltonian is a dict. H[m,n,R], m, n are id of the wannier functions.
        """
        if filename is None:
            filename=self.name+'_hr.dat'
        H_mnR=ham_phaser(filename=filename,cutoff=cutoff)
        if sort:
            H=sorted(H_mnR.items(),key=lambda p: abs(p[1]), reverse=True )
            H_mnR=OrderedDict(H)

        self.hamiltonian=H_mnR


    def output_hamiltonian(self,filename='hamiltonian.txt',print_out=True):
        """
        output the hamiltonian
        """
        print self.basis
        with open(filename,'w') as outfile:
            for key in self.hamiltonian:
                m,n,R=key
                if print_out:
                    print("%s -- %s R=%s : %s"%(self.basis[m],self.basis[n],R, self.hamiltonian[key]))
                outfile.write("%s -- %s R=%s : %s\n"%(self.basis[m],self.basis[n],R, self.hamiltonian[key]))

    def output_diag_hamiltonian(self,filename='diag.txt',print_out=True):
        """
        output the hamiltonian diag part.
        """
        with open(filename,'w') as outfile:
            for key in self.hamiltonian:
                m,n,R=key
                if m==n and norm(R)<0.01:
                    if print_out:
                        #print R
                        print("%s %s"%(self.basis[m], abs(self.hamiltonian[key])))
                    outfile.write("%s %s\n"%(self.basis[m], abs(self.hamiltonian[key])))


def read_eig(filename='wannier90.eig'):
    """
    read the .eig file. The result is a 2d array. E[n,k]. n is the band index and k is the kpoint index.
    """
    with open(filename,'r') as myfile:
        lines=myfile.readlines()
    lastline=lines[-1]
    nn,nk= map(int, lastline.strip().split()[:-1])
    eigs=np.zeros((nn,nk))
    for line in lines:
        words=line.strip().split()
        n=int(words[0])
        k=int(words[1])
        print n,k,float(words[2])
        eigs[n-1,k-1]=float(words[2])
    return eigs


def eig_check(eigs,emin,emax):
    eigs0=eigs[:,0]
    print eigs0
    return len(filter(lambda x: emin<x and x<emax, eigs0))



def wannier_read(cutoff=0.005):
    name=None
    if len(sys.argv)==2:
        name=sys.argv[1]
    else:
        if os.path.exists('wannier90.win'):
            name='wannier90'
        elif os.path.exists('wannier90.up.win'):
            name='wannier90.up'
        elif os.path.exists('wannier90.dn.win'):
            name='wannier90.dn'
        if name is not None:
            print("Warning: no name specified. Guess the name is %s"%name)
        else:
            print("Error: no name specified. Nor any guess is OK. Please specify a name.")
            exit()
    wint=Wannier_interface(name=name)
    wint.read_win()
    wint.read_centers()
    wint.check_center()
    wint.print_basis_list()
    wint.read_hamiltonian(cutoff=cutoff)
    wint.output_hamiltonian()
    wint.output_diag_hamiltonian()

if __name__ == '__main__':
    #eigs=read_eig('wannier90.dn.eig')
    #print eig_check(eigs,-1,9)
    wannier_read()
    print read_basis()
