#!/usr/bin/env python
from ase.io import read
from ase.calculators.vasp import VaspDos
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from shutil import copyfile
from scipy import trapz,integrate
from pyDFTutils.ase_utils.symbol import symnum_to_sym
from .vasp_utils import get_symdict
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict,Iterable
import copy

orb_eg=['dx2+','dz2+','dx2-','dz2-']
orb_t2g=['dxy+','dyz+','dxz+','dxy-','dyz-','dxz-']
orb_p=['px+','py+','pz+','px-','py-','pz-']

orb_eg_label=['$d_{x^2+y^2}$+','$d_{z^2}$+','$d_{x^2+y^2}$-','$d_{z^2}$-']
orb_t2g_label=['$d_{xy}$+','$d_{yz}$+','$d_{xz}$+','$d_{xy}$-','$d_{yz}$-','$d_{xz}$-']
orb_p_label=['$p_x$+','$p_y$+','$p_z$+','$p_x$-','$p_y$-','$p_z$-']

orb_all_local=['s+','p+','d+','s-','p-','d-']
orb_all_local_label=['s+','p+','d+','s-','p-','d-']
 

def getline(file,line_num):
    num_l=0
    for line in file:
        num_l +=1
        if num_l==line_num:
            return line

def is_spin_dos(filename='DOSCAR'):
    dos_text=open(filename)
    line7=getline(dos_text,7)
    nn7=len([fstr for fstr in line7.split()])
    dos_text.close()
    if nn7==3:
        return False
    elif nn7==5:
        return True
    else:
        print("Can't Phase file ./DOSCAR")


def get_dos_CBM_VBM(e,d,efermi=0.0,thr=0.0001,interpolate=True):
    """
    calculate the CBM and VBM according to the density of states.
    Parameters: e: array_like energy
                d: array_like DOS
                efermi: the fermi energy, default=0
                thr: threshold. default=0.0001.
    Returns: CBM,VBM
    """
    if interpolate:
        from scipy.interpolate import interp1d
        f=interp1d(e,d,kind='cubic')
        energy=np.linspace(e[0],e[-1],4000)
        dos=f(energy)
    else:
        energy=e[:]
        dos=d[:]

    idos=integrate.cumtrapz(dos,energy,initial=0)
    N=len(energy)
    CBM=efermi
    VBM=efermi
    for i,e in enumerate(energy):
        if e>=efermi:
            ifermi=i
            break
    print(ifermi)
    idos=idos-idos[ifermi]
    #plt.plot(idos)
    #plt.show()
    for i in range(ifermi,N):
        if idos[i]>thr:
            CBM=energy[i]
            break
    for i in range(ifermi,0,-1):
        if idos[i]<-thr:
            VBM=energy[i]
            break
    return CBM,VBM



def get_total_dos(filename='DOSCAR',nspin=2,efermi=None):
    """
    Get the DOS from DOSCAR. if is spin polarized and nspin

    Args:

     filename: path to DOSCAR

     nspin: 1 | 2.
    """

    is_spin=is_spin_dos(filename=filename)
    dos_text=open(filename)
    line6 = getline(dos_text,6)
    emax,emin,nbands,ef,x= [float(fstr) for fstr in line6.split()]
    if efermi is not None:
         ef=efermi
    efermi=ef
    e_array=[]
    dos_up_array=[]
    dos_down_array=[]
    dos_array=[]
    idos_array=[]
    line_num=6
    for line in dos_text:
        line_num +=1
        if line_num >= nbands+6:
            break
        if is_spin:
            e,dos_up,dos_down,idos_up,idos_down=[float(fstr) for fstr in line.split()]
            e_norm=e-efermi
            if nspin==2:  #e_norm< epmax and e_norm > epmin:
                e_array.append(e_norm)
                dos_up_array.append(dos_up)
                dos_down_array.append(-dos_down)

            elif nspin==1:
                e_array.append(e_norm)
                dos_array.append(dos_up+dos_down)
        else:
            e,dos,idos=[float(fstr) for fstr in line.split()]
            e_norm=e-efermi
            if True:  #e_norm< epmax and e_norm > epmin:
                e_array.append(e_norm)
                dos_array.append(dos)
                idos_array.append(idos)
    if nspin==2:
        return e_array,dos_up_array,dos_down_array
    elif nspin==1:
        return e_array,dos_array
    else:
        raise ValueError('nspin should be 1 or 2')


def plot_dos(filename='DOSCAR',epmin=-20,epmax=5.0,output=None):
    """
    plot the total dos into a file.

    Args:

     filename

     epmin,epmax: Energy min/max in eV, relative to the E_fermi
    """
    if is_spin_dos():
        plot_dos_spin(filename=filename,epmin=epmin,epmax=epmax,output=output)
    else:
        plot_dos_nospin(filename=filename,epmin=epmin,epmax=epmax,output=output)

def plot_dos_nospin(filename='DOSCAR',epmin=-20,epmax=5.0,output=None):
    dos_text=open(filename)
    line6 = getline(dos_text,6)
    emax,emin,nbands,efermi,x= [float(fstr) for fstr in line6.split()]
    e_array=[]
    dos_array=[]
    idos_array=[]
    line_num=6
    for line in dos_text:
        line_num +=1
        if line_num >= nbands+6:
            break
        e,dos,idos=[float(fstr) for fstr in line.split()]
        e_norm=e-efermi
        if e_norm< epmax and e_norm > epmin:
            e_array.append(e_norm)
            dos_array.append(dos)
            idos_array.append(idos)

    fig=plt.figure()
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    tspin_splt=fig.add_subplot(111)
    tspin_splt.axvline(0,color='black')
    tspin_splt.grid(True)
    plt.plot(e_array,dos_array)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    plt.close()

def plot_dos_spin(filename='DOSCAR',epmin=-20.0,epmax=5.0,output=None):
    """
    plot total dos with spin up | down
    """
    dos_text=open(filename)
    line6 = getline(dos_text,6)
    emax,emin,nbands,efermi,x= [float(fstr) for fstr in line6.split()]
    e_array=[]
    dos_up_array=[]
    dos_down_array=[]
    line_num=6
    for line in dos_text:
        line_num +=1
        if line_num >= nbands+6:
            break
        e,dos_up,dos_down,idos_up,idos_down=[float(fstr) for fstr in line.split()]
        e_norm=e-efermi
        if e_norm< epmax and e_norm > epmin:
            e_array.append(e_norm)
            dos_up_array.append(dos_up)
            dos_down_array.append(-dos_down)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig=plt.figure()
    tspin_splt=fig.add_subplot(111)
    tspin_splt.axvline(0,color='black')
    tspin_splt.grid(True)
    plt.plot(e_array,dos_up_array,label="Spin up")
    plt.plot(e_array,dos_down_array,label="Spin down")
    plt.legend()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    plt.close()

def write_all_sum_dos(output='sum_dos.txt',nospin_output='sum_dos_nospin.txt',eg_polar_output='eg_polor.txt',erange=None):
    mydos=MyVaspDos()
    orbs=mydos.get_orbital_names()
    symdict=get_symdict()
    with open(output,'w') as myfile:
        myfile.write('symnum '+'\t'.join([x.ljust(5) for x in orbs])+'\n')
        for symnum in symdict:
            sdoss=list(mydos.sum_dos_all(symdict[symnum],erange=erange).values())
            myfile.write('%s\t%s\n'%(symnum.ljust(6),'\t'.join(["%.3f"%s for s in sdoss])))

    orbs=mydos.get_orbital_names_nospin()
    with open(nospin_output,'w') as myfile:
        myfile.write('symnum '+'\t'.join([x.ljust(5) for x in orbs])+'\n')
        for symnum in symdict:
            sdoss=list(mydos.sum_dos_all_nospin(symdict[symnum],erange=erange).values())
            myfile.write('%s\t%s\n'%(symnum.ljust(6),'\t'.join(["%.3f"%s for s in sdoss])))

    try:
        if eg_polar_output is not None:
            with open(eg_polar_output,'w') as myfile:
                myfile.write('symnum '+'\t'.join(['dx2+','dx2-','dz2+','dz2-','dx2','dz2','rate'])+'\n')
                for symnum in symdict:
                    dx2_up=mydos.sum_dos(symdict[symnum],'dx2+',erange=erange)
                    dx2_down=mydos.sum_dos(symdict[symnum],'dx2-',erange=erange)
                    dz2_up=mydos.sum_dos(symdict[symnum],'dz2+',erange=erange)
                    dz2_down=mydos.sum_dos(symdict[symnum],'dz2-',erange=erange)

                    dx2=mydos.sum_dos_nospin(symdict[symnum],'dx2',erange=erange)
                    dz2=mydos.sum_dos_nospin(symdict[symnum],'dz2',erange=erange)
                    if dx2+dz2<0.0001:  #no eg
                        porlar_rate=0
                    else:
                        porlar_rate=(dx2-dz2)/(dx2+dz2)

                    myfile.write('%s\t%s\n'%(symnum.ljust(6),'\t'.join(["%.3f"%s for s in [dx2_up,dx2_down,dz2_up,dz2_down,dx2,dz2,porlar_rate]])))
    except Exception as exc:
        print(("Warning: This is not a good DOSCAR for calculating eg polarization. \n %s"%exc))

class MyVaspDos(VaspDos):
    """
    Inherited from VaspDos. Add a few function, and bugfix.
    """
    def __init__(self,doscar='DOSCAR',efermi='auto'):
        """
        Args:
         doscar: the 'DOSCAR' file.
         efermi: 'auto' | other. If auto, read efermi from file.
        """
        super(MyVaspDos,self).__init__(doscar=doscar,efermi=0.0)
        self.doscar=doscar
        if efermi=='auto':
            self.auto_set_fermi()

        dos_text=open(doscar,'r').readlines()
        line7=dos_text[7]
        nn7=len([fstr for fstr in line7.split()])
        if nn7==3:
            self.isspin=False
        elif nn7==5:
            self.isspin=True
        else:
            print("Can't decide if the DOS is spin polarized")


    def sum_dos(self,atom,orbital,erange=None):
        """
        sum of dos on orbital of atom. If erange is None, sum the occupied dos.
        Args:
          atom: index of atom
          orbital: orbital eg. 's+','dx-'
        """
        e=self.get_energy()
        dos=self.site_dos(atom,orbital)
        if erange is None:
            f=np.array([1.0 if x<0 else 0 for x in e])
        else:
            emin,emax=erange
            f=np.array([1.0 if x<emax and x>emin  else 0 for x in e])
        sum_dos=trapz(np.asarray(dos)*f,e)
        return sum_dos

    def sum_dos_all(self,atom,erange=None):
        """
        return sumdos of each orbitals of atom. The result is a ordered dict.
        Note the spin components are seperated.
        """
        sorb=OrderedDict()
        for orb in self.get_orbital_names():
            sorb[orb]=self.sum_dos(atom,orb,erange=erange)
        return sorb

    def sum_dos_all_nospin(self,atom,erange=None):
        """
        return sumdos of each orbitals of atom. The result is a ordered dict.
        Note the spin components are added up.
        """
        sorb=OrderedDict()
        for orb in self.get_orbital_names_nospin():
            sorb[orb]=self.sum_dos_nospin(atom,orb,erange=erange)
        return sorb


    def get_orbital_names_nospin(self):
        """
        orbital names, the spin signature is removed.eg. px+ px- -> px
        return a list .
        """
        if self.isspin:
            names=self.get_orbital_names()
            names_nospin=[]
            for n in names:
                if n.endswith(('+','-')):
                    if n[:-1] not in names_nospin:
                        names_nospin.append(n[:-1])
            return names_nospin
        else:
            return self.get_orbital_names()


    def sum_dos_nospin(self,atom,orb,erange=None):
        """
        same as sum_dos but add up two spin component.
        """
        if self.isspin:
            return self.sum_dos(atom,orb+'+',erange=erange)+self.sum_dos(atom,orb+'-',erange=erange)
        else:
            return self.sum_dos(atom,orb,erange=erange)


    def get_orbital_dict(self):
        n = self._site_dos.shape[1]
        if n == 4:
            norb = {'s':1, 'p':2, 'd':3}
        elif n==5:
            norb = {'s':1, 'p':2,'d':3 ,'f':4}
        elif n == 7:
            norb = {'s+':1, 's-up':1, 's-':2, 's-down':2,
                    'p+':3, 'p-up':3, 'p-':4, 'p-down':4,
                    'd+':5, 'd-up':5, 'd-':6, 'd-down':6}
        elif n == 9:
            norb = {'s+':1, 's-up':1, 's-':2, 's-down':2,
                    'p+':3, 'p-up':3, 'p-':4, 'p-down':4,
                    'd+':5, 'd-up':5, 'd-':6, 'd-down':6,
                    'f+':7, 'f-up':7, 'f-':8, 'f-down':8
                    }
        elif n == 16:
            norb = {'s+':1, 's-up':1, 's-':2, 's-down':2,
                    'p+':3, 'p-up':3, 'p-':4, 'p-down':4,
                    'd+':5, 'd-up':5, 'd-':6, 'd-down':6,
                    'f+':7, 'f-up':7, 'f-':8, 'f-down':8
                    }

        elif n == 10:
            norb = {'s':1, 'py':2, 'pz':3, 'px':4,
                    'dxy':5, 'dyz':6, 'dz2':7, 'dxz':8,
                    'dx2':9}
        elif n == 19:
            norb = {'s+':1, 's-up':1, 's-':2, 's-down':2,
                    'py+':3, 'py-up':3, 'py-':4, 'py-down':4,
                    'pz+':5, 'pz-up':5, 'pz-':6, 'pz-down':6,
                    'px+':7, 'px-up':7, 'px-':8, 'px-down':8,
                    'dxy+':9, 'dxy-up':9, 'dxy-':10, 'dxy-down':10,
                    'dyz+':11, 'dyz-up':11, 'dyz-':12, 'dyz-down':12,
                    'dz2+':13, 'dz2-up':13, 'dz2-':14, 'dz2-down':14,
                    'dxz+':15, 'dxz-up':15, 'dxz-':16, 'dxz-down':16,
                    'dx2+':17, 'dx2-up':17, 'dx2-':18, 'dx2-down':18}
        else:
            norb = {'s+':1, 's-up':1, 's-':2, 's-down':2,
                    'py+':3, 'py-up':3, 'py-':4, 'py-down':4,
                    'pz+':5, 'pz-up':5, 'pz-':6, 'pz-down':6,
                    'px+':7, 'px-up':7, 'px-':8, 'px-down':8,
                    'dxy+':9, 'dxy-up':9, 'dxy-':10, 'dxy-down':10,
                    'dyz+':11, 'dyz-up':11, 'dyz-':12, 'dyz-down':12,
                    'dz2+':13, 'dz2-up':13, 'dz2-':14, 'dz2-down':14,
                    'dxz+':15, 'dxz-up':15, 'dxz-':16, 'dxz-down':16,
                    'dx2+':17, 'dx2-up':17, 'dx2-':18, 'dx2-down':18}
        o_norb=OrderedDict()
        for orb in sorted(norb,key=norb.get):
            o_norb[orb]=norb[orb]
        return o_norb

    def get_orbital_names(self):
        norb=self.get_orbital_dict()
        orb_names=[x for x in norb if not(x.endswith('up') or x.endswith('down'))]
        return orb_names


    def site_dos(self, atom, orbital):
        """
        A modified version.
        Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        # Integer indexing for orbitals starts from 1 in the _site_dos array
        # since the 0th column contains the energies
        norb=self.get_orbital_dict()
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]

        return self._site_dos[atom, norb[orbital.lower()], :]


    def auto_set_fermi(self):
        """
        Read the Efermi from DOSCAR and set the efermi value to it
        """
        line6 = open(self.doscar,'r').readlines()[5]
        emax,emin,nbands,efermi,x= [float(fstr) for fstr in line6.split()]
        self._set_efermi(efermi)
        print(efermi)

    def get_efermi(self):
        """
        Return the efermi
        """
        line6 = open(self.doscar,'r').readlines()[5]
        emax,emin,nbands,efermi,x= [float(fstr) for fstr in line6.split()]
        #return self._get_efermi()
        return efermi

    def get_energy(self):
        """
        return the energy in a list.
        """
        return self.energy

    def get_site_dos(self):
        """
        eg. Use VaspDos.site_dos(iatom,'s+') might be a better choice.
        """
        return self._site_dos

    def get_major_minor_dos(self,iatom,anglars=['s','p','d']):
        """
        return the DOS_major, DOS_minor.
        Args:
         iatom: index of atom or symnum.
         anglars: orbital names without up/down. eg. ['s','p'] or [dxy,dxz]
        """
        up_angs=[a+'+' for a in anglars]
        down_angs=[a+'-' for a in anglars]
        up_dos=[self.site_dos(iatom,s) for s in up_angs]

        up_dos_sum=[]
        for di in up_dos:
            x=np.array([self.energy,di])
            e=x[0][x[0]<0]
            d=x[1][x[0]<0]
            up_dos_sum.append(trapz(d,x=e))
        self.up_tot_dos=sum(up_dos_sum)


        down_dos=[self.site_dos(iatom,s) for s in down_angs]
        down_dos_sum=[]
        for di in down_dos:
            x=np.array([self.energy,di])
            e=x[0][x[0]<0]
            d=x[1][x[0]<0]
            down_dos_sum.append(trapz(d,x=e))
        self.down_tot_dos=sum(down_dos_sum)

        print(('up: %s, down: %s'%(self.up_tot_dos,self.down_tot_dos)))
        if self.up_tot_dos > self.down_tot_dos:
            return up_dos,down_dos
        else:
            return down_dos,up_dos


    def is_up_major(self,iatom,anglars=['s','p','d']):
        up_angs=[a+'+' for a in anglars]
        down_angs=[a+'-' for a in anglars]
        up_dos=[self.site_dos(iatom,s) for s in up_angs]

        up_dos_sum=[]
        for di in up_dos:
            x=np.array([self.energy,di])
            e=x[0][x[0]<0]
            d=x[1][x[0]<0]
            up_dos_sum.append(trapz(d,x=e))
        self.up_tot_dos=sum(up_dos_sum)


        down_dos=[self.site_dos(iatom,s) for s in down_angs]
        down_dos_sum=[]
        for di in down_dos:
            x=np.array([self.energy,di])
            e=x[0][x[0]<0]
            d=x[1][x[0]<0]
            down_dos_sum.append(trapz(d,x=e))
        self.down_tot_dos=sum(down_dos_sum)
        if self.up_tot_dos < self.down_tot_dos:
            print('Warning: up down reversed so that up is major ')

        return self.up_tot_dos > self.down_tot_dos

    def get_site_dos_major_up(self,iatom,site):
        mup=self.is_up_major(iatom)
        if not mup:
            if site.endswith('-'):
                nsite=site
                nsite=nsite.replace('-','+')

            elif site.endswith('+'):
                nsite=site
                nsite=nsite.replace('+','-')
            return self.site_dos(iatom,nsite)
        else:
            return self.site_dos(iatom,site)

    def get_total_dos(self):
        return self.dos


def getldos(iatom,sites,location='./',doscar='DOSCAR',efermi='auto',is_up_major=False ,return_efermi=False):
    """
    read DOSCAR in location . return PDOS of iatom ,site is a list of 's+','s-',etc.
    iatom can be also sym_num, in this case there should be a POSCAR in the specified location.
    """
    if isinstance(iatom,str):
        try:
            symdict=get_symdict(filename=os.path.join(location,'POSCAR'))
            iatom=symdict[iatom]
        except Exception as exc:
            print(exc)
            raise ValueError('iatom should be int. Or a proper symnum with a POSCAR should be supplied')

    mydos=MyVaspDos(doscar=join(location,doscar),efermi=efermi)
    energy=mydos.get_energy()
    efermi=mydos.get_efermi()
    dos=[]
    for s in sites:
        if s.endswith('-'):
            if is_up_major:
                dos.append(-mydos.get_site_dos_major_up(iatom,s))
            else:
                dos.append(-mydos.site_dos(iatom,s))
        else:
            if is_up_major:
                dos.append(mydos.get_site_dos_major_up(iatom,s))
            else:
                dos.append(mydos.site_dos(iatom,s))
    if return_efermi:
        return energy, dos, efermi
    else:
        return energy,dos

def read_sumdos(filename='sum_dos.csv'):
    """
    read informations in sum_dos.csv.
    returns a dict {(symnum,orbital):dos}
    """
    text=open(filename,'r').readlines()

    orbs=text[0].strip().split()[1:]
    symnums=[t.strip().split()[0] for t in text[1:]]

    ncol=len(orbs)+1
    a=np.loadtxt(fname=filename,dtype=float,skiprows=1,usecols=list(range(1,ncol)))
    dos_dict=OrderedDict()
    for i,sym_num in enumerate(symnums):
        for j,orb in enumerate(orbs):
            dos_dict[(sym_num,orb)]=a[i,j]

    return  symnums, orbs ,dos_dict






def plot_group(xs,yss,labels,ymin=0,ymax=2,xmin=-15,xmax=5):
    """
    plot a group of x-y's in one figure.
    """

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.figure()
    axes=[]
    plt.subplots_adjust(hspace=0.001)
    for i,x in enumerate(xs):
        #subplot_number=int(str(len(xs))+'1'+str(i+1))
        nplot=len(xs)
        if i==0:
            axes.append(plt.subplot(nplot,1,i+1))
        else:
            axes.append(plt.subplot(nplot,1,i+1,sharex=axes[0]))
        ys=yss[i]
        label=labels[i]
        for y in ys:
            plt.xlabel('Energy (eV)')
            plt.ylabel('%s'%label)
            axes[i].plot(x,y)
            axes[i]
            plt.xlim(xmin,xmax)
            try:
                plt.ylim(ymin[i],ymax[i])
                plt.yticks(np.arange(ymin[i],ymax[i],(ymax[i]-ymin[i])/3.0))
            except Exception:
                plt.ylim(ymin,ymax)
                plt.yticks(np.arange(ymin,ymax,(ymax-ymin)/3.0))
    plt.legend()
    plt.show()

def plot_all_ldos(filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0,element_types=None, has_f=False):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()

    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    else:
        atom_nums=list(symdict.keys())

    if ispin==2:
        if has_f:
            sites=['s+','p+','d+','s-','p-','d-', 'f+', 'f-']
        else:
            sites=['s+','p+','d+','s-','p-','d-']
    else:
        if has_f:
            sites=['s','p','d','f']
        else:
            sites=['s','p','d']

    if not os.path.exists('LDOS'):
        os.mkdir('LDOS')
    for atom_num in atom_nums:
        copyfile(filename,'LDOS/DOSCAR')
        plotldos_group([atom_num],sites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output='LDOS/%s_ldos.png'%atom_num)


def plot_pdos(sites,lsites=None,element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0,output_dir='PDOS'):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()
    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    else:
        atom_nums=list(symdict.keys())
    if sites=='eg':
        if ispin==2:
            lsites=['dx2+','dz2+','dx2-','dz2-']
        else:
            lsites=['dx2','dz2']
    elif sites=='t2g':
        if ispin==2:
            lsites=['dxy+','dyz+','dxz+','dxy-','dyz-','dxz-']
        else:
            lsites=['dxy','dyz','dxz']
    elif sites=='p':
        if ispin==2:
            lsites=['px+','py+','pz+','px-','py-','pz-']
        else:
            lsites=['px','py','pz']
    elif sites=='s':
        if ispin==2:
            lsites=['s+','s-']
        else:
            lsites=['s']
    else:
        lsites=lsites

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for atom_num in atom_nums:
        copyfile(filename,join(output_dir,'DOSCAR'))
        plotldos_group([atom_num],lsites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output=join(output_dir, '%s_%s_dos.png'%(atom_num,sites)))




def plot_all_pdos_eg(element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()
    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    if ispin==2:
        sites=['dx2+','dz2+','dx2-','dz2-']
    else:
        sites=['dx2','dz2']

    if not os.path.exists('PDOS'):
        os.mkdir('PDOS')
    for atom_num in atom_nums:
        copyfile(filename,'PDOS/DOSCAR')
        plotldos_group([atom_num],sites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output='PDOS/%s_eg_dos.png'%atom_num)

def get_xld_d(iatom):
    inplane=['dxy+', 'dx2+', 'dxy-', 'dx2-']
    outplane=['dyz+', 'dxz+', 'dz2+', 'dyz-', 'dxz-', 'dz2-']





def plot_all_pdos_t2g(element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()
    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    if ispin==2:
        sites=['dxy+','dyz+','dxz+','dxy-','dyz-','dxz-']
    else:
        sites=['dxy','dyz','dxz']

    if not os.path.exists('PDOS'):
        os.mkdir('PDOS')
    for atom_num in atom_nums:
        copyfile(filename,'PDOS/DOSCAR')
        plotldos_group([atom_num],sites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output='PDOS/%s_t2g_dos.png'%atom_num)

def plot_all_pdos_p(element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()
    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    if ispin==2:
        sites=['px+','py+','pz+','px-','py-','pz-']
    else:
        sites=['px','py','pz']

    if not os.path.exists('PDOS'):
        os.mkdir('PDOS')
    for atom_num in atom_nums:
        copyfile(filename,'PDOS/DOSCAR')
        plotldos_group([atom_num],sites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output='PDOS/%s_p_dos.png'%atom_num)

def plot_all_pdos_s(element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0):
    """
    plot the local dos of all atoms.
    """
    symdict=get_symdict()
    if element_types is not None:
        atom_nums=[x for x in list(symdict.keys()) if symnum_to_sym(x) in element_types]
    if ispin==2:
        sites=['s+','s-']
    else:
        sites=['s']

    if not os.path.exists('PDOS'):
        os.mkdir('PDOS')
    for atom_num in atom_nums:
        copyfile(filename,'PDOS/DOSCAR')
        plotldos_group([atom_num],sites,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,special_location=None,output='PDOS/%s_s_dos.png'%atom_num)


def plot_all_pdos(element_types=None,filename='DOSCAR',ispin=2,ymin=-2.0,ymax=2.0,xmin=-15.0,xmax=5.0,output_dir='PDOS',orbs=['s','p','eg','t2g']):
    for site in orbs:
        plot_pdos(sites=site,element_types=element_types,filename=filename,ispin=ispin,ymin=ymin,ymax=ymax,xmin=xmin,xmax=xmax,output_dir=output_dir)


class single_dos_plot():
    def __init__(self,xrange=[-15.0,5.0],yrange=[-2.0,2.0],shift=0.0):
        self.xrange=copy.copy(xrange)
        self.yrange=copy.copy(yrange)
        self.dos_info=[]
        self.doses=[]
        self.shift=shift
    def set_xrange(self,xrange):
        self.xrange=xrange
    def set_yrange(self,yrange):
        self.yrange=yrange
    def add_dos(self,sym_num,orbital, path='.',filename='DOSCAR',label=None):

        if isinstance(orbital,Iterable) and not isinstance(orbital,str):
            if isinstance(label,str) or label is None:
                label=[copy.copy(label)]*len(orbital)
            for orb,lab in zip(orbital,label):
                if lab is None:
                    lab="%s(%s)"%(sym_num,orb)
                self.dos_info.append([sym_num,orb,path,filename,lab])
        else:
            if label is None:
                label="%s(%s)"%(sym_num,orbital)
            self.dos_info.append([sym_num,orbital,path,filename,label])
    def get_dos_info(self):
        return self.dos_info
    def get_doses(self):
        for info in self.get_dos_info():
            sym_num,orbital,path,filename,label=info
            energy,doses=getldos(sym_num,[orbital],location=path,doscar=filename,efermi='auto',is_up_major=False)
            self.doses.append([energy,doses[0],label])
        return self.doses




class dos_plot():
    def __init__(self,xrange=[-15,0,5.0]):
        self.xrange=xrange
        self.sub_dos=[]
    def set_xrange(self,xrange):
        self.xrange=xrange
    def add_subdos(self,yrange=[-2.0,2.0],shift=0.0):
        sdos=single_dos_plot(xrange=self.xrange,yrange=yrange,shift=shift)
        self.sub_dos.append(sdos)
        return sdos
    def plot(self,output=None):
        plt.figure()
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes=[]
        plt.subplots_adjust(hspace=0.001)

        for i,ss in enumerate(self.sub_dos):
            doses=ss.get_doses()
            subplot_number=int(str(len(self.sub_dos))+'1'+str(i+1))
            if i==0:
                axes.append(plt.subplot(subplot_number))
            else:
                axes.append(plt.subplot(subplot_number,sharex=axes[0]))
            for dos in doses:
                e,d,label=dos
                plt.xlabel('E-E$_F$ (eV)')
                plt.ylabel('DOS')
                axes[i].plot(np.array(e)-ss.shift,d,label=label)
                plt.xlim(self.xrange[0],self.xrange[1])
                #plt.xlim(ss.xrange[0],ss.xrange[1])
                try:
                    ymin,ymax=ss.yrange
                    plt.ylim(ymin,ymax)
                    plt.yticks(np.arange(ymin,ymax,(ymax-ymin)/3.0))
                except Exception:
                    plt.ylim(ymin,ymax)
                    plt.yticks(np.arange(ymin,ymax,(ymax-ymin)/3.0))
                plt.axvline(x=0-ss.shift,color='black',linestyle='--')
                plt.legend(loc=3)
        if output is not None:
            plt.savefig(output)
        else:
            plt.show()
        plt.close()


def plotldos_group(atom_num,sites,ymin=0.0,ymax=2.0,xmin=-15.0,xmax=5.0,special_location=None,output=None):
    """
    Usage:
      names=['Bi1','Fe1','O1','O3']
      plotldos_group(names,['s+','p+','d+','s-','p-','d-'],ymin=[-0.7,-6,-0.7,-0.7],ymax=[0.7,6,0.7,0.7])
      This will plot a figure consisits of 4 subplots.
    spectial_location: if the i_th atom is not in the present directory, a dict like this: {3:('../BFO1','bulk')}, in which bulk is the special label.
    """
    symdict=get_symdict()

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.figure()
    axes=[]
    plt.subplots_adjust(hspace=0.001)
    for i,ss in enumerate(atom_num):
        if (special_location is None) or (i not in special_location):
            energy,dos=getldos(symdict[ss],sites)
            label=ss
        else:
            symdict_sp=get_symdict(filename=join(special_location[i][0],'POSCAR'))
            energy,dos=getldos(symdict_sp[ss],sites,location=special_location[i][0])
            label=special_location[i][1]
        subplot_number=int(str(len(atom_num))+'1'+str(i+1))
        print(subplot_number)
        if i==0:
            axes.append(plt.subplot(subplot_number))
        else:
            axes.append(plt.subplot(subplot_number,sharex=axes[0]))
        for s,d in zip(sites,dos):
            plt.xlabel('Energy (eV)')
            plt.ylabel('%s'%label)
            axes[i].plot(energy,d,label=label+s)
            plt.xlim(xmin,xmax)
            try:
                plt.ylim(ymin[i],ymax[i])
                plt.yticks(np.arange(ymin[i],ymax[i],(ymax[i]-ymin[i])/3.0))
            except Exception:
                plt.ylim(ymin,ymax)
                plt.yticks(np.arange(ymin,ymax,(ymax-ymin)/3.0))
        plt.axvline(x=0,color='black',linestyle='--')
        plt.legend(loc=3)
    if output is not None:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()



def plotldos(iatom,sites,xmin=-15.0,xmax=5.0,ymin=None,ymax=None):
    mydos=MyVaspDos()
    mydos.auto_set_fermi()
    efermi=mydos.get_efermi()
    print(efermi)
    energy=mydos.get_energy()

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for s in sites:
        if s.endswith('-'):
            plt.plot(energy,-mydos.site_dos(iatom,s),label=s)
        else:
            plt.plot(energy,mydos.site_dos(iatom,s),label=s)
    plt.xlim(xmin,xmax)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin,ymax)
    plt.legend()
    plt.show()

if __name__=='__main__':
    #print get_symdict()
    #plotldos(9,['s+','s-','p+','p-','d+','d-'])
    #plotldos(8,['s+','s-','p+','p-','d+','d-'])
    #plotldos(7,['s+','s-','p+','p-','d+','d-'])
    #plotldos(6,['s+','s-','p+','p-','d+','d-'])
    #plotldos_group(['Fe1','Fe2','Fe3','Fe4'],['s+','p+','d+'])
    #plotldos_group(['Fe1','Fe2','Fe3','Fe4'],['s-','p-','d-'],ymin=-12,ymax=0)
    # plotldos_group(['Ti1','Ti2','Ti3','Ti4'],['s-','p-','d-'],ymin=-12,ymax=0)
    names=['Bi1','Fe1','O1','O3']
    #plotldos_group(names,['s+','p+','d+','s-','p-','d-'],ymin=[-0.7,-6,-0.7,-0.7],ymax=[0.7,6,0.7,0.7])
    plotldos_group(names,['s+','p+','d+','f+', 's-','p-','d-', 'f-'],ymin=[-0.7,-6,-0.7,-0.7],ymax=[0.7,6,0.7,0.7])
