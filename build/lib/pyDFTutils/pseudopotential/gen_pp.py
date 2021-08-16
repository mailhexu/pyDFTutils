#!/usr/bin/env python

from gpaw.atom.configurations import configurations
import os
import numpy as np
import matplotlib.pyplot as plt

econf_strs={
    'He':'1s',
    'Ne':'1s2s2p',
    'Ar':'1s2s2p3s3p',
    'Kr':'1s2s2p3s3p3d4s4p',
    'Xe':'1s2s2p3s3p3d4s4p4d5s5p',
    'Rn':'1s2s2p3s3p3d4s4p4d4f5s5p5d6s6p'
}

def sig_to_num(x):
    """
    s,p,d,... ->0, 1,2...
    """
    d=dict(list(zip(tuple('spdfgh'),[0,1,2,3,4,5])))
    return d[x]


def econf_str_parse(s):
    for k in econf_strs:
        s=s.replace('[%s]'%k,econf_strs[k])
    ns=[int(x) for x in s[0::2]]
    ls=[sig_to_num(x) for x in s[1::2]]
    return list(zip(ns,ls))



class element(object):
    def __init__(self,name):
        self.name=name
        self.atom_number,self.econfs=configurations[name]
        self.unbounded_states=dict()
        self.l_loc=3
        self.E_loc=0
        self.Vloc_scheme='troulliermartins'
        self.shape_function='besselshape'

    def add_empty_bands(self,n,l):
        if [x for x in self.econfs if x[0]==n and x[1]==l] == []:
            self.econfs.append((n,l,0,None))

    def add_unbounded_states(self,l,energy):
        self.unbounded_states[l]=energy

    def set_core(self,core_str):
        core_states=econf_str_parse(core_str)
        self.core_nl=core_states

    def set_vloc_scheme(self,l_loc,E_loc,Vloc_scheme='bessel'):
        self.l_loc=l_loc
        self.E_loc=E_loc
        if Vloc_scheme =='bessel':
            self.Vloc_scheme='bessel'
        elif Vloc_scheme.startswith('t'):
            self.Vloc_scheme='troulliermartins'
        elif Vloc_scheme.startswith('u'):
            self.Vloc_scheme='ultrasoft'
        else:
            raise Exception('Wrong Vloc_scheme')

    def set_rcuts(self,rcuts):
        self.rcuts=rcuts

    def get_atom_number(self):
        return self.atom_number

    def max_n_of_l(self):
        max_ns=[]
        for l in range(6):
            ns=[x[0] for x in self.econfs if x[1]==l]
            if ns==[]:
                max_n=0
            else:
                max_n=max(ns)
            max_ns.append(max_n)
        return max_ns

    def get_partial_nl(self):
        partial_bands=[]
        for econf in self.econfs:
            n,l,occ,energy=econf
            if occ< (l*2+1)*2:
                partial_bands.append((n,l,occ))
        return partial_bands

def test_elem():
    Mn=element('Mn')
    Mn.add_empty_bands(4,1)
    print(Mn.max_n_of_l())
    print(Mn.get_partial_nl())



class atompaw(element):
    def __init__(self,name):
        super(atompaw,self).__init__(name)
        self.input_filename='input.inp'
        self.xc='GGA-PBE'
        self.is_rel=True
        self.nucleus_keyword='point-nucleus'
        self.grid_keyword='loggrid'
        self.ngrid=2001
        self.projector_keyword='Bloechl'

    def set_radius(self,r_paw,r_shape=None, r_vloc=None,r_core=None):
        self.r_paw=r_paw

        if r_shape is None:
            self.r_shape=self.r_paw/1.2
        else:
            self.r_shape=r_shape

        if r_vloc is None:
            self.r_vloc=self.r_paw
        else:
            self.r_vloc=r_vloc

        if r_core is None:
            self.r_core=self.r_paw
        else:
            self.r_core=r_core

    def set_projector(self,keyword):
        if keyword in ['v','van','VNCTV','Vanderbilt']:
            self.projector_keyword='Vanderbilt'
        else:
            self.projector_keyword='Bloechl'


    def write_input_file(self):
        input_text=''
        # Atom_name Z
        input_text+='%s %d\n'%(self.name,self.atom_number)

        # XC_functional rel_keyword nucleus_keyword grid_keyword gridsize r rmax match
        input_text+=self.xc+'\t'

        if self.is_rel:
            input_text+= 'scalarrelativistic\t'
        else:
            input_text+= 'nonrelativistic\t'


        input_text+="%s\t%s\n"%(self.grid_keyword,self.ngrid)

        # n_l_max
        input_text+='\t'.join([str(x) for x in self.max_n_of_l()])+'\n'

        # partial states
        for pstate in self.get_partial_nl():
            input_text+= '\t'.join([str(x) for x in pstate])+'\n'
        input_text += '0\t0\t0\n'

        # core or valence (c or v)
        econfs_sorted=sorted(self.econfs,key=lambda x: (x[1],x[0]))
        for econf in econfs_sorted:
            if (econf[0],econf[1]) in self.core_nl:
                input_text+='c\n'
            else:
                input_text+='v\n'
        # l_max
        l_max_b=max(x[1] for x in self.econfs)
        if len(list(self.unbounded_states.keys()))!=0:
            l_max_ub=max(self.unbounded_states.keys())
        else:
            l_max_ub=0
        self.l_max=max(l_max_b,l_max_ub)
        input_text+='%d\n'%(self.l_max)

        # r_paw, r_shape , r_vloc, r_core
        input_text+= "%s\t%s\t%s\t%s\n"%(self.r_paw,self.r_shape,self.r_vloc,self.r_core)

        #unbounded_states
        for l in range(self.l_max+1):
            if l in self.unbounded_states:
                input_text+= 'y\n%s\nn\n'%(self.unbounded_states[l])
            else:
                input_text+='n\n'

        # projector keyword
        input_text+= "%s\t%s\n"%(self.projector_keyword,self.shape_function)

        # l_loc E_loc Vloc_scheme
        input_text+= "%s\t%s\t%s\n"%(self.l_loc,self.E_loc,self.Vloc_scheme)

        # r_cut
        ub_econf=[(10,l,None) for l in  list(self.unbounded_states.keys())]
        fake_econfs=self.econfs+ub_econf
        fake_econfs=[econf for econf in fake_econfs if(not (econf[0],econf[1]) in self.core_nl)]
        sorted_fake_econfs=sorted(fake_econfs,key=lambda x:(x[1],x[0]))
        for econf in sorted_fake_econfs:
            input_text+='%s\n'%(self.rcuts[econf[1]])

        #test region
        input_text+= '1\n'
        for pstate in self.get_partial_nl():
            input_text+= '\t'.join([str(x) for x in pstate])+'\n'
        input_text += '0\t0\t0\n'

        # output
        input_text +='2\n'
        input_text +='default\n'

        input_text +='3\n'
        input_text +='default\n'

        input_text +='4\n'
        input_text +='default\n'

        input_text +='0\n'
        print(input_text)

        with open('atom.inp','w') as infile:
            infile.write(input_text)

    def gen_pp(self):
        self.write_input_file()
        os.system('atompaw < atom.inp')


    def plot_wave_func(self):
        wfn_filenames=[x for x in os.listdir('.') if x.startswith('wfn') and not x.endswith('png')]
        sfname=sorted(wfn_filenames,key=lambda x:int(x[3:]))
        rs=[]
        phi0=[]
        phi1=[]
        p=[]
        for name in sfname:
            m=np.loadtxt(name)
            rs.append(m[:,0])
            phi0.append(m[:,1])
            phi1.append(m[:,2])
            p.append(m[:,3])
        for i in range(len(rs)):
            plt.clf()
            plt.plot(rs[i],phi0[i],linestyle='-')
            plt.plot(rs[i],phi1[i],linestyle='--')
            plt.axvline(self.r_core,linestyle='--',color='black')
            plt.savefig('wfn%d.png'%i)

            plt.clf()
            plt.plot(rs[i],p[i],'-')

            plt.vlines(self.r_core,-np.Inf,np.Inf,color='black')
            plt.savefig('proj%d.png'%i)






def test_paw():
    elem=atompaw('Mn')
    elem.add_empty_bands(4,1)
    elem.set_core('[Ne]')
    elem.set_radius(2.3)
    elem.set_rcuts([2.41,2.42,2.15])
    elem.add_unbounded_states(2,2)

    elem.write_input_file()
    elem.gen_pp()
    elem.plot_wave_func()

if __name__=='__main__':
    #test_s()
    test_paw()
