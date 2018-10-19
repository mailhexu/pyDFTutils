#! /usr/bin/env python
""" An ASE interface to pwscf.
Users should set the environmental flag $PWSCF_COMMAND
to the command for pw.x e.g. 'pw.x' or 'mpirun -np 12 pw.x'
The default is 'mpirun -np 12 pw.x -npool 2 -ntg 2 -ndiag 3'
"""
import os
import re
from ase.calculators.general import Calculator
from os.path import join, isfile, islink
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from ase.dft.dos import DOS
from ase.units import Bohr,Ang,Hartree,eV,Rydberg
import ase
import matplotlib.pyplot as plt
import tempfile
from qe_pp_finder import find_pp_s

class pwscf(Calculator):
    """
    python interface to pwscf.

    :param atoms: the ase.Atoms object
    :param pp: a list like this: [('Fe' ,'Fe_***.UPF'),('O','O_***.UPF')]. Note that you don't have to define two element tag for one in antiferromagnetic calculation.\n
    :param ignore_bad_restart_file: If the restart file is bad, it will be cleaned.
    :param kpts: K-points, can be given as:\n
           (1,1,1): Gamma-point\n
           (n1,n2,n3): Monkhorst-Pack grid\n
           (n1,n2,n3,'gamma'): Shifted Monkhorst-Pack grid that includes \Gamma\n
           [(k11,k12,k13),(k21,k22,k23),...]: Explicit list in units of the reciprocal lattice vectors\n
           kpts=3.5: k-point density as in 3.5 k-points per ang^{-1}\n
    :param charge: charge
    :param nbands: nbnds
    :param xc: exchange-correlation function. ['pz','pw','pbe',...]
    :param kwargs: See http://www.quantum-espresso.org/wp-content/uploads/Doc/INPUT_PW.html\n
    Notes:
    charge, nbands, and xc are here to meet the standard of the ASE calculator. Do not use total_charge, nbnd, input_dft as in pwscf input file.\n
    The cell parameters, the postition parameters and the starting magnetization parameters should not be given here since they are defined in the Atoms.\n
    Unlike in the pwscf input file, the following parameters should be given as list instead of given as para(i)=...\n
    'celldm','starting_spin_angle', 'starting_magnetization', 'Hubbard_U', 'Hubbard_alpha', 'Hubbard_J', 'starting_ns_eigenvalue', 'angle1','angle2','fixed_magnetization','efield_cart'\n
    eg. Suppose that ntype=3, Hubbard_U(2)=3 should be writen as Hubbard_U=[0,3,0], Hubbard_J should be a 3*ntype array like object.\n
    Note that I do not recommend the using of the Hubbard parameters here. Use the set_Hubbard_U method instead, which is much easier to use.\n
    """
    def __init__(self,atoms=None,pp=None,ignore_bad_restart_file=False,kpts=['automatic',[4,4,4,0,0,0]],xc=None,smearing=None,charge=None,nbands=None,**kwargs):
        """

        """
        self.atoms=atoms
        self.auto_find_pp=False
        if charge is not None:
            kwargs['total_charge']=charge
        if nbands is not None:
            kwargs['nbnd']=nbands
        self.kpts=kpts

        self.special_kpts_names=None
        self.pp_setup = None
        if 'PW_COMMAND' in os.environ:
            self.pw_command = os.environ['PW_COMMAND']
        else:
            self.pw_command='mpirun -np 6 pw.x -npool 2 -ntg 3 -ndiag 1'



        ## params specified by type
        self.int_params=['nstep', 'iprint', 'nberrycyc', 'gdir', 'nppstr', 'ibrav', 'nat', 'ntyp', 'nbnd', 'nr1', 'nr2', 'nr3', 'nr1s', 'nr2s', 'nr3s', 'nspin','nqx1','nqx2', 'nqx3', 'lda_plus_u_kind', 'edir', 'report', 'esm_nfit', 'electron_maxstep', 'mixing_ndim', 'mixing_fixed_ns', 'ortho_para', 'diago_cg_maxiter', 'diago_david_ndim', 'nraise', 'bfgs_ndim']
        self.float_params=['dt', 'max_seconds', 'etot_conv_thr', 'forc_conv_thr','A', 'B', 'C', 'cosAB', 'cosAC', 'cosBC', 'tot_charge', 'tot_magnetization', 'ecutwfc', 'ecutrho', 'ecutfock', 'degauss', 'ecfixed', 'qcutz', 'q2sigma', 'exx_fraction', 'screening_parameter', 'ecutvcut', 'emaxpos', 'eopreg', 'eamp', 'lambda', 'esm_w', 'esm_efield', 'london_s6', 'london_rcut', 'conv_thr', 'conv_thr_init', 'conv_thr_multi', 'mixing_beta', 'diago_thr_init', 'efield', 'tempw', 'tolp', 'delta_t', 'upscale', 'trust_radius_max', 'trust_radius_min', 'trust_radius_ini', 'w_1', 'w_2', 'press', 'wmass', 'cell_factor', 'press_conv_thr']
        self.str_params= ['calculation', 'title', 'verbosity', 'restart_mode', 'outdir', 'wfcdir', 'prefix', 'disk_io', 'pseudo_dir', 'occupations', 'smearing', 'input_dft', 'exxdiv_treatment', 'U_projection_type', 'constrained_magnetization', 'assume_isolated', 'esm_bc', 'mixing_mode', 'diagonalization', 'startingpot', 'startingwfc', 'ion_dynamics', 'ion_positions', 'phase_space', 'pot_extrapolation', 'wfc_extrapolation', 'ion_temperature', 'cell_dynamics', 'cell_dofree']
        self.bool_params=['wf_collect', 'tstress', 'tprnfor', 'lkpoint_dir', 'tefield', 'dipfield', 'lelfield', 'lberry', 'nosym', 'nosym_evc', 'noinv', 'no_t_rev', 'force_symmorphic', 'use_all_frac', 'one_atom_occupations', 'starting_spin_angle', 'noncolin', 'lda_plus_u', 'lspinorb', 'london', 'scf_must_converge', 'adaptive_thr', 'diago_full_acc', 'tqr', 'remove_rigid_rot', 'refold_pos']

        self.list_params=['celldm','starting_spin_angle', 'starting_magnetization', 'Hubbard_U', 'Hubbard_alpha', 'Hubbard_J', 'starting_ns_eigenvalue', 'angle1','angle2','fixed_magnetization','efield_cart']
        ## params specified by tag
        self.control_params=['calculation', 'title', 'verbosity', 'restart_mode', 'wf_collect', 'nstep', 'iprint', 'tstress', 'tprnfor', 'dt', 'outdir', 'wfcdir', 'prefix', 'lkpoint_dir', 'max_seconds', 'etot_conv_thr', 'forc_conv_thr', 'disk_io', 'pseudo_dir', 'tefield', 'dipfield', 'lelfield', 'nberrycyc', 'lberry', 'gdir', 'nppstr']
        self.system_params=['ibrav', 'celldm', 'A', 'B', 'C', 'cosAB', 'cosAC', 'cosBC', 'nat', 'ntyp', 'nbnd', 'tot_charge', 'tot_magnetization', 'starting_magnetization', 'ecutwfc', 'ecutrho', 'ecutfock', 'nr1', 'nr2', 'nr3', 'nr1s', 'nr2s', 'nr3s', 'nosym', 'nosym_evc', 'noinv', 'no_t_rev', 'force_symmorphic', 'use_all_frac', 'occupations', 'one_atom_occupations', 'starting_spin_angle', 'degauss', 'smearing', 'nspin', 'noncolin', 'ecfixed', 'qcutz', 'q2sigma', 'input_dft', 'exx_fraction', 'screening_parameter', 'exxdiv_treatment', 'ecutvcut', 'nqx1', 'nqx2', 'nqx3', 'lda_plus_u', 'lda_plus_u_kind', 'starting_ns_eigenvalue', 'U_projection_type', 'edir', 'emaxpos', 'eopreg', 'eamp', 'angle1', 'angle2', 'constrained_magnetization', 'fixed_magnetization', 'lambda', 'report', 'lspinorb', 'assume_isolated', 'esm_bc', 'esm_w', 'esm_efield', 'esm_nfit', 'london', 'london_s6', 'london_rcut']
        self.electrons_params=['electron_maxstep', 'scf_must_converge', 'conv_thr', 'adaptive_thr', 'conv_thr_init', 'conv_thr_multi', 'mixing_mode', 'mixing_beta', 'mixing_ndim', 'mixing_fixed_ns', 'diagonalization', 'ortho_para', 'diago_thr_init', 'diago_cg_maxiter', 'diago_david_ndim', 'diago_full_acc', 'efield', 'efield_cart', 'startingpot', 'startingwfc', 'tqr']
        self.ions_params=['ion_dynamics', 'ion_positions', 'phase_space', 'pot_extrapolation', 'wfc_extrapolation', 'remove_rigid_rot', 'ion_temperature', 'tempw', 'tolp', 'delta_t', 'nraise', 'refold_pos', 'upscale', 'bfgs_ndim', 'trust_radius_max', 'trust_radius_min', 'trust_radius_ini', 'w_1', 'w_2']

        self.cell_params=['cell_dynamics','press', 'wmass','cell_factor','press_conv_thr','cell_dofree']
        ## This is a work out
        ## Currently inplemented Hubbard_U Hubbard_alpha. TODO: inplement others
        self.special_params=['celldm','starting_spin_angle', 'starting_magnetization', 'Hubbard_U', 'Hubbard_alpha', 'Hubbard_J', 'starting_ns_eigenvalue', 'angle1','angle2','fixed_magnetization','efield_cart']

        #set initial values to None, If value is not None, write to input file.
        self.param_keys=self.control_params+self.system_params+self.electrons_params+self.ions_params+self.cell_params+self.special_params
        self.params=dict()
        for key in self.param_keys:
            self.params[key]=None
        self.params['calculation']='scf'
        self.params['outdir']='OUTPUT'
        self.params['ibrav']=0
        self.params['nat']=0
        self.params['ntyp']=0
        self.params['ecutwfc']=35.0

        self.pp=pp

        # /data is in the local node, which is used to accelarate the calculation
        if 'wfcdir' not in kwargs and os.path.exists('/data'):
            kwargs['wfcdir']=tempfile.mkdtemp(prefix='qewfc',dir='/data')
        self.set(**kwargs)

        self.U_dict=None
        self.U_type=None

    def set(self,pp=None,kpts=None,**kwargs):
        """
        set paramters.

        """
        if pp is not None:
            self.pp=pp
        if kpts is not None:
            self.kpts=kpts
        for key in kwargs:
            if kwargs[key] is not None:
                # check value type
                if key in self.int_params:
                    try:
                        self.params[key]=int(kwargs[key])
                    except ValueError:
                        print("The value of %s should be int" )
                elif key in self.float_params:
                    try:
                        self.params[key]=float(kwargs[key])
                    except ValueError:
                        print("The value of %s should be float" )
                elif key in self.bool_params:
                    try:
                        assert(type(kwargs[key])==bool)
                        self.params[key]=kwargs[key]
                    except AssertionError:
                        print("The value of %s should be bool" )
                elif key in self.str_params:
                    try:
                        assert(isinstance(kwargs[key],str))
                        self.params[key]=kwargs[key]
                    except AssertionError:
                        print("The value of %s should be string" )
                elif key in self.list_params:
                    self.params[key]=list(kwargs[key])

    def set_pp(self,pp_type='US',xc_name='pz',rel=1,setups={}):
        """
        set pseudo potentials.

        :param pp_type: 'US'|'PAW' ultrasoft or paw
        :param xc_name: name of exchange-correlation functional. eg. 'pz','pbe'
        """
        self.pp_setup={ 'pp_type':pp_type,
                        'xc_name':xc_name,
                        'rel':rel,
                        'setups':setups
                        }
        self.params['pseudo_dir']='./pseudopotentials'

    def set_Hubbard_U(self,U_dict,type=1):
        """
        set Hubbard_U parameters.

        :param U_dict: A dict like this {'Fe':{'L':,2,'U':4,'J',0.3}, 'O':{'L':1,'U':1,'J':0.3} }. 'L' is orbital angular momentem. 0 1 2 3 -> s p d f. 'U' is the Hubbard_U parameter, Hubbard_J is the J parameter.
        :param type: type of LDA+U correction. If type==0, only U should be specified.

        Usage::

            calc.set_Hubbard_U({'Fe':{'L':,2,'U':4,'J',0.3}, 'O':{'L':1,'U':1,'J':0.3} },type=1)

        """
        print("setting hubbard U")
        self.U_type=type
        self.U_dict=U_dict
        self.params['lda_plus_u']=True
        self.params['lda_plus_u_kind']=type



    def set_atoms(self,atoms):
        """
        set the atoms.
        """
        self.atoms=atoms
        self.params['ntyp']=get_pw_ntyp(self.atoms)
        self.params['nat']=len(atoms)

    def set_pw_command(self,pw_command):
        """
        set pw.x command manually.
        """
        self.pw_command=pw_command

    def get_atoms(self,atoms):
        atoms=self.atoms.copy()
        atoms.set_calculator(self)


    def gen_kpoints_text(self):
        """
        generate K_POINTS card text
        """
        kpts=self.kpts
        print "Here The kpoint is",kpts
        if kpts is None:
            return 'K_POINTS {gamma}\n'
        elif isinstance(kpts,str):
            if kpts.lower()=='gamma':
                return 'K_POINTS {gamma}\n'
        elif isinstance(kpts,list):
            if kpts[0].lower()=='gamma':
                return 'K_POINTS {gamma}\n'
            elif kpts[0].lower()=='automatic':
                if len(kpts[1])==3:
                    return 'K_POINTS {automatic}\n  '+ ' '.join(map(str,kpts[1]+[0,0,0]))+'\n'+'\n'
                else:  ## len(kpts[1] == 6)
                    return 'K_POINTS automatic\n  '+ ' '.join(map(str,kpts[1]))+'\n'+'\n'
            else:   ## tpiba| crystal | tpiba_b | crystal_b | tpiba_c | crystal_c
                kpts_text='K_POINTS {%s}\n'%kpts[0]
                kpts_text+=str(len(kpts[1]))+'\n'
                for kpt in kpts[1]:
                    kpts_text += '  '+ ' '.join(map(str,kpt))+'\n'
                kpts_text+= '\n'
                return kpts_text




    def initialize(self,atoms):
        """
        generate input file
        """

        self.atoms=atoms

        ###  value_formatter

        def format_value(x):
            if type(x)==int or type(x)==float:
                return x
            elif type(x)==str:
                return "'%s'"%x
            elif type(x)==bool:
                return ('T' if x else 'F')


        ### CONTROL SECTION
        control_section='&CONTROL\n'
        for key in self.control_params:
            value=self.params[key]
            if value is not None:
                control_section+=('%s = %s \n'%(key,format_value(value)))

        control_section+='/\n'

        ### SYSTEM SECTION
        system_section='&SYSTEM\n'
        for key in self.system_params:
            value=self.params[key]
            if value is not None:
                system_section+=('%s = %s \n'%(key,format_value(value)))
        #Hubbard_U, the set_Hubbard_U method
        if self.U_dict is not None:
            pw_sym_dict=get_pw_chemical_symbols(self.atoms)
            ulist=[]

            for elem_sym in self.U_dict:
                print elem_sym
                for isym,pw_sym in enumerate(sort_set(pw_sym_dict)):

                    if re.findall('[a-zA-Z]+',pw_sym)[0]==elem_sym:
                        print "Fount"
                        ulist.append(self.U_dict[elem_sym]['U'])
                        if self.U_type==1:
                            #j_list.append((self.U_dict['L'],self.U_dict[''])
                            system_section+='Hubbard_J(%d,%d) = %f\n'%(1,isym+1,self.U_dict[elem_sym]['J'])
                            system_section+='Hubbard_U(%d) = %f\n'%(isym+1,self.U_dict[elem_sym]['U'])
                        if self.U_type==0:
                            system_section+='Hubbard_U(%d) = %f\n'%(isym+1,self.U_dict[elem_sym]['U'])
                    else:
                        ulist.append(0)
            self.params['Hubbard_U']=ulist
            print 'Hubbard U: ',ulist

        ulist =self.params['Hubbard_U']
        if ulist is not None:
            itype=0
            for i,u in enumerate(ulist):
                if abs(u)>0.001: #TODO clear this
                    pass
                    #system_section+='Hubbard_U(%d) = %f\n'%(itype,u)

        alpha_list =self.params['Hubbard_alpha']
        if alpha_list is not None:
            for i,alpha in enumerate(alpha_list):
                system_section+='Hubbard_alpha(%d) = %f\n'%(i+1,alpha)



        for itype,moment in get_pw_itype_moment(self.atoms):
            if abs(moment)>0.001:
                system_section+='starting_magnetization(%d) = %f \n'%(itype,moment)
        system_section+='/\n'
        ###TODO Impelment other list parameters.

        ### ELECTRONS SECTION
        electrons_section='&ELECTRONS\n'
        for key in self.electrons_params:
            value=self.params[key]
            if value is not None:
                electrons_section+=('%s = %s \n'%(key,format_value(value)))

        electrons_section+='/\n'

        ### IONS SECTION
        ions_section='&IONS\n'
        for key in self.ions_params:
            value=self.params[key]
            if value is not None:
                ions_section+=('%s = %s \n'%(key,format_value(value)))

        ions_section+='/\n'

        ### CELL SECTION
        cell_section='&CELL\n'
        for key in self.cell_params:
            value=self.params[key]
            if value is not None:
                cell_section+=('%s = %s \n'%(key,format_value(value)))

        cell_section+='/\n'


        ### ATOMIC_SPECIES
        atom_sp_section='ATOMIC_SPECIES\n'
        def get_elem_mass(elem_name):
            return ase.Atom(elem_name).mass

        elem_sym_set=[]
        if self.pp is not None:
            elem_pp_dict=dict(self.pp)
        else:
            elem_pp_dict=dict()

        if self.pp_setup is not None:
            # Use the pslib pp.
            self.params['pseudo_dir']='./pseudopotentials'
            if not os.path.exists('./pseudopotentials'):
                os.mkdir('./pseudopotentials')
            pp_info_file=open('./pseudopotentials/pp_info.txt','w')
            elem_set=set(self.atoms.get_chemical_symbols())
            elem_pp_dict={}
            for e in elem_set:
                if e in self.pp_setup['setups']:
                    label=self.pp_setup['setups'][e]
                else:
                    label='soft'
                pp_infos=find_pp_s(e,self.pp_setup['pp_type'],self.pp_setup['xc_name'],self.pp_setup['rel'],label=label)
                pp_info_file.write('\t'.join(str(x) for x in pp_infos))
                pp_info_file.write('\n')
                location=pp_infos[4]
                path,pp_filename=os.path.split(location)
                if os.path.exists(os.path.join('./pseudopotentials',pp_filename)):
                    os.remove(os.path.join('./pseudopotentials',pp_filename))
                os.symlink(location,os.path.join('./pseudopotentials',pp_filename))
                elem_pp_dict[e]=pp_filename


        for elem_sym in get_pw_chemical_symbols(self.atoms):
            if elem_sym not in elem_sym_set:
                elem_sym_set.append(elem_sym)
                elem=re.match('[A-Za-z]+',elem_sym).group()
                pp_filename=elem_pp_dict[elem]
                atom_sp_section+= ' '.join([elem_sym,str(get_elem_mass(elem)),pp_filename])+'\n'

        atom_sp_section+='\n'

        ### CELL_PARAMETERS
        cell_param_section='CELL_PARAMETERS {angstrom}\n'
        cell_vecs=self.atoms.get_cell()
        for vec in cell_vecs:
            cell_param_section+= ' '+ ' '.join(map(str,vec))+'\n'
        cell_param_section+='\n'


        ### ATOMIC_POSITIONS
        atom_pos_section='ATOMIC_POSITIONS {angstrom}\n'
        _iatom=0
        has_selective_dynamics=('selective_dynamics' in self.atoms.__dict__)
        for atom,sym in zip(self.atoms,get_pw_chemical_symbols(self.atoms)):
            if has_selective_dynamics:
                selective_dynamics_str='\t'+'\t'.join(map(str,self.atoms.selective_dynamics[_iatom]))
            else:
                selective_dynamics_str=''
            atom_pos_section+= ' '+ sym+' '+' '.join(map(str,atom.position))+selective_dynamics_str+'\n'
        atom_pos_section+='\n'
        _iatom+=1

        ### K_POINTS
        kpts_section= self.gen_kpoints_text()


        input_text='\n'.join([control_section,system_section,electrons_section,ions_section,cell_section,atom_sp_section,cell_param_section,atom_pos_section,kpts_section])

        self.input_filename='%s.%s.in'%(self.params['prefix'],self.params['calculation'])
        self.output_filename='%s.%s.out'%(self.params['prefix'],self.params['calculation'])
        with open(self.input_filename,'w') as myfile:
            myfile.write(input_text)

    def calculation_required(self,atoms,quantities):
        """
        Currently not implemented. Always return True. XXXX FIXME XXXX
        """
        return True

    def calculate(self,tmp_dir='/data'):
        #if tmp_dir is not None and os.path.exists(tmp_dir):
        #    pwd=os.getcwd()
        #    calc_tmp_dir=tempfile.mkdtemp(prefix='qetmp',dir=tmp_dir)
        #    os.chdir(calc_tmp_dir)
        self.initialize(self.atoms)
        command='%s < %s |tee %s'%(self.pw_command,self.input_filename,self.output_filename)
        exit_code=subprocess.call(command,shell=True)
        if exit_code !=0:
            raise Exception("pw.x failed")
        #if tmp_dir is not None and os.path.exists(tmp_dir):
        #    os.chdir(pwd)
        #    subprocess.call('mv -f %s/* .'%tmp_dir)
        #    os.removedirs(tmp_dir)

        self.read_output()

    def read_output(self):
        """
        Read the calculated results from the output files.
        """
        ### First READ from data-file.xml    # there are something missing in this file: The total energy
        ### the HOMO and LUMO, which can be read from the output text file.
        self.output_path=os.path.join('./',self.params['outdir'],'%s.save'%self.params['prefix'])
        self.filename=os.path.join(self.output_path,'data-file.xml')
        tree=ET.parse(self.filename)
        root=tree.getroot()
        self.creator_version=root[0].findall('CREATOR')[0].attrib

        ### LATTICE VECTORS

        LAT_VEC=list(root.iter('DIRECT_LATTICE_VECTORS'))[0]
        assert(LAT_VEC.find('UNITS_FOR_DIRECT_LATTICE_VECTORS').attrib['UNITS']=='Bohr')

        to_ang=lambda x: float(x)*Bohr/Ang
        a1=map(to_ang,LAT_VEC.findtext('a1').split())
        a2=map(to_ang,LAT_VEC.findtext('a2').split())
        a3=map(to_ang,LAT_VEC.findtext('a3').split())
        self.cell=[a1,a2,a3]


        self.natoms=int(float(list(root.iter('NUMBER_OF_ATOMS'))[0].text))
        self.atom_types=[a.text.strip() for a in root.iter('ATOM_TYPE')]
        self.atom_masses=[float(a.text.strip()) for a in root.iter('MASS')]
        self.atom_pseudos=[a.text.strip() for a in root.iter('PSEUDO')]
        self.pseudo_dir=list(root.iter('PSEUDO_DIR'))[0].text.strip()
        self.positions=[]
        for i in range(self.natoms):
            pos_text=list(root.iter('ATOM.%d'%(i+1)))[0].attrib['tau'].split()
            self.positions.append(map(to_ang,pos_text))
        print self.positions

        self.xc=list(root.iter('DFT'))[0].text.strip()

        T_or_F=(lambda x:False if x.strip().startswith('F') else True)
        self.plus_U=T_or_F(list(root.iter('DFT'))[0].text.strip())
        self.spin_polarized=T_or_F(list(root.iter('LSDA'))[0].text.strip())

        if self.spin_polarized:
            self.nspin=2
        else:
            self.nspin=1

        self.non_colinear=T_or_F(list(root.iter('NON-COLINEAR_CALCULATION'))[0].text.strip())

        ## get fermi energy
        MAG=list(root.findall('MAGNETIZATION_INIT'))[0]

        self.two_fermi_energys=T_or_F(MAG.findtext('TWO_FERMI_ENERGIES').strip())
        BAND=list(root.iter('BAND_STRUCTURE_INFO'))[0]
        if self.two_fermi_energys:
            self.fermi_energy_up=float(BAND.findtext('FERMI_ENERGY_UP'))*Hartree/eV
            self.fermi_energy_down=float(BAND.findtext('FERMI_ENERGY_DOWN'))*Hartree/eV
            self.fermi_energy=max(self.fermi_energy_up,self.fermi_energy_down)
        else:
            self.fermi_energy=float(BAND.findtext('FERMI_ENERGY'))*Hartree/eV

        ##nbands
        self.nbands=int(BAND.findtext('NUMBER_OF_BANDS').strip())

        ##get KPOINTS
        BRN=list(root.findall('BRILLOUIN_ZONE'))[0]
        self.nkpoints=int(BRN.findtext('NUMBER_OF_K-POINTS'))
        self.k_points=[]
        self.k_point_weights=[]
        for i in range(self.nkpoints):
            kp=BRN.findall('K-POINT.%d'%(i+1))[0]
            kp_xyz=[float(x) for x in kp.attrib['XYZ'].split()]
            kp_weight=float(kp.attrib['WEIGHT'])
            self.k_points.append(kp_xyz)
            self.k_point_weights.append(kp_weight)
        print self.k_points

        ### get eigenvalues.
        def read_eigen_xml(filename):
            """
            read the eigenvalues and occupations from xml file.
            """
            eig_tree=ET.parse(filename)
            eig_root=eig_tree.getroot()
            to_eV=lambda x: float(x)*Hartree/eV
            eigenvalues=map(to_eV,eig_root.findtext('EIGENVALUES').strip().split())
            occupations=map(to_eV,eig_root.findtext('OCCUPATIONS').strip().split())
            return eigenvalues, occupations

        self.eigenvalues=[]
        self.occupations=[]
        EIG=list(root.findall('EIGENVALUES'))[0]
        if not self.spin_polarized:
            for i in range(self.nkpoints):
                kp=EIG.findall('K-POINT.%d'%(i+1))[0]
                eig_filename=list(kp.findall('DATAFILE'))[0].attrib['iotk_link']
                eig_file=os.path.join(self.output_path,eig_filename[2:])
                print eig_file
                eigenvalue,occupation= read_eigen_xml(eig_file)

                self.eigenvalues.append(eigenvalue)
                self.occupations.append(occupation)

        else:
            for i in range(self.nkpoints):
                kp=EIG.findall('K-POINT.%d'%(i+1))[0]
                eig_filename1=list(kp.findall('DATAFILE.1'))[0].attrib['iotk_link']
                eig_file1=os.path.join(self.output_path,eig_filename1[2:])
                eigenvalue1,occupation1= read_eigen_xml(eig_file1)

                eig_filename2=list(kp.findall('DATAFILE.2'))[0].attrib['iotk_link']
                eig_file2=os.path.join(self.output_path,eig_filename2[2:])
                eigenvalue2,occupation2= read_eigen_xml(eig_file2)

                self.eigenvalues.append((eigenvalue1,eigenvalue2))
                self.occupations.append((occupation2,occupation2))

        ### Read from the output textfile.
        output_text=open(self.output_filename,'r').read()
        try:
            energy_str=re.findall('!\s*total energy\s*=\s*([-.\d]+)\s*Ry',output_text)
            if energy_str:
                self.total_energy=float(energy_str[-1])*Rydberg/eV
            else:
                print('Warning: total energy not found')

            self.homo=None
            self.lumo=None
            pattern=re.compile('highest\s*occupied.*')
            if re.search(pattern,output_text) is not None:
                line=re.findall(pattern,output_text)[-1]
                self.homo=float(line.strip().split()[-2])
                self.lumo=float(line.strip().split()[-1])
            else:
                self.homo=self.fermi_energy
                self.lumo=self.fermi_energy

        except Exception:
            print('Warning: HOMO and LUMO not found. Set them to Efermi, and hence Eg=0')

    def relax_calculation(self):
        """
        do relax calculation.
        """
        conv_thr=self.params['conv_thr']
        mixing_beta=self.params['mixing_beta']
        etot_conv_thr=self.params['etot_conv_thr']
        forc_conv_thr=self.params['forc_conv_thr'] # ion
        press_conv_thr=self.params['etot_conv_thr']

        nspin=self.params['nspin']

        nstep=self.params['nstep']

        self.set(calculation='vc-relax',conv_thr=1e-6,electron_maxstep=60,scf_must_converge=False,
                 mixing_beta=0.7,#diagonalization='cg', #electron
                 etot_conv_thr=1e-2,forc_conv_thr=2e-2, nstep=30,# ion
                 press_conv_thr=2
                 #nspin=1,
                ) #cell
        self.calculate()

        self.read_output()
        atoms=self.get_relaxed_atoms()

        self.set_atoms(atoms)
        print (conv_thr,mixing_beta,etot_conv_thr, forc_conv_thr, press_conv_thr)

        self.set(calculation='vc-relax',conv_thr=conv_thr,scf_must_converge=False,
                 mixing_beta=0.3,#diagonalization='cg', #electron
                 etot_conv_thr=etot_conv_thr,forc_conv_thr=forc_conv_thr, nstep=nstep, # ion
                 press_conv_thr=press_conv_thr,  #cell
                 nspin=nspin)
        self.calculate()
        self.read_output()
        atoms=self.get_relaxed_atoms()
        return atoms


    def get_forces(self,atoms):
        raise NotImplementedError

    def get_relaxed_atoms(self):
        """
        return the relaxed structure.
        """
        atoms=self.atoms.copy()
        atoms.set_positions(self.positions)
        atoms.set_cell(self.cell)
        return atoms

    def get_potential_energy(self,atoms=None,force_consistent=False):
        """
        get the total energy.
        """
        return self.total_energy

    def get_homo(self):
        """
        return HOMO energy
        """
        if self.homo is not None:
            return self.homo
        else:
            print('Warning: The homo was not found in the output text file.')

    def get_lumo(self):
        """
        return LUMO energy
        """
        if self.lumo is not None:
            return self.lumo
        else:
            print('Warning: The lumo was not found in the output text file.')

    def get_bandgap(self):
        """
        return the band gap
        """
        if self.homo is not None and self.lumo is not None:
            return self.lumo-self.homo
        else:
            print('Warning: The homo or lumo was not found in the output text file.\n')
            return 0.0


    def get_stress(self,atoms):
        """
        return stress
        """
        return self.stress


    def get_bz_k_points(self):
        """
        return k-points.
        """
        return self.k_points

    def get_effective_potential(spin=0,pad=True):
        raise NotImplementedError

    def get_eigenvalues(self,kpt=0,spin=0):
        """
        Input:
         kpt: k point.
         spin: 1|2
        return eigenvalues.
        """
        if self.get_number_of_spins()==1:
            print "nspin =1"
            return self.eigenvalues[kpt]
        else:
            return self.eigenvalues[kpt][spin]

    def get_fermi_level(self):
        """
        return the Fermin energy.
        """
        return self.fermi_energy

    def get_ibz_k_points(self):
        return self.k_points
        raise NotImplementedError

    def get_k_point_weights(self):
        """
        return the weights of k-points.
        """
        return self.k_point_weights

    def get_magnetic_moment(self,atoms=None):
        """
        return the total magnetic moment
        """
        return self.total_magnetic_moment

    def get_number_of_bands(self):
        """
        return the number of bands.
        """
        return self.nbands

    def get_number_of_grid_points(self):
        raise NotImplementedError

    def get_number_of_spins(self):
        """
        return the number of spin channels
        """
        return self.nspin

    def get_occupation_numbers(self,kpt=0,spin=0):
        """
        return the occupation numbers.

        :param kpt: the id of k-point.
        :param spin: which spin 0|1
        """
        return self.occupations[kpt][spin]

    def get_pseudo_density(self,spin=None,pad=True):
        raise NotImplementedError

    def get_pseudo_wave_function(self,band=0, kpt=0, spin=0, broadcast=True, pad=True):
        raise NotImplementedError
        ## TODO

    def get_spin_polarized(self):
        return self.spin_polarized


    def get_wannier_localization_matrix(self,nbands, dirG, kpoint, nextkpoint, G_I, spin):
        raise NotImplementedError

    def get_xc_functional(self):
        return self.xc

    def initial_wannier(self,initialwannier, kpointgrid, fixedstates, edf, spin, nbands):
        raise NotImplementedError


    def get_dos(self,kpoints=None,width=0.1,window=None,npts=201,spin=None):
        """
        calculate density of states.

        :param kpoints: set k-points for DOS calculation.
        :param width: smearing width
        :param window: energy windown. e.g. [-5,5]
        :param npts: number of points.
        :param spin: No use here. Just to keep with the ase API.
        """
        if kpoints is not None:
            self.params['calculation']='nscf'
            self.kpts=['automatic',kpoints]
            self.params['occupations']='tetrahedra'
            self.initialize(self.atoms)
            self.calculate()
            self.read_output()
        mydos=DOS(self,width=width,window=window,npts=201)
        self.dos_energies=mydos.get_energies()
        self.dos=mydos.get_dos()
        if self.spin_polarized:
            self.dos_up=mydos.get_dos(spin=0)
            self.dos_down=mydos.get_dos(spin=1)
        #return mydos.get_energies(),mydos.get_dos()

    def plot_dos(self,two_spins=True,output_filename=None,show=False):
        """
        Plot DOS to file.

        :param two_spins: if two_spins, draw spin_up and spin down in one figure, else in two seperate figures.
        :param output_filename: the name of the output file. The default is prefix + '.dos.png'.
        :param show: whether to show on screen.
        """
        if output_filename is None:
            output_filename=self.params['prefix']+'.dos.png'
        plt.clf()
        if not(two_spins and self.spin_polarized):
            plt.plot(self.dos_energies,self.dos)
        else:
            plt.plot(self.dos_energies,self.dos_up)
            plt.plot(self.dos_energies,-np.array(self.dos_down))
        plt.xlabel('Energy (eV)')
        plt.ylabel('DOS')
        if output_filename is not None:
            plt.savefig(output_filename)
        if show:
            plt.show()

    def get_bands(self,kpoints,kpts_names=None):
        """
        calculate the bands.

        :param kpoints: [(kpoint0,nk0-1),(kpoint1,nk1-2),...]
        :param kpts_names: list of points of special kpts. ['\Gamma','R','M',...]

        """
        self.params['calculation']='bands'
        self.kpts=['tpiba_b',kpoints]
        self.initialize(self.atoms)
        self.calculate()
        self.read_output()
        self.band_xs=range(sum([x[-1] for x in kpoints[:-1]]))
        self.band_special_xs=[0]
        i=0
        for k in kpoints[:-1]:
            i=i+k[3]
            self.band_special_xs.append(i)
        self.special_kpts_names=kpts_names

    def plot_bands(self,window=None,output_filename=None,show=False,spin=0):
        """
        plot the bands.
        window: (Emin,Emax), the range of energy to be plotted in the figure.
        speicial_kpts_name
        """
        if output_filename is None:
            output_filename=self.params['prefix']+'.bands.png'
        plt.clf()
        if window is not None:
            plt.ylim(window[0],window[1])
        if not self.spin_polarized:
            eigenvalues=np.array(self.eigenvalues)
            for i in range(self.nbands):
                band_i=eigenvalues[:,i]-self.get_fermi_level()
                plt.plot(band_i)
        else:
            eigenvalues=np.array(self.eigenvalues)
            print eigenvalues
            for i in range(self.nbands):
                band_i=eigenvalues[:,spin,i]-self.get_fermi_level()
                plt.plot(band_i)
        plt.xlabel('K-points')
        plt.ylabel('$Energy-E_{fermi} (eV)$')

        plt.axhline(0,color='black',linestyle='--')
        if self.special_kpts_names is not None:
            plt.xticks(self.band_special_xs,self.special_kpts_names)

        if output_filename is not None:
            plt.savefig(output_filename)
        if show:
            plt.show()

    def calculate_all_step(self,vc_relax=True,relax=False,dos=True,bands=True,dos_kpts=[5,5,5],bands_kpts=None,special_kpt_names=None):
        """
        All the calculations, including vc_relax, relax, scf, dos, and band.
        Input:

        :param vc_relax: whether to do vc_relax. True|False
        :param relax: whether to do relax. True|False
        :param dos:  whether to calculate DOS. True|False
        :param dos:  whether to calculate band diagram. True|False
        :param dos_kpts: [nk1, nk2, nk3].
        :param bands_kpts: [ kpt1, kpt2, ...]
         eg.::

            kp=ase.dft.kpoints.ibz_points['tetragonal']
            sp_kps=[kp['Gamma'],kp['X'],kp['R'],kp['Gamma'],kp['M']]
            bands_kpts=[x+[10] for x in sp_kps]
            special_kpt_names=['$\Gamma$','X','R','$\Gamma$','M']

        :param special_kpts_names: The name of special k-points. See bands_kpts.
        """
        if vc_relax:  ## vc-relax
            self.params['calculation']='vc-relax'
            self.calculate()
            atoms=self.get_relaxed_atoms()
            self.set_atoms(atoms)

        #scf
        self.params['calculation']='scf'
        self.calculate()

        ## write selection of properties.
        ase.io.write('atoms.xsf',self.atoms)


        #nscf: DOS
        self.get_dos()
        self.plot_dos()

        #Bands
        if bands:
            if bands_kpts is None:
                kp=ase.dft.kpoints.ibz_points['tetragonal']
                sp_kps=[kp['Gamma'],kp['X'],kp['R'],kp['Gamma'],kp['M']]
                bands_kpts=[x+[10] for x in sp_kps]
                special_kpt_names=['$\Gamma$','X','R','$\Gamma$','M']
            self.get_bands(bands_kpts,kpts_names=special_kpt_names)
            self.plot_bands(window=(-8,8),show=False)


class pwpp():
    """
    python interface to pp.x
    """
    def __init__(self,mycalc=None,prefix=None,outdir=None):
        if mycalc is not None:
            self.prefix=mycalc.params['prefix']
            self.outdir=mycalc.params['outdir']
        elif prefix is not None and outdir is not None:
            self.prefix=prefix
            self.outdir=outdir
        else:
            raise ValueError("You must specify a calculator or the prefix and outdir of the calculation")

    def run_ppx(self):
        subprocess.call('pp.x < %s |tee %s'%(self.infilename,self.outfilename),shell=True)


    def get_charge(self,spin=None,plot_type='3D',output_format='cube',output_filename='charge.cube',npoints=[50,50,50]):
        """
        output charge
        Inputs:
        spin: None|1|2
        plot_type: '3D'|'2D'|'1D'
        output_format:'gnuplot'|'contour.x'|'plotrho'|'xcrysden-2d'|'xcrysden-3d'|'gopenmol'|'cube'
        output_filename:output file name
        npoints: a list of [nx, ny, nz] (if 3D) | [nx,ny] (if 2D)| [nx] (if 1D)
        """
        if spin is None:
            spin_component=0
        elif spin==1:
            spin_component=1
        elif spin==2:
            spin_component=2
        else:
            raise ValueError('Spin must be None,1,or 2')

        plot_type_dict={'1D':1, '2D':2 ,'3D':3}
        if plot_type in plot_type_dict:
            iflag=plot_type_dict[plot_type]
        else:
            raise ValueError('Plot type must be 1D, 2D or 3D')

        output_format_dict={'gnuplot':0, 'contour.x':1,'plotrho':2, 'xcrysden-2d':3, 'gopenmol':4,'xcrysden-3d':5,'cube':6}
        output_format=output_format_dict[output_format]
        self.pp_text="""
        &inputpp
         prefix={prefix}
         outdir={outdir}
         filplot={output_filename}
         plot_num=0
         spin_component={spin_component}
        /

        &plot
         iflag={iflag}
         output_format={output_format}
         fileout={output_filename}
         nx={npoints[0]}
         ny={npoints[1]}
         nz={npoints[2]}
        /
        """.format(prefix=self.prefix, outdir=self.outdir, output_filename=output_filename,spin_component=spin_component,iflag=iflag,output_format=output_format,npoints=npoints)

        print self.pp_text
        self.infilename='%s.pp_gencube.in'%self.prefix
        self.outfilename='%s.pp_gencube.out'%self.prefix
        with open(self.infilename,'w') as infile:
            infile.write(self.pp_text)
        self.run_ppx()



def sort_set(x):
    """
    remove the duplicate elements,which is like converting a list to a set in python, but keeps the order.
    """
    result=[]
    for i in x:
        if not i in result:
            result.append(i)
    return result



def get_pw_chemical_symbols(atoms):
    """
    This is a workout for the calculation of antiferromagnetic structure.
    Because different species should be defined for one element.
    A method to get the modified name is implemented here.
    get symbol and initial magnetic moment.if  one atom has the same symbol and different moment with another, it is given a new name. eg.

    ..code-block:: python

        atoms=ase.Atoms('Fe4',positions=[(0,0,0),(1,1,1),(2,2,2),(3,3,3)])
        atoms.set_initial_magnetic_moments([1,-1,1,2])
        get_pw_chemical_symbols(atoms)

        >>>['Fe_1', 'Fe_2', 'Fe_1', 'Fe_3']

    """
    symbols=atoms.get_chemical_symbols()
    moments=atoms.get_initial_magnetic_moments()
    sym_mmt=zip(symbols,moments)

    smset=[] # use a list instead of a real set, to keep the order.
    for s,m in sym_mmt:
        if (s,m) not in smset:
            smset.append((s,m))
    new_sym_dict=dict()
    for sym in set(symbols):
        sm=filter((lambda x: x[0]==sym), smset)
        if len(sm)==1:
            new_sym_dict[sm.pop()]=sym
        else:
            for i,v in enumerate(sm):
                new_sym_dict[v]=sym+'%d'%(i+1)
    new_symbols=[]
    for symbol,moment in zip(symbols,moments):
        new_symbols.append(new_sym_dict[(symbol,moment)])
    return new_symbols

def get_pw_itype_moment(atoms):
    """
    This is a workout for the calculation of antiferromagnetic structure.
    Return a list of tuple (symbol, moment) with out duplicate elements.
    """
    symbols=atoms.get_chemical_symbols()
    moments=atoms.get_initial_magnetic_moments()
    sym_mmt=zip(symbols,moments)

    smset=[] # use a list instead of a real set, to keep the order.
    for s,m in sym_mmt:
        if (s,m) not in smset:
            smset.append((s,m))
    result=[]
    for i,sm in enumerate(smset):
        itype,moment=i+1,sm[1]
        result.append((itype,moment))
    return result


def get_pw_ntyp(atoms):
    """
    number of element types (same element but different magnetization treated as different type)

    """
    return len(set(get_pw_chemical_symbols(atoms)))
