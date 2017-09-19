#!/usr/bin/env python
import os
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
from ase_utils.myabinit import read_output
from ase_utils.kpoints import cubic_kpath
from ase.io.vasp import write_vasp
from pyFA.abinit import DDB_reader
from pyFA.ifc import ifc_parser
from pyFA.phonon import dym_to_freq, eig_vec_to_eig_disp
from pyFA.matrix import matrix_elem_contribution, matrix_elem_sensitivity, mat_heatmap
import numpy as np
import copy
import abipy
from abipy.dfpt.ddb import DdbFile
from collections import OrderedDict
import json
import cPickle
import re
import string
from plotphon import plot_phon_from_nc

def read_DDB_info(name='LiNbO3'):
    myDDB = DdbFile(filepath='%s/abinit_mrgddb.out'%name, read_blocks=True)
    #print(myDDB._guess_ngqpt())
    #print(myDDB.params)
    qpoints = OrderedDict([('Gamma', (0, 0, 0)), ('X', (0.5, 0, 0)),
                           ('M', (0.5, 0.5, 0)), ('R', (0.5, 0.5, 0.5))])
    #qpoints=OrderedDict()
    #Full phonon band.

    # band without LO-TO

    # band with LO-TO

    # Born effective charge
    emacro, becs = myDDB.anaget_emacro_and_becs()

    #print("")
    #print("Macroscopic dielectric: ", emacro)
    #print("Born effective charges: ", becs)
    # dielectric
    # piezoelectricity
    # Analysis for each Q point.
    for qname in qpoints:
        #print("=============")
        #print("Qpoint: %s: %s" % (qname, qpoints[qname], ))
        #print(myDDB.qindex(qpoints[qname]))
        myphon_bands = myDDB.anaget_phmodes_at_qpoint(
            qpoint=qpoints[qname], workdir=None)
        #print("Freqencies:")
        #print myphon_bands.phfreqs
        #print("Eigenvectors:")
        #print myphon_bands.phfreqs
        #print("norm")
        #print np.linalg.norm(myphon_bands.dyn_mat_eigenvect[:, 1].real)
        #print myphon_bands.minfreq
        #print myphon_bands.maxfreq
    #print myDDB.anaget_emacro_and_becs()


#read_DDB_info()


def read_input_info(name='LiNbO3'):
    filefname = os.path.join(name, 'abinit.files')
    infname = os.path.join(name, 'abinit.in')
    #with open(filefname) as myfile:
    #print([os.path.split(x)[-1].strip() for x in myfile.readlines()[5:]])
    with open(infname) as myfile:
        text = myfile.read()
        nks = re.findall(r'ngkpt\s*\d*\s*\d*\s*\d*', text)[0].split()[1:]
        #print map(int, nks)


#read_input_info()


def gen_mrgddb_input(dirname):
    fnames = os.listdir(dirname)
    #print fnames
    DDBs = []
    for fname in fnames:
        if (fname.endswith('DDB') and fname.find('DS') != -1 and
                fname.find('DS1_') == -1):
            DDBs.append(fname)
    text = 'abinit_mrgddb.out\nUnnamed\n'
    text += str(len(DDBs)) + '\n'
    for DDB in sorted(DDBs):
        text += DDB + '\n'
    with open(os.path.join(dirname, 'abinit.mrgddb.in'), 'w') as myfile:
        myfile.write(text)

def gen_ifc_in(**kwargs):
    """ifcflag=1,brav=1,nk=2,qpts=[2,2,2],nqshft=1, ifcana=1, ifcout=20, natifc=5, atifc="1 2 3 4 5", chneut=1, dipdip=1,kpath=cubic_kpath()[0], nph2l=None"""
    #print kwargs
    if 'qpts' in kwargs:
        kwargs['qpts']=' '.join(map(str, kwargs['qpts']))

    text="""
!Input file for anaddb to generate ifc.
    asr 1
    alphon 1
    eivec=2
    ifcflag $ifcflag
    brav $brav
    ngqpt $qpts
    nqshft $nqshft
    q1shft 3*0.0
 
    ifcana $ifcana
    ifcout $ifcout
    natifc $natifc
    atifc $atifc

    chneut $chneut

    dipdip $dipdip\n"""
    template = Template(text)
    t=template.substitute(kwargs)
    if 'kpath' in kwargs:
        kpath=kwargs['kpath']
        nk=len(kpath)
        t+="!====Phonon band kpoints===\n"
        t+="    nph1l %s\n"%nk
        t+=" qph1l %s 1.0\n"%(' '.join([str(x) for x in kpath[0]]))
        for kpt in kpath[1:]:
            t+="       %s 1.0\n"%(' '.join([str(x) for x in kpt]))
    if nph2l in kwargs:
        t+= 'nph2l\n %s\n'%kwargs['nph2l']
    return t



class ifc_analyzer():
    def __init__(self, dirname, report_dir=None):
        self.dirname = dirname
        self.name=self.dirname
        if report_dir is None:
            self.report_dir = os.path.joint(dirname, 'Report')
        else:
            self.report_dir = report_dir
        if not os.path.exists(report_dir):
            os.makedirs(self.report_dir)
        self.read_input()
        self.report_tex = ""

    def run_anaddb(self, dipdip=1):
        print("Run anaddb")
        #os.system('cp ./abinit.mrgddb.tmpl %s/abinit.mrgddb.in' % self.dirname)
        gen_mrgddb_input(self.dirname)
        os.chdir(self.dirname)
        os.system('mrgddb <abinit.mrgddb.in>abinit.mrgddb.log')
        myDDB = DdbFile(
            filepath='abinit_mrgddb.out', read_blocks=True)

        nqpts = ' '.join(map(str, myDDB._guess_ngqpt()))
        natifc = len(self.atoms)
        atifc = ' '.join([str(i + 1) for i in range(len(self.atoms))])
        dipdip = dipdip

        tdict = {
            'nqpts': nqpts,
            'natifc': natifc,
            'atifc': atifc,
            'dipdip': dipdip
        }

        with open("../abinit_ifc.in.tmpl") as myfile:
            tmpl=string.Template(myfile.read())
        text=tmpl.substitute(tdict)

        os.system('cp ../abinit_ifc.files.tmpl abinit_ifc.files')
        with open('abinit_ifc.in','w') as myfile:
            myfile.write(text)
        #os.system('cp ../abinit_ifc.in.tmpl %s/abinit_ifc.in' % self.dirname)
        os.system('anaddb <abinit_ifc.files >abinit_ifc.log')
        os.chdir('../')

    def read_input(self, run_anaddb=True):
        dirname = self.dirname
        ifc_fname = os.path.join(dirname, 'abinit_ifc.out')
        out_fname = os.path.join(dirname, 'abinit.txt')
        DDB_file = os.path.join(dirname, 'abinit_mrgddb.out')
        #self.atoms = DDB_reader(DDB_file).read_atoms()
        self.atoms = read_output(out_fname)
        if run_anaddb:
            self.run_anaddb()
        #print self.atoms.get_cell()
        self.symbols = self.atoms.get_chemical_symbols()
        self.parser = ifc_parser(
            symbols=self.atoms.get_chemical_symbols(), fname=ifc_fname)
        self.model = self.parser.model

    def analyze_all_modes(self):
        self.results = OrderedDict()

        self.results[
            'lattice_parameter'] = self.atoms.get_cell_lengths_and_angles()

        # emacro & BEC
        myDDB = DdbFile(
            filepath='%s/abinit_mrgddb.out' % (self.dirname), read_blocks=True)

        emacro, becs = myDDB.anaget_emacro_and_becs()
        self.results['dielectric_tensor'] = emacro[0].reduced_tensor
        self.results['Born_effective_charges'] = becs

        self.results['number_of_qpoints'] = myDDB._guess_ngqpt()
        self.results['ecut'] = myDDB.params[u'ecut']
        self.results['tsmear'] = myDDB.params[u'tsmear']
        self.results['nsppol'] = myDDB.params[u'nsppol']
        self.results['nkpt'] = myDDB.params[u'nkpt']

        filefname = os.path.join(self.dirname, 'abinit.files')
        with open(filefname) as myfile:
            self.results['pseudopotentials'] = [
                os.path.split(x)[-1].strip() for x in myfile.readlines()[5:]
            ]

        # Phonon
        self.results['phonon'] = OrderedDict()
        qpoints = OrderedDict([('Gamma', (0, 0, 0)), ('X', (0.5, 0, 0)),
                               ('M', (0.5, 0.5, 0)), ('R', (0.5, 0.5, 0.5))])
        # Phonon: Gamma_reduced
        qname = 'Gamma_reduced'
        self.results['phonon'][qname] = OrderedDict()
        hamk, freqs, evals, evecs, edisps, dc, ds = self.analyze_gamma(
            ibands=None, label='DYM_Gamma_reduced', is_asr=False)
        self.results['phonon'][qname]['dynamic_matrix'] = hamk
        self.results['phonon'][qname]['eigenvalues'] = evals
        self.results['phonon'][qname]['frequencies'] = freqs
        self.results['phonon'][qname]['eigenvectors'] = evecs
        self.results['phonon'][qname]['eigen_displacements'] = edisps
        self.results['phonon'][qname]['dynamic_matrix_contributions'] = dc
        self.results['phonon'][qname]['dynamic_matrix_sensitivities'] = ds

        # Phonon: Other Qpoints
        for qname in qpoints:
            qpoint = qpoints[qname]
            self.results['phonon'][qname] = OrderedDict()
            self.results['phonon'][qname]['qpoint'] = qpoint
            hamk, freqs, evals, evecs, edisps, dc, ds = self.analyze_qpoint(
                qpoint=qpoint, qpt_name=qname, ibands=None)
            self.results['phonon'][qname]['dynamic_matrix'] = hamk
            self.results['phonon'][qname]['eigenvalues'] = evals
            self.results['phonon'][qname]['frequencies'] = freqs
            self.results['phonon'][qname]['eigenvectors'] = evecs
            self.results['phonon'][qname]['eigen_displacements'] = edisps
            self.results['phonon'][qname]['dynamic_matrix_contributions'] = dc
            self.results['phonon'][qname]['dynamic_matrix_sensitivities'] = ds
        #with open('data/%s.json' % (self.dirname), 'w') as myfile:
        #    json.dump(self.results, myfile)
        with open('data/%s.pickle' % (self.dirname), 'wb') as myfile:
            cPickle.dump(self.results, myfile)
        datadir = 'data/%s' % self.dirname
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        os.system('cp %s/*.in %s' % (self.dirname, datadir))
        os.system('cp %s/*.nc %s' % (self.dirname, datadir))
        os.system('cp %s/*DDB %s' % (self.dirname, datadir))
        os.system('cp %s/*.files %s' % (self.dirname, datadir))
        os.system('cp %s/CONTCAR %s' % (self.dirname, datadir))
        os.system('cp %s/*.txt %s' % (self.dirname, datadir))
        os.system('cp %s/*.out %s' % (self.dirname, datadir))
        os.system('tar -zcvf data/%s.tar.gz %s' % (self.dirname, datadir))

    def analyze_gamma(self,
                      ibands=None,
                      label='DYM_Gamma_reduced',
                      is_asr=False):
        if ibands is None:
            ibands = range(5)

        hamk_gamma = self.model.make_hamk([0, 0, 0])
        hamk_gamma = (hamk_gamma + hamk_gamma.conj().T) / 2

        red_hamk_gamma = hamk_gamma[1::3, 1::3]
        red_hamk_gamma = asr(red_hamk_gamma)

        #print("Freqs: ", dym_to_freq(red_hamk_gamma)[0])

        freqs, evecs, evals = dym_to_freq(red_hamk_gamma, eigenvalues=True)
        masses = self.atoms.get_masses()
        #print red_hamk_gamma
        #print np.sum(red_hamk_gamma, axis=0)
        mat_heatmap(
            red_hamk_gamma,
            title='Dynamic matrix',
            savefig=os.path.join(self.report_dir, '%s.png' % (label)),
            annot=True,
            linewidths=0.5,
            fmt='.4f')

        # dc: dynamic matrix contribution
        # ds: dynamic matrix sensitivity
        dc = OrderedDict()
        ds = OrderedDict()
        edisps = OrderedDict()
        for iband in ibands:
            edisps[iband] = eig_vec_to_eig_disp(
                evecs[iband], masses, dimension=1)
            dc[iband] = matrix_elem_contribution(
                red_hamk_gamma, evecs[:, iband], asr=is_asr)
            ds[iband] = matrix_elem_sensitivity(evecs[:, iband], asr=is_asr)
            mat_heatmap(
                dc[iband],
                title='Contribution to eigenvalue',
                savefig=os.path.join(self.report_dir, '%s_%s_Contribution.png'
                                     % (label, iband)),
                annot=True,
                linewidths=0.5,
                fmt='.4f')
            mat_heatmap(
                ds[iband],
                title='Eigenvalue sensitivity to DM',
                savefig=os.path.join(self.report_dir,
                                     '%s_%s_Sensitivity.png' % (label, iband)),
                annot=True,
                linewidths=0.5,
                fmt='.4f')
        return red_hamk_gamma, freqs, evals, evecs.T, edisps, dc, ds

    def prepare_wannier_gamma(self, band=0):
        hamk_gamma = self.model.make_hamk([0, 0, 0])
        hamk_gamma = (hamk_gamma + hamk_gamma.conj().T) / 2

        red_hamk_gamma = hamk_gamma[1::3, 1::3]
        red_hamk_gamma = asr(red_hamk_gamma)

        #print("Freqs: ", dym_to_freq(red_hamk_gamma)[band])

        freqs, evecs = dym_to_freq(red_hamk_gamma)
        masses = self.atoms.get_masses()
        disp = eig_vec_to_eig_disp(evecs[:, band], masses, dimension=1)
        disp = np.real(disp)
        #print disp
        disp = np.kron(disp, [0, 0, 1])
        disp.resize([5, 3])
        # print disp
        for amp in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            disp_atoms = copy.copy(self.atoms)
            disp_atoms.translate(disp * amp)
            disp_dir = os.path.join(
                './Wannier',
                self.atoms.get_chemical_formula(mode='reduce'),
                'Gamma_%s' % amp)
            # print disp_dir
            # print disp * amp
            if not os.path.exists(disp_dir):
                os.makedirs(disp_dir)
            write_vasp(
                os.path.join(disp_dir, 'POSCAR_disp.vasp'),
                disp_atoms,
                vasp5=True)

        # print red_hamk_gamma
        # print np.sum(red_hamk_gamma, axis=0)

    def analyze_qpoint(self, qpoint, qpt_name, ibands=None):
        label = 'DYM_' + qpt_name
        hamk = self.model.make_hamk(qpoint)
        hamk = (hamk + hamk.conj().T) / 2

        #print("Qpoint: ", qpt_name, qpoint)
        #print("Freqs: ", dym_to_freq(hamk))
        freqs, evecs, evals = dym_to_freq(hamk, eigenvalues=True)
        # print red_hamk_gamma
        # print np.sum(red_hamk_gamma, axis=0)
        mat_heatmap(
            hamk,
            annot=False,
            linewidths=0.5,
            savefig=os.path.join(self.report_dir, 'DYM_%s.png' % label))
        if ibands is None:
            ibands = range(len(freqs))
        dc = OrderedDict()
        ds = OrderedDict()
        edisps = OrderedDict()
        for iband in ibands:
            masses = self.atoms.get_masses()
            edisps[iband] = eig_vec_to_eig_disp(
                evecs[:, iband], masses, dimension=3)
            dc[iband] = matrix_elem_contribution(
                hamk, evecs[:, iband], asr=False)
            ds[iband] = matrix_elem_sensitivity(evecs[:, iband], asr=False)
            mat_heatmap(
                dc[iband],
                annot=False,
                savefig=os.path.join(self.report_dir, '%s_%s_Contribution.png'
                                     % (label, iband)),
                linewidths=0.5)
            mat_heatmap(
                ds[iband],
                annot=False,
                savefig=os.path.join(self.report_dir,
                                     '%s_%s_Sensitivity.png' % (label, iband)),
                linewidth=0.5)
        return hamk, freqs, evals, evecs.T, edisps, dc, ds

    def plot_phonon(self):
        #self.parser.solve_model(
        #    show=True,
        #    output_figure=os.path.join(self.report_dir, 'phonon_band.png'))
        fname=os.path.join(self.name, 'abinit_ifc.out_PHBST.nc')
        output_filename=os.path.join(self.report_dir, 'phonon_band.png')
        plot_phon_from_nc(fname,title=self.name,output_filename=output_filename)


def asr(mat):
    for i in range(mat.shape[0]):
        mat[i, i] = 0
    for i in range(mat.shape[0]):
        mat[i, i] = (-np.sum(mat[:, i]) - np.sum(mat[i])) / 2
    return mat

def gen_data(name):
    myifc = ifc_analyzer(name, report_dir='image/%s' % name)
    #myifc.run_anaddb()
    myifc.plot_phonon()
    plt.clf()
    #myifc.analyze_gamma(ibands=None)
    #myifc.analyze_qpoint([.5, .5, .0], 'M', ibands=None)
    myifc.analyze_all_modes()
    with open('Done.txt', 'a') as myfile:
        myfile.write('%s\n' % name)



def gen_disps():
    #for name in ['BaTiO3_bak','BaZrO3','SrTiO3','PbTiO3','LiNbO3','PbZrO3']:
    names = []
    for d in os.listdir('./'):
        if d.endswith('O3'):
            names.append(d)
    names=['LiOsO3','BaTiO3','PbTiO3','BiFeO3','LiNbO3']
    for name in names:#['BaTiO3']:
        print name
        try:  # not os.path.exists('data/%s.pickle'%name) and not name in ['BaFeO3','LaVO3','CaFeO3','SrFeO3','LaMnO3']:
            myifc = ifc_analyzer(name, report_dir='image/%s' % name)
            #myifc.run_anaddb()
            myifc.plot_phonon()
            #lt.clf()
            #myfic.run_anaddb(dipdip=0)
            myifc.analyze_gamma(ibands=None)
            #myifc.analyze_qpoint([.5, .5, .0], 'M', ibands=None)
            myifc.analyze_all_modes()
            with open('Done.txt', 'a') as myfile:
                myfile.write('%s\n' % name)
        except Exception:
            with open('fail.txt','a') as myfile:
                myfile.write('%s\n'%name)



if __name__=='__main__':
    gen_disps()
