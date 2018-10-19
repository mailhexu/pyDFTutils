#!/usr/bin/env python
from collections import OrderedDict, namedtuple
import re
import os
import numpy as np
from scipy.linalg import eigh
from ase.atoms import string2symbols
from ase.data import atomic_masses, atomic_numbers
from ase.units import Ha

irpert = namedtuple('irpert', ['idir', 'ipert'])


class abinit_reader():
    """
    read from abinit output file.
    """

    def __init__(self, fname):
        self.fname = fname
        self.ndtset = 1
        with open(self.fname) as myfile:
            self.lines = myfile.readlines()
        self.ndtset = self.read_ndtset()
        self.dataset_in = []
        self.dataset_out = []
        self.split_dataset()

    def read_ndtset(self):
        for line in iter(self.lines):
            if line.find('ndtset') != -1:
                self.ndtset = int(line.strip().split()[1])
                return self.ndtset
        else:
            self.ndtset = 1
            return self.ndtset

    def split_dataset(self):
        lines = iter(self.lines)
        success = False
        idtset_in = -1
        idtset_out = -1
        for line in lines:
            # read the input datasets
            if line.strip().startswith('DATASET'):
                idtset_in = int(line.strip().split()[1])
                self.dataset_in.append([line])
                l = next(lines)  # Add the ===== and go to next line.
                self.dataset_in[-1].append(l)
                l = next(lines)
                while not l.startswith('====='):
                    self.dataset_in[-1].append(l)
                    l = next(lines)
                    self.dataset_in[-1].append(l)

            # read the output datasets
            if line.strip().startswith('== DATASET') \
               and line.strip().endswith('==='):
                idtset_out = int(line.strip().split()[2])
                self.dataset_out.append([line])
                next(lines)
            if idtset_out != -1:
                self.dataset_out[idtset_out - 1].append(line)

            if line.strip().startswith('== END DATASET'):
                success = True
                return 0

        if not success:
            print(
                ">>> WARNING: No END DATASET is found, maybe the output is imcompleted\n"
            )
        if len(self.dataset_in) != self.ndtset:
            print(
                ">>> WARNING: Number of input datasets not equal to ndtset\n")
        if len(self.dataset_out) != self.ndtset:
            print(
                ">>> WARNING: Number of input datasets not equal to ndtset\n")

    def parse_dataset(self, idtset):
        lines = self.dataset_out[idtset - 1]

    def get_matrix_part(self, idtset, name):
        """
        idtset: number of dataset.
        name: name of part. Ewald,Frozen_wf_non_local, Frozen_wf_xc1, Frozen_wf_xc2, non_stationary_local, non_stationary_non_local, non_stationary_wf_overlap,2nd_order (ifc), dynamical_matrix
        """
        namedict = {
            "Ewald": "Ewald",
        }

    def parse_perturbation_dataset(self, idtset, detail=True):
        lines = iter(self.dataset_out[idtset - 1])
        qpoints = None
        irperts = []
        comments = []
        E2ds = []  # Energy 2nd derivitives
        index_pert = 0
        read2DEmode = False
        for line in lines:
            # get Q point
            if line.strip().startswith('getcut:'):
                qpoints = list(map(float, line.strip().split()[2:5]))
            # get irreducible perturbations
            elif line.strip().startswith(
                    "The list of irreducible perturbations for this q vector is"
            ):
                l = next(lines)
                i = 0
                while l.strip().startswith("%s)" % (i + 1)):
                    res = re.findall("idir\s*=\s*(\d*)\s*ipert\s*=\s*(\d*)", l)
                    if res != []:
                        idir, ipert = list(map(int, res[0]))
                        irperts.append(irpert(idir, ipert))
                    l = next(lines)
                    i += 1
            # results for each perturbation.
            elif line.strip().startswith('Perturbation wavevector'):
                comment = ''
                comment += line
                comment += next(lines)
                comments.append(comment)
            # 2nd order E.
            elif line.strip().endswith(
                    'components of 2nd-order total energy (hartree) are'):
                # switch read on
                read2DEmode = True
                Edict = {}
            if detail:
                items = [
                    'kin0', 'eigvalue', 'local', 'loc psp', 'Hartree', 'xc',
                    'edocc', 'enl0', 'enl1', 'epaw1', 'erelax', 'fr\.local',
                    'fr\.nonlo', 'Ewald', 'frxc 1', 'frxc 2', 'eovl1',
                    '2DEtotal', '2DErelax', '2DEnonrelax'
                ]
            else:
                items = ['2DEtotal', '2DErelax', '2DEnonrelax', 'Ewald']
            if read2DEmode:
                for item in items:
                    pattern = "%s\s*=\s*(\S*)" % (item)
                    # print pattern
                    result = re.findall(pattern, line)
                    if result != []:
                        Edict[item] = float(result[0])
                if line.strip().startswith('---------') \
                   or line.strip().startswith('====='):
                    # end read mode
                    read2DEmode = False
                    E2ds.append(Edict)
        print("======================")
        print("E2Ds readed.")
        print(("Perturbation Wave vector: %s" % qpoints))
        for pert, E2d in zip(irperts, E2ds):
            print(('Irreducible perturbation: ', pert))
            print(('2nd Energies: %s' % E2d))
        print("======================")
        return qpoints, irperts, E2ds


class ddb_reader():
    def __init__(self, fname):
        self.fname = fname
        with open(self.fname) as myfile:
            self.lines = myfile.readlines()
        self.header, self.data = self.split_head()

    def split_head(self):
        header = []
        data = []
        in_header = True
        for line in self.lines:
            if line.strip().startswith(
                    '**** Database of total energy derivatives'):
                in_header = False
            if in_header:
                header.append(line)
            else:
                data.append(line)
        return header, data

    def read_elem(self):
        dym = {}
        ds = self.data[5:]
        for d in ds:
            idir1, ipert1, idir2, ipert2 = [
                int(x) for x in d.strip().split()[0:4]
            ]
            val = float(d.strip().split()[4].replace('D', 'E'))
            masses = [147.37, 47.88, 16, 16, 16, 8, 46, 34]
            val = val / np.sqrt(masses[ipert1 - 1] * masses[ipert2 - 1])
            dym[(idir1, ipert1, idir2, ipert2)] = val

        #Gfor d in self.data[5:]
        a = []
        for ipert2 in range(5):
            a.append(dym[(1, 1, 1, ipert2 + 1)])
        print(sum(a))

    def split_pjte_data(self):
        datahead = []
        d2matr_lines = []
        d2nfr_lines = []
        d2fr_lines = []
        d2ew_lines = []
        ind2matr = False
        ind2nfr = False
        ind2fr = False

        datahead = self.data[0:5]
        lines = iter(self.data)
        for line in lines:
            if line.strip().startswith('- Total 2nd-order matrix'):
                ind2matr = True
                while ind2matr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2matr_lines.append(l)
                    else:
                        ind2matr = False

            if line.strip().startswith(
                    '- Frozen part of the 2nd-order matrix'):
                ind2fr = True
                while ind2fr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2fr_lines.append(l)
                    else:
                        ind2fr = False

            if line.strip().startswith(
                    '- Non-frozen part of the 2nd-order matrix'):
                ind2nfr = True
                while ind2nfr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2nfr_lines.append(l)
                    else:
                        ind2nfr = False
            if line.strip().startswith('- Ewald part of the 2nd-order matrix'):
                ind2nfr = True
                while ind2nfr:
                    l = next(lines)
                    if not l.strip().startswith('End'):
                        d2ew_lines.append(l)
                    else:
                        ind2nfr = False

        return datahead, d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines

    def gen_pjte_ddbs(self):
        datahead, d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines = self.split_pjte_data(
        )
        prename = self.fname[0:-3]
        for ddbname, data in zip(
            ['TOT', 'NFR', 'FR', 'EW'],
            [d2matr_lines, d2nfr_lines, d2fr_lines, d2ew_lines]):
            fname = '%s%s_DDB' % (prename, ddbname)
            with open(fname, 'w') as myfile:
                myfile.write(''.join(self.header))
                myfile.write(''.join(datahead))
                myfile.write(''.join(data))
                #print("DDBfile %s is generated" % fname)


def read_DDB(fname):
    """
    Read total energy derivatives from DDB files.
    """
    dds = {}
    with open(fname) as myfile:
        for line in myfile:
            if line.find(
                    '**** Database of total energy derivatives ****') != -1:
                l = next(myfile)
                nblock = int(l.strip().split()[-1])
                #print "Nblock:",nblock
                next(myfile)
                l = next(myfile)
                nelem = int(l.strip().split()[-1])
                #print nelem
                l = next(myfile)
                qpts = [
                    float(x.replace('D', 'E')) for x in l.strip().split()[1:4]
                ]
                #print qpts
                for i in range(nelem):
                    try:
                        l = next(myfile)
                        idir1, ipert1, idir2, ipert2 = [
                            int(x) for x in l.strip().split()[0:4]
                        ]
                        realval, imageval = [
                            float(x.replace('D', 'E'))
                            for x in l.strip().split()[4:6]
                        ]
                        dds[(idir1, ipert1, idir2,
                             ipert2)] = realval + 1j * imageval
                    except:
                        pass
    return dds


def test_read(name):
    dirname = "%s_PM_pjte" % name
    abreader = abinit_reader(os.path.join(dirname, 'abinit.txt'))
    print((abreader.ndtset))
    abreader.split_dataset()
    print((abreader.dataset_in[0]))
    print((abreader.dataset_out[0][-6:-1]))
    abreader.parse_perturbation_dataset(4, detail=False)


def asr_1d(mat):
    for i in range(mat.shape[0]):
        mat[i, i] -= np.sum(mat[:, i])
    return mat


def get_mat1d(dirname, masses, term='TOT'):
    tot_name = os.path.join(dirname, 'abinito_DS2_%s_DDB') % term
    ifc = read_DDB(tot_name)
    ifcmat = np.zeros([15, 15], dtype=float)
    ifcmat1d = np.zeros([5, 5], dtype=float)
    for ipert1 in range(5):
        for ipert2 in range(5):
            ifcmat1d[ipert1, ipert2] = np.real(
                ifc[1, ipert1 + 1, 1, ipert2 + 1]) / np.sqrt(masses[ipert1] *
                                                             masses[ipert2])
    return ifcmat1d


def test_ddb(name):
    symbols = string2symbols(name)
    masses = [atomic_masses[atomic_numbers[s]] for s in symbols]

    # split ddb file
    dirname = "%s_PM_pjte" % name
    fname = os.path.join(dirname, 'abinito_DS2_DDB')
    reader = ddb_reader(fname)
    reader.gen_pjte_ddbs()
    ifcmat1d = get_mat1d(dirname, masses, term='TOT')
    #ifcmat1d=asr_1d(ifcmat1d)
    #print ifcmat1d
    eigvals, eigvecs = eigh(
        ifcmat1d, )
    s = np.sign(eigvals)
    #print eigvals
    #print "Phonon Freq (Ha):"
    #print s*np.sqrt(s*eigvals)/Ha
    print("Phonon Freq (cm-1):")
    print(s * np.sqrt(s * eigvals) * 19474.63 / Ha)
    #print np.dot(np.dot(v,ifcmat1d),v)

    print("--Eigen vector:--")
    v = eigvecs[:, 0]
    ifcmat1d_fr = get_mat1d(dirname, masses, term='FR')
    ifcmat1d_nfr = get_mat1d(dirname, masses, term='NFR')
    ifcmat1d_ew = get_mat1d(dirname, masses, term='EW')
    print(ifcmat1d_ew)
    print("self IFC")
    print([ifcmat1d[i, i] for i in range(5)])

    #func=lambda mat: np.sign(v)* np.sqrt(np.sign(v)*)
    def func(mat):
        eigv = np.dot(np.dot(v, mat), v)
        s = np.sign(eigv)
        return s * np.sqrt(s * eigv) * 19474.63 / Ha

    #print "w:"

    print("Total:", func(ifcmat1d))
    print("Frozen:", func(ifcmat1d_fr))
    print("Non-Frozen:", func(ifcmat1d_nfr))
    print("Ewald:", func(ifcmat1d_ew))

    for i in range(5):
        print("-- %s --" % symbols[i])
        v = [0, 0, 0, 0, 0]
        v[i] = 1.0

        #func=lambda mat: np.sign(v)* np.sqrt(np.sign(v)*)
        def func(mat):
            eigv = np.dot(np.dot(v, mat), v)
            s = np.sign(eigv)
            return s * np.sqrt(s * eigv) * 19474.63 / Ha

        print("Total:", func(ifcmat1d))
        print("Frozen:", func(ifcmat1d_fr))
        print("Non-Frozen:", func(ifcmat1d_nfr))
        print("Ewald:", func(ifcmat1d_ew))


if __name__ == '__main__':
    for name in [
            #'BaTiO3',
            'SrTiO3',
            'CaTiO3',
            'SnTiO3',
            'PbTiO3',
            'SrZrO3',
            'BaZrO3',
            'PbZrO3',
            'ZnTiO3',
            #'CsPbCl3',
            'CsPbBr3',
            'CsPbI3',
            'CsSrF3',
            'CsSnI3',
            'KCaF3',
            'KNbO3',
            'CsPbF3',
            #'CuTiO3',  # not yet
            'NaCaF3',
            'LiNbO3',
            'BiScO3',
            'BiAlO3',
            'BiGaO3',
            'NaNbO3',
            'KTaO3',
    ]:
        try:
            print("===================")
            print(name)
            test_ddb(name)
        except:
            pass
