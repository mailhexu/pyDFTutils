#!/usr/bin/env python
from collections import OrderedDict, namedtuple
import re

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
                items = ['2DEtotal', '2DErelax', '2DEnonrelax','Ewald']
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
        print(("Perturbation Wave vector: %s" % qpoints))
        for pert, E2d in zip(irperts, E2ds):
            print(('Irreducible perturbation: ', pert))
            print(('2nd Energies: %s' % E2d))
        print("======================")
        return qpoints, irperts, E2ds


def test_read():
    abreader = abinit_reader('abinit.txt')
    print((abreader.ndtset))
    abreader.split_dataset()
    print((abreader.dataset_in[0]))
    print((abreader.dataset_out[0][-6:-1]))
    abreader.parse_perturbation_dataset(4, detail=False)


#test_read()
