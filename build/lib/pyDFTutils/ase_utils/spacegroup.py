#! /usr/bin/env python

from lxml import etree
from ase.io.exciting import read_exciting
from ase.io.xsf import read_xsf
from ase.io.elk import read_elk
import os
from ase.units import Bohr, Angstrom


class lattice():
    def __init__(self,
                 spacegroup,
                 cellpars,
                 species,
                 positions,
                 repeat=[1, 1, 1]):
        self.spacegroup = spacegroup
        self.cellpars = cellpars
        self.repeat = repeat
        self.species = species
        self.positions = positions

    def inp_gen(self, outfile='spacegroup.in'):
        """
        elk spacegroup input file generator
        """
        text = ''
        text += "'%s'\n" % self.spacegroup
        text += ' '.join(
            map(str, [x / Bohr * Angstrom for x in self.cellpars[:3]])) + '\n'
        text += ' '.join(map(str, self.cellpars[3:])) + '\n'
        text += ' '.join(map(str, self.repeat)) + '\n'
        text += '.true.\n'
        text += str(len(list(self.species))) + '\n'
        for i, sp in enumerate(self.species):
            text += "'%s'\n" % sp
            text += '1\n'
            text += ' '.join(map(str, self.positions[i])) + '\n'
        with open('spacegroup.in', 'w') as myfile:
            myfile.write(text)
        return text

    def xml_gen(self, outfile='spacegroup.xml'):
        root = etree.Element("symmetries", HermannMauguinSymbol='Bmab')
        title = etree.SubElement(root, 'title')
        title.text = 'tmp_input'

        lattice = etree.SubElement(
            root,
            'lattice',
            a=str(self.cellpars[0]),
            b=str(self.cellpars[1]),
            c=str(self.cellpars[2]),
            ab=str(self.cellpars[3]),
            ac=str(self.cellpars[4]),
            bc=str(self.cellpars[5]),
            ncell=' '.join([str(s) for s in self.repeat]))

        wp = etree.SubElement(root, 'WyckoffPositions')

        for s, p in zip(self.species, self.positions):
            stree = etree.SubElement(wp, 'wspecies', speciesfile="%s.xml" % s)
            ptree = etree.SubElement(
                stree, 'wpos', coord=' '.join((str(s) for s in p)))

        print((etree.tostring(
            root, pretty_print=True, xml_declaration=True, encoding="UTF-8")))
        if outfile:
            with open(outfile, 'w') as myfile:
                myfile.write(
                    etree.tostring(
                        root,
                        pretty_print=True,
                        xml_declaration=True,
                        encoding="UTF-8"))

    def gen_atoms(self, type='elk'):
        if type == 'elk':
            self.inp_gen()
            os.system('spacegroup')
            atoms = read_elk('GEOMETRY.OUT')
        elif type == 'exciting':
            self.xml_gen()
            os.system('spacegroup')
            atoms = read_exciting('geometry.out.xml')
        return atoms


def gen_lattice(spacegroup, cellpars, species, positions, repeat=[1, 1, 1]):
    a = lattice(spacegroup, cellpars, species, positions, repeat=[1, 1, 1])
    return a.gen_atoms()


def test():
    a = lattice('Bmab', [10, 10, 24, 90, 90, 90], ['La', 'Cu', 'O', 'O'],
                [[0, 0, 0.36], [0, .0, 0], [.25, .25, 0], [0, 0, 0.18]])
    print(a.inp_gen())
    atoms = a.gen_atoms()
    return atoms


if __name__ == '__main__':
    atoms = test()
    from .ase_utils import vesta_view
    vesta_view(atoms)
