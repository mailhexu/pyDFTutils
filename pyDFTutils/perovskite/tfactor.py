#! /usr/bin/env python
"""
calculate tolrance factor for perovskite ABO3 and ABF3,
data and function are stolen from http://www.me.utexas.edu/~benedekgroup/ToleranceFactorCalculator
And translated from javascript into python.
"""
from collections import defaultdict, OrderedDict
from itertools import product
import math

elem_data = [{
    "element": "Ac 3+",
    "ion": 3,
    "name": "Actinium",
    "oVal": 2.24,
    "fVal": 2.13
}, {
    "element": "Ag 1+",
    "ion": 1,
    "name": "Silver",
    "oVal": 1.805,
    "fVal": 1.8
}, {
    "element": "Al 3+",
    "ion": 3,
    "name": "Aluminum",
    "oVal": 1.651,
    "fVal": 1.545
}, {
    "element": "Am 3+",
    "ion": 3,
    "name": "Americium",
    "oVal": 2.11,
    "fVal": 2
}, {
    "element": "As 3+",
    "ion": 3,
    "name": "Arsenic",
    "oVal": 1.789,
    "fVal": 1.7
}, {
    "element": "As 5+",
    "ion": 5,
    "name": "Arsenic",
    "oVal": 1.767,
    "fVal": 1.62
}, {
    "element": "Au 3+",
    "ion": 3,
    "name": "Gold",
    "oVal": 1.833,
    "fVal": 1.81
}, {
    "element": "B 3+",
    "ion": 3,
    "name": "Boron",
    "oVal": 1.371,
    "fVal": 1.31
}, {
    "element": "Ba 2+",
    "ion": 2,
    "name": "Barium",
    "oVal": 2.29,
    "fVal": 2.19
}, {
    "element": "Be 2+",
    "ion": 2,
    "name": "Beryllium",
    "oVal": 1.381,
    "fVal": 1.28
}, {
    "element": "Bi 3+",
    "ion": 3,
    "name": "Bismuth",
    "oVal": 2.09,
    "fVal": 1.99
}, {
    "element": "Bi 5+",
    "ion": 5,
    "name": "Bismuth",
    "oVal": 2.06,
    "fVal": 1.97
}, {
    "element": "Bk 3+",
    "ion": 3,
    "name": "Berkelium",
    "oVal": 2.08,
    "fVal": 1.96
}, {
    "element": "Br 7+",
    "ion": 7,
    "name": "Bromine",
    "oVal": 1.81,
    "fVal": 1.72
}, {
    "element": "C 4+",
    "ion": 4,
    "name": "Carbon",
    "oVal": 1.39,
    "fVal": 1.32
}, {
    "element": "Ca 2+",
    "ion": 2,
    "name": "Calcium",
    "oVal": 1.967,
    "fVal": 1.842
}, {
    "element": "Cd 2+",
    "ion": 2,
    "name": "Cadmium",
    "oVal": 1.904,
    "fVal": 1.811
}, {
    "element": "Ce 3+",
    "ion": 3,
    "name": "Cerium",
    "oVal": 2.151,
    "fVal": 2.036
}, {
    "element": "Ce 4+",
    "ion": 3,
    "name": "Cerium",
    "oVal": 2.028,
    "fVal": 1.995
}, {
    "element": "Cf 3+",
    "ion": 3,
    "name": "Californium",
    "oVal": 2.07,
    "fVal": 1.95
}, {
    "element": "Cl 7+",
    "ion": 7,
    "name": "Chlorine",
    "oVal": 1.632,
    "fVal": 1.55
}, {
    "element": "Cm 3+",
    "ion": 3,
    "name": "Curium",
    "oVal": 2.23,
    "fVal": 2.12
}, {
    "element": "Co 2+",
    "ion": 2,
    "name": "Cobalt",
    "oVal": 1.692,
    "fVal": 1.64
}, {
    "element": "Co 3+",
    "ion": 3,
    "name": "Cobalt",
    "oVal": 1.7,
    "fVal": 1.62
}, {
    "element": "Cr 2+",
    "ion": 2,
    "name": "Chromium",
    "oVal": 1.73,
    "fVal": 1.67
}, {
    "element": "Cr 3+",
    "ion": 3,
    "name": "Chromium",
    "oVal": 1.724,
    "fVal": 1.64
}, {
    "element": "Cr 6+",
    "ion": 6,
    "name": "Chromium",
    "oVal": 1.794,
    "fVal": 1.74
}, {
    "element": "Cs 1+",
    "ion": 1,
    "name": "Cesium",
    "oVal": 2.42,
    "fVal": 2.33
}, {
    "element": "Cu 1+",
    "ion": 1,
    "name": "Copper",
    "oVal": 1.593,
    "fVal": 1.6
}, {
    "element": "Cu 2+",
    "ion": 2,
    "name": "Copper",
    "oVal": 1.679,
    "fVal": 1.6
}, {
    "element": "Dy 3+",
    "ion": 3,
    "name": "Dysprosium",
    "oVal": 2.036,
    "fVal": 1.922
}, {
    "element": "Er 3+",
    "ion": 3,
    "name": "Erbium",
    "oVal": 2.01,
    "fVal": 1.906
}, {
    "element": "Eu 2+",
    "ion": 2,
    "name": "Europium",
    "oVal": 2.147,
    "fVal": 2.04
}, {
    "element": "Eu 3+",
    "ion": 3,
    "name": "Europium",
    "oVal": 2.076,
    "fVal": 1.961
}, {
    "element": "Fe 2+",
    "ion": 2,
    "name": "Iron",
    "oVal": 1.734,
    "fVal": 1.65
}, {
    "element": "Fe 3+",
    "ion": 3,
    "name": "Iron",
    "oVal": 1.759,
    "fVal": 1.67
}, {
    "element": "Ga 3+",
    "ion": 3,
    "name": "Gallium",
    "oVal": 1.73,
    "fVal": 1.62
}, {
    "element": "Gd 3+",
    "ion": 3,
    "name": "Gadolinium",
    "oVal": 2.065,
    "fVal": 1.95
}, {
    "element": "Ge 4+",
    "ion": 4,
    "name": "Germanium",
    "oVal": 1.748,
    "fVal": 1.66
}, {
    "element": "H 4+",
    "ion": 4,
    "name": "Hydrogen",
    "oVal": .95,
    "fVal": .92
}, {
    "element": "Hf 4+",
    "ion": 4,
    "name": "Hafnium",
    "oVal": 1.923,
    "fVal": 1.85
}, {
    "element": "Hg 1+",
    "ion": 1,
    "name": "Mercury",
    "oVal": 1.9,
    "fVal": 1.6
}, {
    "element": "Hg 2+",
    "ion": 2,
    "name": "Mercury",
    "oVal": 1.93,
    "fVal": 1.6
}, {
    "element": "Ho 3+",
    "ion": 3,
    "name": "Holmium",
    "oVal": 2.023,
    "fVal": 1.908
}, {
    "element": "I 5+",
    "ion": 5,
    "name": "Iodine",
    "oVal": 2,
    "fVal": 1.9
}, {
    "element": "I 7+",
    "ion": 7,
    "name": "Iodine",
    "oVal": 1.93,
    "fVal": 1.83
}, {
    "element": "In 3+",
    "ion": 3,
    "name": "Indium",
    "oVal": 1.902,
    "fVal": 1.79
}, {
    "element": "Ir 5+",
    "ion": 5,
    "name": "Iridium",
    "oVal": 1.916,
    "fVal": 1.82
}, {
    "element": "K 1+",
    "ion": 1,
    "name": "Potassium",
    "oVal": 2.13,
    "fVal": 1.99
}, {
    "element": "La 3+",
    "ion": 3,
    "name": "Lanthanum",
    "oVal": 2.172,
    "fVal": 2.057
}, {
    "element": "Li 1+",
    "ion": 1,
    "name": "Lithium",
    "oVal": 1.466,
    "fVal": 1.36
}, {
    "element": "Lu 3+",
    "ion": 3,
    "name": "Lutetium",
    "oVal": 1.971,
    "fVal": 1.876
}, {
    "element": "Mg 2+",
    "ion": 2,
    "name": "Magnesium",
    "oVal": 1.693,
    "fVal": 1.581
}, {
    "element": "Mn 2+",
    "ion": 2,
    "name": "Manganese",
    "oVal": 1.79,
    "fVal": 1.698
}, {
    "element": "Mn 3+",
    "ion": 3,
    "name": "Manganese",
    "oVal": 1.76,
    "fVal": 1.66
}, {
    "element": "Mn 4+",
    "ion": 4,
    "name": "Manganese",
    "oVal": 1.753,
    "fVal": 1.71
}, {
    "element": "Mn 7+",
    "ion": 7,
    "name": "Manganese",
    "oVal": 1.79,
    "fVal": 1.72
}, {
    "element": "Mo 6+",
    "ion": 6,
    "name": "Molybdenum",
    "oVal": 1.907,
    "fVal": 1.81
}, {
    "element": "N 3+",
    "ion": 3,
    "name": "Nobelium",
    "oVal": 1.361,
    "fVal": 1.37
}, {
    "element": "N 5+",
    "ion": 5,
    "name": "Nobelium",
    "oVal": 1.432,
    "fVal": 1.36
}, {
    "element": "Na 1+",
    "ion": 1,
    "name": "Sodium",
    "oVal": 1.8,
    "fVal": 1.677
}, {
    "element": "Nb 5+",
    "ion": 5,
    "name": "Niobium",
    "oVal": 1.911,
    "fVal": 1.87
}, {
    "element": "Nd 3+",
    "ion": 3,
    "name": "Neodymium",
    "oVal": 2.117,
    "fVal": 2.008
}, {
    "element": "Ni 2+",
    "ion": 2,
    "name": "Nickel",
    "oVal": 1.654,
    "fVal": 1.599
}, {
    "element": "Ni 3+",
    "ion": 3,
    "name": "Nickel",
    "oVal": 1.686,
    "fVal": 0
}, {
    "element": "Os 4+",
    "ion": 4,
    "name": "Osmium",
    "oVal": 1.811,
    "fVal": 1.72
}, {
    "element": "P 5+",
    "ion": 5,
    "name": "Phosphorus",
    "oVal": 1.604,
    "fVal": 1.521
}, {
    "element": "Pb 2+",
    "ion": 2,
    "name": "Lead",
    "oVal": 2.112,
    "fVal": 2.03
}, {
    "element": "Pb 4+",
    "ion": 4,
    "name": "Lead",
    "oVal": 2.042,
    "fVal": 1.94
}, {
    "element": "Pd 2+",
    "ion": 2,
    "name": "Palladium",
    "oVal": 1.792,
    "fVal": 1.74
}, {
    "element": "Pr 3+",
    "ion": 3,
    "name": "Praseodymium",
    "oVal": 2.135,
    "fVal": 2.022
}, {
    "element": "Pt 2+",
    "ion": 2,
    "name": "Platinum",
    "oVal": 1.768,
    "fVal": 1.68
}, {
    "element": "Pt 4+",
    "ion": 4,
    "name": "Platinum",
    "oVal": 1.879,
    "fVal": 1.759
}, {
    "element": "Pu 3+",
    "ion": 3,
    "name": "Plutonium",
    "oVal": 2.11,
    "fVal": 2
}, {
    "element": "Rb 1+",
    "ion": 1,
    "name": "Rubidium",
    "oVal": 2.26,
    "fVal": 2.16
}, {
    "element": "Re 7+",
    "ion": 7,
    "name": "Rhenium",
    "oVal": 1.97,
    "fVal": 1.86
}, {
    "element": "Rh 3+",
    "ion": 3,
    "name": "Rhodium",
    "oVal": 1.791,
    "fVal": 1.71
}, {
    "element": "Ru 4+",
    "ion": 4,
    "name": "Ruthenium",
    "oVal": 1.834,
    "fVal": 1.74
}, {
    "element": "S 4+",
    "ion": 4,
    "name": "Sulfur",
    "oVal": 1.644,
    "fVal": 1.6
}, {
    "element": "S 6+",
    "ion": 6,
    "name": "Sulfur",
    "oVal": 1.624,
    "fVal": 1.56
}, {
    "element": "Sb 3+",
    "ion": 3,
    "name": "Antimony",
    "oVal": 1.973,
    "fVal": 1.9
}, {
    "element": "Sb 5+",
    "ion": 5,
    "name": "Antimony",
    "oVal": 1.942,
    "fVal": 1.8
}, {
    "element": "Sc 3+",
    "ion": 3,
    "name": "Scandium",
    "oVal": 1.849,
    "fVal": 1.76
}, {
    "element": "Se 4+",
    "ion": 4,
    "name": "Selenium",
    "oVal": 1.811,
    "fVal": 1.73
}, {
    "element": "Se 6+",
    "ion": 6,
    "name": "Selenium",
    "oVal": 1.788,
    "fVal": 1.69
}, {
    "element": "Si 4+",
    "ion": 4,
    "name": "Silicon",
    "oVal": 1.624,
    "fVal": 1.58
}, {
    "element": "Sm 3+",
    "ion": 3,
    "name": "Samarium",
    "oVal": 2.088,
    "fVal": 1.977
}, {
    "element": "Sn 2+",
    "ion": 2,
    "name": "Tin",
    "oVal": 1.984,
    "fVal": 1.925
}, {
    "element": "Sn 4+",
    "ion": 4,
    "name": "Tin",
    "oVal": 1.905,
    "fVal": 1.84
}, {
    "element": "Sr 2+",
    "ion": 2,
    "name": "Strontium",
    "oVal": 2.118,
    "fVal": 2.019
}, {
    "element": "Ta 5+",
    "ion": 5,
    "name": "Tantalum",
    "oVal": 1.92,
    "fVal": 1.88
}, {
    "element": "Tb 3+",
    "ion": 3,
    "name": "Terbium",
    "oVal": 2.049,
    "fVal": 1.936
}, {
    "element": "Te 4+",
    "ion": 4,
    "name": "Tellurium",
    "oVal": 1.977,
    "fVal": 1.87
}, {
    "element": "Te 6+",
    "ion": 6,
    "name": "Tellurium",
    "oVal": 1.917,
    "fVal": 1.82
}, {
    "element": "Th 4+",
    "ion": 4,
    "name": "Thorium",
    "oVal": 2.167,
    "fVal": 2.07
}, {
    "element": "Ti 3+",
    "ion": 3,
    "name": "Titanium",
    "oVal": 1.791,
    "fVal": 1.723
}, {
    "element": "Ti 4+",
    "ion": 4,
    "name": "Titanium",
    "oVal": 1.815,
    "fVal": 1.76
}, {
    "element": "Tl 1+",
    "ion": 1,
    "name": "Thallium",
    "oVal": 2.172,
    "fVal": 2.15
}, {
    "element": "Tl 3+",
    "ion": 3,
    "name": "Thallium",
    "oVal": 2.003,
    "fVal": 1.88
}, {
    "element": "Tm 3+",
    "ion": 3,
    "name": "Thulium",
    "oVal": 2,
    "fVal": 1.842
}, {
    "element": "U 4+",
    "ion": 4,
    "name": "Uranium",
    "oVal": 2.112,
    "fVal": 2.034
}, {
    "element": "U 6+",
    "ion": 4,
    "name": "Uranium",
    "oVal": 2.075,
    "fVal": 1.966
}, {
    "element": "V 3+",
    "ion": 3,
    "name": "Vanadium",
    "oVal": 1.743,
    "fVal": 1.702
}, {
    "element": "V 4+",
    "ion": 4,
    "name": "Vanadium",
    "oVal": 1.784,
    "fVal": 1.7
}, {
    "element": "V 5+",
    "ion": 5,
    "name": "Vanadium",
    "oVal": 1.803,
    "fVal": 1.71
}, {
    "element": "W 6+",
    "ion": 5,
    "name": "Tungsten",
    "oVal": 1.921,
    "fVal": 1.83
}, {
    "element": "Y 3+",
    "ion": 5,
    "name": "Yttrium",
    "oVal": 2.014,
    "fVal": 1.904
}, {
    "element": "Yb 3+",
    "ion": 3,
    "name": "Ytterbium",
    "oVal": 1.985,
    "fVal": 1.875
}, {
    "element": "Zn 2+",
    "ion": 2,
    "name": "Zinc",
    "oVal": 1.704,
    "fVal": 1.62
}, {
    "element": "Zr 4+",
    "ion": 4,
    "name": "Zirconium",
    "oVal": 1.937,
    "fVal": 1.854
}]

val_dict = defaultdict(list)
elem_dict = OrderedDict()

for item in elem_data:
    elem_val = item["element"]
    elem, val = elem_val.split()
    val_dict[val].append(elem)
    elem_dict[elem_val] = item

#print(valdict['2-'])
def tfactor(A, B, X):
    if X == 'O':
        e = elem_dict[A]['oVal']
        t = elem_dict[B]['oVal']
    elif X == 'F':
        e = elem_dict[A]['fVal']
        t = elem_dict[B]['fVal']
    else:
        raise NotImplementedError("X other than O and F is not implemented")
    ionA=elem_dict[A]['ion']
    ionB=elem_dict[B]['ion']
    n=e-0.37*math.log(ionA/12.0)
    r=t-0.37*math.log(ionB/6.0)
    i = n/(math.sqrt(2)*r)
    s=round(i*1e3)/1e3
    return s


def test():
    print(tfactor('Sr 2+','Ti 4+', 'O'))
    print(tfactor('Ba 2+','Si 4+', 'O'))
    print(tfactor('Rb 1+','Hg 2+', 'F'))
    print(tfactor('K 1+','Hg 2+', 'F'))

test()

A1B5O3_AB=list(product(val_dict['1+'], val_dict['5+']))
A1B5O3_t=[tfactor('%s 1+'%item[0], '%s 5+'%item[1],'O' ) for item in A1B5O3_AB]
A1B5O3=['%s%sO3'%(item) for item in A1B5O3_AB]


A2B4O3_AB=list(product(val_dict['2+'], val_dict['4+']))
A2B4O3_t=[tfactor('%s 2+'%item[0], '%s 4+'%item[1],'O' ) for item in A2B4O3_AB]
A2B4O3=['%s%sO3'%(item) for item in A2B4O3_AB]

A3B3O3_AB=list(product(val_dict['3+'], val_dict['3+']))
A3B3O3_t=[tfactor('%s 3+'%item[0], '%s 3+'%item[1],'O' ) for item in A3B3O3_AB]
A3B3O3=['%s%sO3'%(item) for item in A3B3O3_AB]

#tlist=sorted(zip(A3B3O3,A3B3O3_t), key=lambda x: x[1])
#for name, t in tlist:
#    print("%s\t%s"%(name, t))

tlist=sorted(zip(A1B5O3,A1B5O3_t), key=lambda x: x[1])
for name, t in tlist:
    print("%s\t%s"%(name, t))

