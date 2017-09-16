#!/usr/bin/env python

from ase.dft.kpoints import special_paths, special_points, parse_path_string, bandpath
import numpy as np


def cubic_kpath():
    special_path = special_paths['cubic']
    points = special_points['cubic']
    paths = parse_path_string(special_path)
    special_kpts = [points[k] for k in paths[0]]
    kpts, xs, xspecial = bandpath(special_kpts, cell=np.eye(3), npoints=500)
    return kpts, xs, xspecial


print(cubic_kpath())
