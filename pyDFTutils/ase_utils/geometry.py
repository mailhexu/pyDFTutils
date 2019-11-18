#! /usr/bin/env python

from ase import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.io import read
from ase.dft.kpoints import get_bandpath
import os
import math
from numpy import array
import numpy as np
from spglib import spglib
import matplotlib.pyplot as plt
#from ase.optimize import BFGS,BFGSLineSearch
from ase.utils.geometry import cut
from .ioput import my_write_vasp
from .symbol import symbol_number, symnum_to_sym, get_symdict
import copy

def gen_STO():
    a = b = c = 3.94
    alpha = beta = theta = 90
    atoms = Atoms(
        symbols='SrTiO3',
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0, 0.5, 0.5), (
            0.5, 0, 0.5), (0.5, 0.5, 0)],
        cell=cellpar_to_cell([a, b, c, alpha, beta, theta]))
    atoms = atoms.repeat([1, 1, 2])
    #atoms.set_initial_magnetic_moments()
    return atoms


def find_sym(atoms):
    return spglib.get_spacegroup(atoms, symprec=5e-4)


def get_prim_atoms(atoms, symprec=1e-4):
    return spglib.find_primitive(atoms, symprec=symprec)


def ref_atoms_mag(atoms):
    """
    substitute atom with magnetic moment to another atom object. Use He Ne Ar Kr Xe Rn as subsititutions. So if you have these atoms , this fucntion can be rather buggy. Do *NOT* use it in that case.
    """
    symbols = atoms.get_chemical_symbols()
    magmoms = atoms.get_initial_magnetic_moments()
    sub_syms = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
    sym_dict = {}
    syms = []
    for sym, mag in zip(symbols, magmoms):
        if sym not in syms:
            syms.append(sym)
            sym_dict[(sym, mag)] = sym
        elif (sym, mag) not in sym_dict:
            sym_dict[(sym, mag)] = sub_syms.pop()
        else:
            pass
    new_sym = ''
    for sym, mag in zip(symbols, magmoms):
        new_sym += sym_dict[(sym, mag)]
    new_atoms = atoms.copy()
    new_atoms.set_chemical_symbols(new_sym)

    return new_atoms, sym_dict


def rev_ref_atoms(atoms, sym_dict):
    rev_dict = {}
    for key in sym_dict:
        rev_dict[sym_dict[key]] = key
    old_symbols = []
    old_magmons = []
    for sym in atoms.get_chemical_symbols():
        old_symbols.append(rev_dict[sym][0])
        old_magmons.append(rev_dict[sym][1])
    old_atoms = atoms.copy()
    old_atoms.set_chemical_symbols(old_symbols)
    old_atoms.set_initial_magnetic_moments(old_magmons)
    return old_atoms


def find_primitive(atoms, symprec=1e-4):
    """
    find the primitive cell withh regard to the magnetic structure. a atoms object is returned.
    """
    #atoms_mag,sym_dict=ref_atoms_mag(atoms)
    cell, scaled_pos, chem_nums = spglib.find_primitive(atoms, symprec=symprec)
    chem_sym = 'H%d' % (len(chem_nums))
    new_atoms = Atoms(chem_sym)

    new_atoms.set_atomic_numbers(chem_nums)
    new_atoms.set_cell(cell)
    new_atoms.set_scaled_positions(scaled_pos)
    #new_atoms=rev_ref_atoms(new_atoms,sym_dict)
    return new_atoms


def find_primitive_mag(atoms, symprec=1e-4):
    """
    find the primitive cell withh regard to the magnetic structure. a atoms object is returned.
    """
    atoms_mag, sym_dict = ref_atoms_mag(atoms)
    cell, scaled_pos, chem_nums = spglib.find_primitive(
        atoms_mag, symprec=symprec)
    chem_sym = 'H%d' % (len(chem_nums))
    new_atoms = Atoms(chem_sym)

    new_atoms.set_atomic_numbers(chem_nums)
    new_atoms.set_cell(cell)
    new_atoms.set_scaled_positions(scaled_pos)
    new_atoms = rev_ref_atoms(new_atoms, sym_dict)
    return new_atoms


def get_refined_atoms(atoms):
    """
    using spglib.refine_cell, while treat atoms with different magnetic moment as different element.
    """
    atoms_mag, sym_dict = ref_atoms_mag(atoms)
    cell, scaled_pos, chem_nums = spglib.refine_cell(atoms_mag, symprec=1e-4)
    chem_sym = 'H%d' % (len(chem_nums))
    new_atoms = Atoms(chem_sym)

    new_atoms.set_atomic_numbers(chem_nums)
    new_atoms.set_cell(cell)
    new_atoms.set_scaled_positions(scaled_pos)
    new_atoms = rev_ref_atoms(new_atoms, sym_dict)
    return new_atoms


def to_smallest_positive_pos(pos):
    """
    make scaled positions in [0,1)
    """
    return np.mod(pos, 1.000)


def normalize(atoms, set_origin=False):
    """
    set the most near 0 to 0
    make cell -> cellpar ->cell
    make all the scaled postion between 0 and 1
    """
    newatoms = atoms.copy()
    newatoms = force_near_0(newatoms)
    if set_origin:
        positions = newatoms.get_positions()
        s_positions = sorted(positions, key=np.linalg.norm)
        newatoms.translate(-s_positions[0])

    positions = newatoms.get_scaled_positions()
    newcell = cellpar_to_cell(cell_to_cellpar(newatoms.get_cell()))
    newatoms.set_cell(newcell)

    for i, pos in enumerate(positions):
        positions[i] = np.mod(pos, 1.0)

    newatoms.set_scaled_positions(positions)
    return newatoms


def shift_origin(atoms, orig_atom):
    """
    shift the atoms so that the orig_atom is at (0,0,0)
    """
    sdict = symbol_number(atoms)
    shift = -atoms.get_positions()[sdict[orig_atom]]
    atoms.translate(shift)
    return atoms


def to_same_cell(pos, ref_pos):
    """
    make every ion  position in pos in the same cell as ref_pos
    pos, ref_pos: array or list of positions.
    """
    pos = array(pos)
    for i, position in enumerate(pos):
        for j, xj in enumerate(position):
            if xj - ref_pos[i][j] > 0.5:
                pos[i][j] -= 1.0
            elif xj - ref_pos[i][j] < -0.5:
                pos[i][j] += 1.0
    return pos


def force_near_0(atoms, max=0.93):
    """
    force the atom near the "1" side (>max) to be near the 0 side
    """
    positions = atoms.get_scaled_positions()
    new_positions = []
    for pos in positions:
        new_pos = [(x if x < max else x - 1) for x in pos]
        new_positions.append(new_pos)
    atoms.set_scaled_positions(new_positions)
    return atoms


def force_near_1(atoms, min=0.03):
    """
    force the atom near the "0" side (<min) to be near the 1 side
    """
    positions = atoms.get_scaled_positions()
    new_positions = []
    for pos in positions:
        new_pos = [(x if x > min else x + 1) for x in pos]
        new_positions.append(new_pos)
    atoms.set_scaled_positions(new_positions)
    return atoms


def calc_bands(atoms, kpts, calculator='vasp'):
    """
    calcultate band structure.
    """
    if calculator == 'vasp':
        atoms.calc.set(nsw=0, ibrion=-1, icharg=11)
        atoms.calc.set(kpts=kpts, reciprocal=True)
        eigenvals = atoms.calc.get_eigenvalues()
        kpoints = atoms.calc.get_ibz_k_points()
        efermi = atoms.calc.get_fermi_level()
        return kpoints, eigenvals, efermi


def plot_bands(atoms,
               sp_kpts,
               kpts_names=None,
               nkpts=60,
               calculator='vasp',
               window=None,
               output_filename=None,
               show=False,
               spin=0):
    """
    plot the bands.
    window: (Emin,Emax), the range of energy to be plotted in the figure.
    speicial_kpts_name
    """
    kpts, xcords, sp_xcords = get_bandpath(sp_kpts, atoms.get_cell(), nkpts)
    kpoints, eigenvalues, efermi = calc_bands(atoms, kpts)
    mycalc = atoms.calc
    if output_filename is None:
        output_filename = 'band.png'
    plt.clf()
    if window is not None:
        plt.ylim(window[0], window[1])
    if not mycalc.get_spin_polarized():
        eigenvalues = []
        for ik in range(nkpts):
            eigenvalues_ik = np.array(mycalc.get_eigenvalues(kpt=ik))
            eigenvalues.append(eigenvalues_ik)
        eigenvalues = np.array(eigenvalues)
        for i in range(mycalc.get_number_of_bands()):
            band_i = eigenvalues[:, i] - efermi
            plt.plot(xcords, band_i)
    else:
        eigenvalues = []
        for ik in range(nkpts):
            eigenvalues_ik = np.array(
                mycalc.get_eigenvalues(kpt=ik, spin=spin))
            eigenvalues.append(eigenvalues_ik)
        eigenvalues = np.array(eigenvalues)
        for i in range(mycalc.get_number_of_bands()):
            band_i = eigenvalues[:, i] - efermi
            plt.plot(xcords, band_i)
    plt.xlabel('K-points')
    plt.ylabel('$Energy-E_{fermi} (eV)$')

    plt.axhline(0, color='black', linestyle='--')
    if kpts_names is not None:
        plt.xticks(sp_xcords, kpts_names)
    if output_filename is not None:
        plt.savefig(output_filename)
    if show:
        plt.show()


def pos_equal(pos1, pos2, thr=0.01):
    """
    Test whether two postions are equivalent.
    """
    eqv_pos1 = [p % 1.0 for p in pos1]
    eqv_pos2 = [p % 1.0 for p in pos2]
    dist = [abs(p1 - p2) for p1, p2 in zip(eqv_pos1, eqv_pos2)]
    is_equal = all([x < thr for x in dist])
    return is_equal


def translation(atoms, trans_vector):
    """
    translate atoms.
    """
    trans_vector = np.asarray(trans_vector)
    positions = atoms.get_positions()
    new_positions = [np.asarray(pos) + trans_vector for pos in positions]
    new_atoms = atoms.copy()
    new_atoms.set_positions(new_positions)
    return new_atoms


def pos_in_cell(pos, cell, shift=-0.002):
    """
    is the position (not scaled) inside the cell.
    pos: position
    cell: a 3*3 array define the cell.
    shift: shift< scaled_pos <= 1.0+shift . To avoid bugs caused by floating point error.
    """
    cell = np.asarray(cell)
    pos = np.asarray(pos)
    scaled_pos = np.dot(pos, np.linalg.inv(cell))

    is_in = all(scaled_pos <= 1.0 + shift) and all(shift < scaled_pos)
    #print scaled_pos
    #if is_in:
    #    print scaled_pos
    return is_in


def pos_to_scaled_pos(pos, cell):
    """
    Input:
    pos: vector 3d
    cell: 3*3 matrix
    position -> scaled position
    """
    pos = np.asarray(pos)
    cell = np.asarray(cell)
    scaled_pos = np.dot(pos, np.linalg.inv(cell))
    return scaled_pos


def scaled_pos_to_pos(scaled_pos, cell):
    """
    scaled_pos->pos
    scaled_pos: 3d vector
    cell: 3*3 matrix
    """
    scaled_pos = np.asarray(scaled_pos)
    cell = np.asarray(cell)
    pos = np.dot(scaled_pos, cell)
    return pos


def mirror(atoms, direction=2, center=0.5):
    """
    eg. direction=2,center=0.5 :atoms postion z-> 1-z
    direction: 0,1,2
    center:
    """
    new_atoms = atoms.copy()
    scaled_positions = atoms.get_scaled_positions()

    def m_pos(pos):
        pos[direction] = center * 2 - pos[direction]
        return pos

    new_scaled_positions = []
    for pos in scaled_positions:
        new_scaled_positions.append(m_pos(pos))
    new_atoms.set_scaled_positions(new_scaled_positions)
    return new_atoms


def cut_lattice(atoms, new_cell, nrepeat=8):
    """
    set atoms in new cell. The method is as follows.
     1. generate a (nrepeat,nrepeat,nrepeat) repeat of atoms centered at (0,0,0) and keep those inside the new_cell
     2. remove the duplicates because of the floating point error.Maybe this can be avoid by using shift. Not tested.
    """
    r_atoms = atoms.copy()
    cell = atoms.get_cell()
    #print cell
    vec = np.dot(np.array([1, 1, 1]), cell) * (-nrepeat / 2.0)
    r_atoms = r_atoms.repeat([nrepeat, nrepeat, nrepeat])
    #print r_atoms
    r_atoms = translation(r_atoms, vec)
    #print len( [atom.index for atom in r_atoms if pos_in_cell(atom.position,new_cell)])
    del r_atoms[[
        atom.index for atom in r_atoms
        if not pos_in_cell(atom.position, new_cell)
    ]]
    r_atoms.set_cell(new_cell)
    return r_atoms


def set_element_mag(atoms, element, magmoms):
    """
    set the magetic moments of specific element in the atoms.
    """
    try:
        mags = atoms.get_initial_magnetic_moments()
    except Exception as exp:
        mags = np.zeros(len(atoms))
    sym_dict = symbol_number(atoms)
    for i, mag in enumerate(magmoms):
        mags[sym_dict['%s%d' % (element, i + 1)]] = mag

    atoms.set_initial_magnetic_moments(mags)
    return atoms


def test():
    myatoms = gen_STO()
    print(find_sym(myatoms))
    #print myatoms.get_chemical_symbols()
    #natoms,d=ref_atoms_mag(myatoms)
    ##print ref_atoms_mag(myatoms)[0].get_chemical_symbols()
    #old_atoms=rev_ref_atoms(natoms,d)
    #print old_atoms.get_chemical_symbols()
    #print get_prim_atoms(myatoms)
    new_atoms = find_primitive_mag(myatoms)
    #print new_atoms.get_chemical_symbols()
    print(new_atoms.get_cell())
    print(new_atoms.get_scaled_positions())
    print(new_atoms.get_volume())
    print(spglib.get_spacegroup(new_atoms))
    #write('POSCAR',new_atoms,sort=True,vasp5=True)
    natoms = normalize(new_atoms)
    print(natoms.get_positions())
    print(natoms.get_volume())
    print(cell_to_cellpar(natoms.get_cell()))


def vesta_view(atoms):
    if not os.path.exists('/tmp/vesta_tmp'):
        os.mkdir('/tmp/vesta_tmp')
    my_write_vasp('/tmp/vesta_tmp/POSCAR', atoms)
    os.system('vesta /tmp/vesta_tmp/POSCAR')


def set_atoms_select_dynamics(atoms, selective_dynamics=None):
    natom = len(atoms)
    if selective_dynamics is None and not 'selective_dynamics' in atoms.__dict__:
        atoms.select_dynamics = np.ones([natom, 3], dtype='float')
    else:
        atoms.select_dynamics = selective_dynamics
    return atoms


def set_atom_select_dynamics(atoms, iatom, m):
    atoms = set_atoms_select_dynamics(atoms)
    atoms.selective_dynamics[iatom] = m
    return atoms


def relax_cell(atoms,
               max_step=60,
               thr=0.01,
               logfile='relaxation.log',
               mask=[0, 0, 1, 1, 1, 0],
               optimizer='BFGS',
               restart='relax_restart.pckl'):
    """
    mask: [xx,yy,zz,xz,yz,xy]
    """

    from ase.constraints import StrainFilter

    sf = StrainFilter(atoms, mask=mask)
    qn = BFGS(sf, logfile='relaxation.log', restart=restart)
    qn.run(fmax=thr, steps=max_step)
    return atoms


def vasp_relax_cell(atoms,
                    max_step=60,
                    thr=0.01,
                    logfile='relaxation.log',
                    mask=[0, 0, 1, 1, 1, 0],
                    optimizer='BFGS',
                    restart='relax_restart.pckl'):
    """
    relax cell which are far from equibrium.
    """

    from ase.constraints import StrainFilter

    calc = atoms.calc

    nelmdl = calc.int_params['nelmdl']
    ibrion = calc.int_params['ibrion']
    sigma = calc.float_params['sigma']
    if sigma is None:
        sigma = 0.1
    ediff = calc.exp_params['ediff']
    if ediff is None:
        ediff = 1e-4
    ediffg = calc.exp_params['ediffg']
    if ediffg is None:
        ediffg = -0.01

    ldipol = calc.bool_params['ldipol']
    if ldipol is None:
        ldipol = False
    nsw = calc.int_params['nsw']

    #first do this
    calc.set(
        nelmdl=6,
        nelmin=-9,
        ediff=1e-3,
        ediffg=-0.3,
        nsw=20,
        ibrion=2,
        sigma=sigma * 3,
        ldipol=False)
    atoms.set_calculator(calc)
    sf = StrainFilter(atoms, mask=mask)
    if optimizer == 'BFGSLineSearch':
        qn = BFGSLineSearch(
            sf, logfile='relaxation.log', use_free_energy=False)
    else:
        qn = BFGS(sf, logfile='relaxation.log')

    qn.run(fmax=0.3, steps=5)

    # then increase the accuracy.
    calc.set(
        nelmdl=nelmdl,
        nelmin=5,
        ediff=ediff,
        ediffg=ediffg,
        ibrion=ibrion,
        sigma=sigma,
        ldipol=ldipol,
        nsw=nsw)
    calc.set(istart=1)
    atoms.set_calculator(calc)

    sf = StrainFilter(atoms, mask=mask)
    qn = BFGS(sf, logfile='relaxation.log', restart=restart)
    qn.run(fmax=0.01, steps=max_step)
    return atoms


def mycut(atoms,
          a_atom,
          b_atom,
          c_atom,
          origo_atom,
          nlayers=None,
          extend=1.0,
          tolerance=0.01,
          maxatoms=None):
    """
    atoms: atoms
    a_atoms,b_atoms,c_atoms: the symol_number of the atoms (a_atom-origo_atom)-> a; ...
    origo_atom: the atom at the origin.
    other parameters are as ase.io.geometry.cut
    """
    symdict = symbol_number(atoms)
    origo = symdict[origo_atom]
    a = symdict[a_atom]
    b = symdict[b_atom]
    c = symdict[c_atom]
    atoms = cut(atoms,
                a=a,
                b=b,
                c=c,
                clength=None,
                origo=origo,
                nlayers=None,
                extend=1.0,
                tolerance=0.01,
                maxatoms=None)
    return atoms


def cut_z(atoms, symnum1, symnum2, shift):
    """
    cut along the z direction.
    """
    symdict = symbol_number(atoms)
    z1 = atoms.get_positions()[symdict[symnum1]][2] - shift
    z2 = atoms.get_positions()[symdict[symnum2]][2] - shift
    c = atoms.get_cell()[2][2]
    dz = z2 - z1
    newatoms = cut(atoms, clength=dz, origo=(0, 0, z1 / c))
    newatoms.translate([0, 0, -shift])
    return newatoms


def swap_axis(atoms, a1, a2):
    """
    swap axis of atoms
    """
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    newpos = []
    cell[[a1, a2]] = cell[[a2, a1]]
    for pos in positions:
        apos = pos
        apos[a1], apos[a2] = apos[a2], apos[a1]
        newpos.append(apos)
    atoms.set_cell(cell)
    atoms.set_scaled_positions(newpos)
    atoms = normalize(atoms, set_origin=True)
    return atoms


def swap_yz(atoms):
    """
    swap the y and z cell vector
    """
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    newpos = []
    x, y, z = cell
    cell = x, z, y
    for pos in positions:
        x, y, z = pos
        newpos.append([x, z, y])
    atoms.set_cell(cell)
    atoms.set_scaled_positions(newpos)
    atoms = normalize(atoms, set_origin=True)
    return atoms


def set_substrate(atoms,
                  a=None,
                  b=None,
                  c=None,
                  angle_ab=90.0,
                  all_angle_90=False,
                  m=1,
                  fix_volume=False):
    """
    set atoms cellpars to fit the substrate cellpars.
    a,b , angle_ab are the  the inplane cellpars of the substrate
    all_angle_90: if True,all the angles will be set to 90
    m is the multiplier. a*m b*m
    fix_volume: whether to set c so that the volume is unchanged.
    Note that this is not always really the case if angles of a-b plane and c is changed.
    """
    cellpars = cell_to_cellpar(atoms.get_cell())
    print(a,b,c)

    a0, b0, c0 = cellpars[0:3]
    if a is not None:
        cellpars[0] = a * m
    if b is not None:
        cellpars[1] = b * m
    if c is not None:
        cellpars[2] = c * m
    if angle_ab is not None:
        cellpars[5] = angle_ab
    if all_angle_90:
        cellpars[3:] = [90, 90, 90]

    print(cellpars)
    if fix_volume:
        ab = cellpars[5]
        c = (a0 * b0 * c0 * math.sin(math.radians(ab)) /
             (a * b * m * m * math.sin(math.radians(angle_ab))))
        cellpars[2] = c

    cell = cellpar_to_cell(cellpars)
    print(cell)
    spos = atoms.get_scaled_positions()

    atoms.set_cell(cell)
    atoms.set_scaled_positions(spos)
    return atoms


def split_layer(atoms, thr=0.03, direction=2, sort=True, return_pos=False):
    """
    split atoms into layers
    Parameters:
      thr: max distance in direction from atoms in same layer.
      direction: 0 | 1| 2
      sort: whether to sort the layers by posiontion.
      return_pos: whether to return the positions of each atom.
    Returns:
       A list of symnum lists, each is the symnum of a layer.
       [['Ni1','O2'],['']]
    """
    for i in [0, 1, 2]:
        if i != direction:
            if atoms.get_cell()[direction][i] > 0.5:
                raise NotImplementedError(
                    "cellparameters should be orthgonal in the direction")
    atoms = force_near_0(atoms)
    z = cell_to_cellpar(atoms.get_cell())[direction]
    positions = [pos[direction] for pos in atoms.get_positions()]

    def is_near(pos1, pos2):
        if abs(pos1 - pos2) % z < thr or z - abs(pos1 - pos2) % z < thr:
            return True
        else:
            return False

    symdict = symbol_number(atoms)

    layer_indexes = []
    layer_poses = []
    layer_symnums = []
    for i, pos in enumerate(positions):
        got_layer = False
        if layer_indexes == []:
            layer_indexes.append([i])
            layer_poses.append([pos])
            layer_symnums.append([list(symdict.keys())[i]])
            continue

        for ind, layer_ind_pos in enumerate(zip(layer_indexes, layer_poses)):

            lp = np.average(layer_ind_pos[1])
            #print "ind_pos",layer_ind_pos[1]
            if is_near(pos, lp):
                print("got: %s" % ind, pos, lp)
                got_layer = True
                index = ind
                break
        if got_layer:
            layer_indexes[index].append(i)
            layer_poses[index].append(pos)
            layer_symnums[index].append(list(symdict.keys())[i])
        else:
            layer_indexes.append([i])
            layer_poses.append([pos])
            layer_symnums.append([list(symdict.keys())[i]])
    if sort:
        sort_i = sorted(list(range(len(layer_poses))), key=layer_poses.__getitem__)
        layer_symnums = [layer_symnums[i] for i in sort_i]
        layer_poses = [layer_poses[i] for i in sort_i]
    if return_pos:
        return layer_symnums, layer_poses
    else:
        return layer_symnums


def substitute_atoms(atoms, slist_from, symbols_to):
    """
    substitute the symbols in atoms.
    slist_from: the sym_nums to be substitute. eg.['Ca1','Ca5','O3']
    slist_to: the chemical symbols to be. eg. ['Fe','Fe','O']
    """
    sdict = symbol_number(atoms)
    syms = atoms.get_chemical_symbols()
    for s0, s1 in zip(slist_from, symbols_to):
        syms[sdict[s0]] = s1
    atoms.set_chemical_symbols(syms)
    return atoms


def expand_bonds(atoms, center, target, add_length=0.1, maxlength=3.0):
    """
    expand the bond lengths.
    :param atoms: atoms
    eg. atoms=expand(atoms, 'Fe1','O', add_length=0.1 ,maxlength=3.0)
    """
    sdict = symbol_number(atoms)
    for symnum in sdict:
        if symnum_to_sym(symnum) == target:
            l = atoms.get_distance(sdict[center], sdict[symnum], mic=True)
            if l <= maxlength:
                atoms.set_distance(
                    sdict[center],
                    sdict[symnum],
                    l + add_length,
                    fix=0,
                    mic=True)
    return atoms


def gen_disped_atoms(atoms, sym, distance, direction='all'):
    """
    shift one of the atoms. Often used for calculating the Born Effective Charge. sym: like 'Fe1'. direction can be 0|1|2|all. If direction is 'all', return a list of displaced structures with disp along x, y, z.
    """
    sdict = get_symdict(atoms=atoms)
    poses = atoms.get_positions()
    if direction in [0, 1, 2]:
        d = np.zeros(3, dtype=float)
        d[direction] = 1.0
        disp = distance * d
        poses[sdict[sym]] += disp
        natoms = copy.deepcopy(atoms)
        natoms.set_positions(poses)
        return natoms
    elif direction == 'all':
        return [
            gen_disped_atoms(atoms, sym, distance, direction=direct)
            for direct in [0, 1, 2]
        ]
    else:
        raise NotImplementedError



if __name__ == '__main__':
    test()
