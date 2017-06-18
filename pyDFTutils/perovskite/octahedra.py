#! /usr/bin/env python

from ase.utils.geometry import rotate
from ase.io import read
from ase.geometry.cell import cellpar_to_cell, cell_to_cellpar
from pyDFTutils.ase_utils.symbol import get_symdict,symbol_number,symnum_to_sym
from pyDFTutils.ase_utils.ase_utils import scaled_pos_to_pos,force_near_0,pos_to_scaled_pos
import numpy as np
from collections import OrderedDict,Iterable
from ase.calculators.neighborlist import NeighborList

def distance(a1,a2):
    """
    get the distance of a0 and a1, they are not necissarily in one Atoms object. Note that this is different from the Atoms.get_distance.
    a1 and a2 are atom projects
    """
    pos1=a1.position
    pos2=a2.position
    d=np.linalg.norm(pos1-pos2)
    return d


def vec_ang(a,b):
    """angle of two vectors, 0 ~ 180"""
    return np.arccos(np.inner(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))/np.pi*180

def atom_angle(a1,a2,a3):
    """
    angle a1-a2-a3
    a1 a2 a3 are atom objects
    """
    vec1=a2.position-a1.position
    vec2=a2.position-a3.position
    return vec_ang(vec1,vec2)


def even_or_odd_path(atoms,from_symnum,node_sym_list,to_symnum_list=None,first_neighbor_min=2.5,first_neighbor_max=4.5):
    """
    Find whether the distance of the atoms with symnum in to_list to the atom(from_symnum) is even or odd.
    The distance to the first neighbor is 1. The 1st neighbor's 1st neighbor is 2. etc....
    Args:
      atoms:
      from_symnum
      to_sym_list: The symbol of the atoms, eg:['Fe','Ni' ], Note the from_symnum should be start with the symbol in this list
      to_symnum_list: The symbol_number of the atoms, eg:['Fe1','Fe2','Ni1'].Only one of the to_sym_list and to_symnum_list should should be specified.
      first_neighbor_min/max: The min/max distance to the first neighbor
    Returns:
      a dict.  The values are 1/-1. 1 is odd, -1: even eg {'Fe1':1,'Ni1':-1}
    """
    #1. Get the first neighbor list
    symdict=symbol_number(atoms)
    symnums=list(symdict.keys())
    node_list=[]
    for s in symnums:
        #print s
        if symnum_to_sym(s) in node_sym_list:
            node_list.append(s)
    if not from_symnum in node_list:
        raise Exception('from_symnum should be of one of the nodes')
    positions=atoms.get_positions()
    node_positions=[positions[symdict[i]] for i in node_list]
    from scipy.sparse import csr_matrix
    N=len(node_list)
    #dmat=csr_matrix((N,N))
    row=[]
    col=[]
    val=[]
    dist=lambda pos1,pos2 :np.linalg.norm(np.asarray(pos1)-np.asarray(pos2))
    for i,pi in enumerate(node_positions):
        for j,pj in enumerate(node_positions):
            if i==j:
                row.append(i)
                col.append(j)
                val.append(0)
            else:
                #print pi,pj
                #print dist(pi,pj)
                if first_neighbor_min <dist(pi,pj)<first_neighbor_max:
                    row.append(i)
                    col.append(j)
                    val.append(1)
    dmat=csr_matrix((val,(row,col)),shape=(N,N))
    #print dmat
    from scipy.sparse.csgraph import dijkstra
    path_mat=dijkstra(dmat,directed=False,unweighted=True)

    i_node=node_list.index(from_symnum)
    def even_odd(x):
        if int(x)%2==0:
            return 1
        else:
            return -1
    return node_list,[even_odd(x) for x in path_mat[i_node]]

def rotation_angle_about_axis(vec0,from_vec,axis_vec):
    """
    rotate from from_vec to vec0 about the axis. (there may be other actions), get the rotation angle.
    """
    v1=np.cross(vec0,axis_vec)/np.linalg.norm(axis_vec)

    f1=np.cross(from_vec,axis_vec)/np.linalg.norm(axis_vec)
    #angle=np.arcsin(np.inner(axis_vec,np.cross(v1,f1))/(np.linalg.norm(v1)*np.linalg.norm(f1))*np.linalg.norm(axis_vec))/np.pi*180
    angle=np.arccos(np.inner(v1,f1)/(np.linalg.norm(v1)*np.linalg.norm(f1)))/np.pi*180
    if np.inner(np.cross(v1,f1),axis_vec)<0:
        angle=-angle
    return angle



class octahedra():
    """
    deal with octahedra in ase.Atoms
    """
    def __init__(self):

        self.center=None
        self.vertexes=[]

        self.vertex_symbol=None
        self.max_distance=None

        self.number_of_vertexes=0

        self.upper=None
        self.lower=None
        self.left=None
        self.right=None
        self.forward=None
        self.backward=None

        x=(1,0,0)
        y=(0,1,0)
        z=(0,0,1)

        self.x=np.array(x)
        self.y=np.array(y)
        self.z=np.array(z)

        self.filename=None
        self.atoms=None
        self.sym_dict=None

        self.direction_list=['forward','backward','right','left','upper','lower']


    def get_vertex(self,direction):

        return self.__dict__[direction]


    def set_axis(self,x=(1,0,0), y=(0,1,0),z=(0,0,1),axis_type=None):
        """
        axis_type: default None, if not None: 'rotate45_xy': means x,y are 45 degree rotated about z
        """
        if axis_type is None:
            self.x=np.array(x)
            self.y=np.array(y)
            self.z=np.array(z)
        elif axis_type=='rotate45_xy':
            self.x=np.array([1,-1,0])
            self.y=np.array([1,1,0])
            self.z=np.array([0,0,1])
        else:
            raise ValueError('axis_type %s invalid.'%axis_type)

    def get_octahedra(self,atoms,center_atom,atom_symbol,max_distance,var_distance=False,do_force_near_0=True,repeat=True):
        """
        atoms: the atoms
        center_atom: a atom object, or sym_num
        atoms_symbol: symbol of atom in the vertexes
        max_distance: max distance
        var_distance: variable distance.
        """
        self.sym_dict=get_symdict(filename=None,atoms=atoms)
        if isinstance(center_atom,str):
            center_atom=atoms[self.sym_dict[center_atom]]


        oatoms=atoms.copy()
        cell=oatoms.get_cell()
        if do_force_near_0:
            oatoms=force_near_0(oatoms)


            cspos=pos_to_scaled_pos(center_atom.position,cell)
            cpos=[x-1 if x>0.9 else x for x in cspos]
            center_atom.position=scaled_pos_to_pos(cpos,cell)

        self.atoms=oatoms

        self.center=center_atom
        self.vertex_symbol=atom_symbol

        if repeat:
            ratoms=oatoms.copy()
            ratoms=ratoms.repeat([3,3,3])
            ratoms.translate(scaled_pos_to_pos((-1,-1,-1),cell))
        else:
            ratoms=oatoms.copy()

        self.vertexes=[]
        for atom in ratoms:
            if atom.symbol==atom_symbol and distance(center_atom,atom)<=max_distance:
                self.vertexes.append(atom)
                if len(self.vertexes)==6:
                    break
        if var_distance and len(self.vertexes)<6:
            #print("max_distance too small,tring to increase by 0.06")
            self.get_octahedra(self.atoms,center_atom,atom_symbol,max_distance+0.06)
        elif var_distance and len(self.vertexes)>6:
            #print("max_distance too small,tring to decrease by 0.05")
            self.get_octahedra(self.atoms,center_atom,atom_symbol,max_distance-0.05)
        else:
            pass
            #print("%d vertexes found,max distance is %s"%(len(self.vertexes),max_distance))

        self.number_of_vertexes=len(self.vertexes)
        self.max_distance=max_distance

        self.pair()
        self.get_bond_angle()
        self.get_bond_lengths()
        self.get_rotations()
        self.get_avg_rotations()

        return self.vertexes

    def read_octahedra(self,filename,center_sym_number,vertex_symbol,max_distance,var_distance=False):
        """
        read octahedra infomation from file filename.
        filename: filename
        center_sym_number: eg. 'Fe1'
        vertex_symbol: eg. 'O'
        max_distance : maximum distance between center and vertex
        var_distance: if max_distance too small/ large, find a better one. Use if you are sure there is really a octahedra.
        """
        sym_dict=get_symdict(filename=filename)
        self.filename=filename
        self.sym_dict=sym_dict

        atoms=read(filename)
        self.get_octahedra(atoms,atoms[sym_dict[center_sym_number]],vertex_symbol,max_distance,var_distance=var_distance)


    def pair(self,ang_thr=20):
        """
        divide the atoms to pairs ,(upper,lower) (left, right)...
        """

        cpos=self.center.position
        positions=[x.position for x in self.vertexes]
        ang_x=[vec_ang(pos-cpos,self.x) for pos in positions]

        ang_y=[vec_ang(pos-cpos,self.y) for pos in positions]

        ang_z=[vec_ang(pos-cpos,self.z) for pos in positions]

        i_paired=[]
        for i in range(len(positions)):
            if ang_x[i]<ang_thr:
                i_paired.append(i)
                self.forward=self.vertexes[i]
            if ang_x[i]>180-ang_thr:
                i_paired.append(i)
                self.backward=self.vertexes[i]
            if ang_y[i]<ang_thr:
                i_paired.append(i)
                self.right=self.vertexes[i]
            if ang_y[i]>180-ang_thr:
                i_paired.append(i)
                self.left=self.vertexes[i]
            if ang_z[i]<ang_thr:
                i_paired.append(i)
                self.upper=self.vertexes[i]
            if ang_z[i]>180-ang_thr:
                i_paired.append(i)
                self.lower=self.vertexes[i]

        if len(i_paired)>self.number_of_vertexes:
            #print('warning ang_thr too large,reducing by 0.1')
            self.pair(ang_thr=ang_thr-0.1)
        elif len(i_paired)<self.number_of_vertexes:
            #print('warning ang_thr too large,increasing by 0.2')
            self.pair(ang_thr=ang_thr+0.2)
        self.vertexes=[self.forward,self.backward,self.left,self.right,self.upper,self.lower]

    def get_bond_lengths(self):
        """
        get the bond lengths between  the central atom and atoms on the vertexes
        """
        self.bond_lengths=[distance(x,self.center) for x in self.vertexes]
        return self.bond_lengths

    def get_vertex_diag_length(self):
        """
        get the distances of vertexes (right,left),(forward,backward),(up-down)
        """
        dy=distance(self.right,self.left)
        dx=distance(self.forward,self.backward)
        dz=distance(self.upper,self.lower)
        return [dx,dy,dz]

    def get_directions(self):
        """
        get the true x,y,z of octahedron.
        """
        x=self.forward.position-self.backward.position
        y=self.right.position-self.left.position
        z=self.upper.position-self.lower.position
        return [x,y,z]


    def get_rotations(self):
        """
        get the rotation angle about special axises.
        """
        self.rotation_angles=[]
        angle_x_about_z=rotation_angle_about_axis(self.forward.position-self.backward.position,self.x,self.z)


        angle_y_about_z=rotation_angle_about_axis(self.right.position-self.left.position,self.y,self.z)


        angle_x_about_y=rotation_angle_about_axis(self.forward.position-self.backward.position,self.x,self.y)


        angle_z_about_y=rotation_angle_about_axis(self.upper.position-self.lower.position,self.z,self.y)

        angle_y_about_x=rotation_angle_about_axis(self.right.position-self.left.position,self.y,self.x)

        angle_z_about_x=rotation_angle_about_axis(self.upper.position-self.lower.position,self.z,self.x)

        self.rotation_angles.append(angle_y_about_x)
        self.rotation_angles.append(angle_z_about_x)
        self.rotation_angles.append(angle_z_about_y)
        self.rotation_angles.append(angle_x_about_y)
        self.rotation_angles.append(angle_x_about_z)
        self.rotation_angles.append(angle_y_about_z)

        return self.rotation_angles

    def get_avg_rotations(self):
        """
        get averaged rotations about x,y,z respectively
        """
        self.get_rotations()
        self.rot_x=(self.rotation_angles[0]+self.rotation_angles[1])/2
        self.rot_y=(self.rotation_angles[2]+self.rotation_angles[3])/2
        self.rot_z=(self.rotation_angles[4]+self.rotation_angles[5])/2
        return [self.rot_x,self.rot_y,self.rot_z]

    def get_distortion(self):
        """
        The distortion of the MX6 octahedra is defined as:
        \delta d= 1/6 * sum{[(<M-x>-avg(<M_X>))/avg((M-X))]^2}
        i.e. \delta_d=variant(<M-x>)/avg(<M-X>)^2
        """
        self.get_bond_lengths()
        d=np.array(self.bond_lengths)
        avg_d=np.average(d)
        norm_d=d/avg_d
        self.distortion_factor=np.var(norm_d)
        return self.distortion_factor

    def get_bond_info(self,show=True):
        self.get_distortion()
        print(("The bond lengths: %s"%self.bond_lengths))
        print(("Average bond length: %s"%np.average(self.bond_lengths)))
        print(("Distortion factor: %s"%self.distortion_factor))




    def get_bond_angle(self):
        bond_angle1=atom_angle(self.right,self.center,self.left)
        bond_angle2=atom_angle(self.forward,self.center,self.backward)
        bond_angle3=atom_angle(self.upper,self.center,self.lower)
        self.bond_angles=[bond_angle1,bond_angle2,bond_angle3]
        return self.bond_angles


    def find_neighbouring_octahedron(self,direction='upper',target_symbol_list=None,do_force_near_0=False):
        """
        args:
        direction: the directio of the target to be found.
        target_symbol_list: the center of the octahdron to be found, should be a tuple. if None, the same as the center of the current octahedron
        eg. ('Fe','Ti')
        """

        if target_symbol_list is None:
            target_symbol_list=(self.center.symbol,)
        else:
            target_symbol_list=tuple(target_symbol_list) # -> a tuple because .startswith shoulcd be with tuples but lists.

        pdict={'upper':'lower','left':'right','forward':'backward','lower':'upper','right':'left','backward':'forward'}
        if do_force_near_0:
            natoms=force_near_0(self.atoms,max=0.9)
        else:
            natoms=self.atoms

        target=None

        ratoms=natoms.copy()
        cell=ratoms.get_cell()
        ratoms=ratoms.repeat([3,3,3])
        ratoms.translate(scaled_pos_to_pos((-1,-1,-1),cell))

        rsym_dict=symbol_number(ratoms)
        for sym_num in rsym_dict:
            target_center_atom=ratoms[rsym_dict[sym_num]]
            if sym_num.startswith(target_symbol_list) and distance(self.center,target_center_atom)<=2*self.max_distance:

                t_octa=octahedra()
                t_octa.set_axis(x=self.x,y=self.y,z=self.z)

                try:
                    t_octa.get_octahedra(ratoms,center_atom=ratoms[rsym_dict[sym_num]],atom_symbol=self.vertex_symbol,max_distance=self.max_distance,repeat=False,do_force_near_0=False)
                    #print sym_num, t_octa.number_of_vertexes
                    nm=np.linalg.norm(self.__dict__[direction].position- t_octa.__dict__[pdict[direction]].position)
                    #print nm
                    if nm<0.1:
                        target= t_octa
                        break
                        #print sym_num
                except Exception:
                    pass
                    #print( 'exception:',exc)

        """
        for sym_num in self.sym_dict:
            if sym_num.startswith(target_symbol_list):
                t_octa=octahedra()
                t_octa.set_axis(x=self.x,y=self.y,z=self.z)

                try:
                    t_octa.get_octahedra(natoms,center_atom=self.atoms[self.sym_dict[sym_num]],atom_symbol=self.vertex_symbol,max_distance=self.max_distance)
                    #t_octa.read_octahedra(self.filename,center_sym_number=sym_num,vertex_symbol=self.vertex_symbol,max_distance=self.max_distance)
                    if self.__dict__[direction].index == t_octa.__dict__[pdict[direction]].index:
                        target= t_octa
                        #print sym_num
                except Exception as exc:
                    print exc
                    pass
                    #print( 'exception:',exc)
        """
        if target is None:
            raise Exception("No neighbour found")
        else:
            #print "Found"
            pass
        return target

    def get_vertex_angle(self,direction='upper',target_symbol_list=None,do_force_near_0=False):
        """
        args: Find the neigbouring octahedron and get the center-vertex-center angle.
        """
        try:
            noct=self.find_neighbouring_octahedron(direction=direction,target_symbol_list=target_symbol_list,do_force_near_0=do_force_near_0)
        except Exception as excp:
            print(excp)
            noct=None
        return atom_angle(self.center,self.get_vertex(direction),noct.center)

    def get_vertex_angles(self,target_symbol_list=None,do_force_near_0=False):
        """
        get vertex angles of all six directions. forward, backward,right, left, upper lower.
        """
        return OrderedDict([(d,self.get_vertex_angle(direction=d,target_symbol_list=target_symbol_list,do_force_near_0=do_force_near_0)) for d in self.direction_list])


def get_octahedron(atoms,center_sym_number,vertex_symbol,max_distance, axis_type=None,x=(1,0,0),y=(0,1,0),z=(0,0,1), var_distance=False):
    """
    axis_type: default None, if not None: 'rotate45_xy': means x,y are 45 degree rotated about z
    """
    octa=octahedra()
    octa.set_axis(x=x,y=y,z=z,axis_type=axis_type)
    octa.get_octahedra(atoms,center_sym_number,vertex_symbol,max_distance,var_distance=var_distance)
    return octa



def octa_chain(start_octahedron,direction='upper',target_symbol_list=None):
    """
    generator of a chain of octahedra.
    """
    octa=start_octahedron
    while octa is not None:
        yield octa
        octa=octa.find_neighbouring_octahedron(direction=direction,target_symbol_list=target_symbol_list)

def rotate_atoms_with_octahedra(atoms,octa):
    """
    rotate atoms so that the x,y,z axis are along the octahedra axises.
    if octahedra is a list of octahedras, use the average value.
    """
    if isinstance(octa,octahedra):
        x,y,z=octa.get_directions()
        rotate(atoms,x,[1,0,0],y,[0,1,0],rotate_cell=True)
    elif isinstance(octa, Iterable):
        xs=[]
        ys=[]
        for oc in octa:
            x,y,z=oc.get_directions()
            xs.append(np.array(x))
            ys.append(np.array(y))
        avg_x=np.average(xs,axis=0)
        avg_y=np.average(ys,axis=0)
        rotate(atoms,avg_x,[1,0,0],avg_y,[0,1,0],rotate_cell=True)
    else:
        raise ValueError("octa should be a octahedra or a iterable object ")
    return atoms

def rotate_atoms_with_octahedra_average(atoms,octahedras):
    """
    rotate atoms so that the x,y,z axis are along the average octahedra axises.
    """


def write_all_octahedra_info(atoms,center_symbol_list,vertex_symbol,max_distance,output='octa_info.csv',axis_type=None,x=(1,0,0),y=(0,1,0),z=(0,0,1), var_distance=False):
    """
    write all octahedra info into a file.
    """
    symdict=symbol_number(atoms)
    myfile=open(output,'w')
    for symnum in symdict:
        sym=symnum_to_sym(symnum)
        print(sym)
        if sym in center_symbol_list:
            print(sym)
            this_oct=get_octahedron(atoms,symnum,vertex_symbol,max_distance,axis_type=axis_type,x=x,y=y,z=z,var_distance=var_distance)
            avg_rotations=this_oct.get_avg_rotations()
            bond_lengths=this_oct.get_bond_lengths()
            avg_bond_length=np.average(bond_lengths)
            distortion_factor=this_oct.get_distortion()
            vertex_angles=this_oct.get_vertex_angles(target_symbol_list=center_symbol_list)

            myfile.write('Info of %s :\n'%symnum)
            myfile.write("avg_rot: %s\n"%('\t'.join([str(a) for a in avg_rotations])))
            myfile.write("avg_bond: %s\n"%avg_bond_length)
            myfile.write("distrotion factor: %s\n"%distortion_factor)
            myfile.write("vertex lengths: %s\n"%('\t'.join([str(a) for a in bond_lengths])))
            myfile.write("vertex angles: %s\n\n"%('\t'.join([str(a) for a in list(vertex_angles.values())])))
    myfile.close()

def main():
    import argparse
    parser=argparse.ArgumentParser(description='write all the octahedra info')
    parser.add_argument('-f','--filename',type=str,help='POSCAR filename',default='POSCAR')
    parser.add_argument('-o','--output',help='output filename',default='octa_info.csv')
    parser.add_argument('-c','--center',nargs='+',type=str,help='octahedra symbol list',default='Mn')
    parser.add_argument('-v','--vertex',help='vertex symbol', default='O')
    parser.add_argument('-d','--distance',type=float,help='max distance', default=3.0)
    parser.add_argument('-a','--axis_type',help='axis type: None or rotate45_xy',default=None)
    parser.add_argument('-x','--xvec',help='x vector',nargs='+',type=float,default=(1,0,0))
    parser.add_argument('-y','--yvec',help='y vector',nargs='+',type=float,default=(0,1,0))
    parser.add_argument('-z','--zvec',help='z vector',nargs='+',type=float,default=(0,0,1))
    args=parser.parse_args()
    atoms=read(args.filename)

    write_all_octahedra_info(atoms,tuple(args.center), args.vertex, args.distance, output=args.output,axis_type=args.axis_type,x=args.xvec,y=args.yvec,z=args.zvec)



if __name__=='__main__':
    main()
