from netCDF4 import Dataset
from ase import Atoms
from ase.io import read, write
import matplotlib.pyplot as plt


def read_structure_form_MDnc(fname, istep):
    """
    read atomic structure from siesta MD.nc file
    fname: the name of siesta MD.nc file
    istep: the i'th step
    """
    d = Dataset(fname)
    atomic_number = d.variables['iza'][:]
    positions = d.variables['xa'][istep, :, :]
    cell = d.variables['cell'][istep, :, :]
    atoms=Atoms(numbers=atomic_number, positions=positions, cell=cell, pbc=True)
    return atoms

def MDnc_to_file(fname, istep, output):
    """
    read atomic structure as a snapshot of the MD from siesta MD.nc file and write to another file (e.g xyz file)
    fname: the name of siesta MD.nc file
    istep: the i'th step
    output: the name of the file, could be .xyz, .cif, .vasp, etc.
    """
    atoms=read_structure_form_MDnc(fname, istep)
    write(output, atoms)

def plot_temperature(fname):
    """
    fname: file name of siesta MD.nc file
    """
    d = Dataset(fname)
    temperatures = d.variables['temp'][:]
    plt.plot(temperatures)
    plt.xlabel('step')
    plt.ylabel('Temperature (K)')
    plt.show()


def plot_pressure(fname):
    """
    fname: file name of siesta MD.nc file
    """
    d = Dataset(fname)
    p = d.variables['psol'][:]
    plt.plot(p)
    plt.xlabel('step')
    plt.ylabel('Pressure (Kbar)')
    plt.show()


def plot_KS_energy(fname):
    """
    plot Kohn Sham energy as function of MD step
    fname: file name of siesta MD.nc file
    """
    d = Dataset(fname)
    Eks = d.variables['eks'][:]
    plt.plot(Eks)
    plt.xlabel('step')
    plt.ylabel('KS energy (unknow unit)')
    plt.show()

def plot_total_energy(fname):
    """
    plot Kohn Sham energy as function of MD step
    fname: file name of siesta MD.nc file
    """
    d = Dataset(fname)
    Etot = d.variables['etot'][:]
    plt.plot(Etot)
    plt.xlabel('step')
    plt.ylabel('Total energy (unknow unit)')
    plt.show()



def plot_volume(fname):
    """
    plot volume as function of MD step
    fname: file name of siesta MD.nc file
    """
    d = Dataset(fname)
    V = d.variables['volume'][:]
    plt.plot(V)
    plt.xlabel('step')
    plt.ylabel('Volume ($\\AA^3$)')
    plt.show()

def plot_all():
    fname='GeSe.MD.nc'
    plot_temperature(fname)
    plot_pressure(fname)
    plot_volume(fname)
    plot_total_energy(fname)

#plot_all()

MDnc_to_file(fname='GeSe.MD.nc', istep=20, output='GeSe_step20.cif')
