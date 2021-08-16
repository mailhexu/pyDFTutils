from scipy.io.netcdf import netcdf_file
from ase.units import Ha, eV
import numpy as np
import re

class GSR(object):
    def __init__(self,fname):
        self._fname=fname
        self._file = netcdf_file(self._fname)
        self._variables=self._file.variables

    def get_efermi(self):
        return float(self._variables['e_fermie'].data)*Ha

    def get_kpoints(self):
        return self._variables['reduced_coordinates_of_kpoints'].data

class outfile(object):
    def __init__(self,fname):
        self._fname=fname
        with open(self._fname) as myfile:
            self.text=myfile.read()
        

    def get_kpoints(self):
        ptr=re.compile(r'kpt\s+([0-9ED\-\+\.\s]+)', re.MULTILINE)
        ktext=ptr.findall(self.text)[0]
        print(ktext)
        ktext=ktext.replace('D', 'E')
        self._kpoints= [float(x) for x in ktext.split()]
        nkpts=len(self._kpoints)//3
        self._kpoints=np.reshape(self._kpoints, (nkpts, 3))
        return self._kpoints


def get_efermi(fname):
    gsr=GSR(fname)
    return gsr.get_efermi()

def get_kpoints(fname):
    txt=outfile(fname)
    return txt.get_kpoints()


#print(get_kpoints('text.txt'))
