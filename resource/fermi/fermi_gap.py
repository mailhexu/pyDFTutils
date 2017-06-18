#
# Script developed by H. Lambert and S. Ponce [2016]
#
import sys
import re
import numpy as np

def parse_args(args):
  extra = []
  vars  = []
  current_var = None
  for arg in args:
    if arg.startswith('--'):
      current_var = arg[2:]
    else:
      if current_var is not None:
        vars.append((current_var, arg))
        current_var = None
      else:
        extra.append(arg)
  return (extra, vars)

def split_vars(vars):
  vars_values = []
  for var_name, values in vars:
    values = values.split(",")
    try:
      if any(['.' in value for value in values]):
        values = map(float, values)
      else:
        values = map(int, values)
    except ValueError:
      pass
    vars_values.append((var_name, values))
  vars_dict = dict(vars_values)
  return vars_dict

#f is a string with the contents of the file
class FermiSurface(object):
  def __init__(self):
    self.nx = 60
    self.ny = 60
    self.nz = 60
    self.dimvec = np.array([float(self.nx), float(self.ny), float(self.nz)])
    self.fermixyz = {}
    self.gap = {}
    self.prefix = 'MgB2'
    self.nbndmin = 2
    self.nbndmax = 4

  def __repr__(self):
    return 'Fermi Surface/Mu Tensor Object'

  def cryst_to_cart(self, kvec):
#MgB2 crystal axes
    at1 = np.array([ 1.000000,   0.000000,   0.000000])
    at2 = np.array([-0.500000,   0.866025,   0.000000])
    at3 = np.array([ 0.000000,   0.000000,   1.142069])
    at  = np.array([at1, at2, at3])
    outvec = np.dot( kvec,at)
    return outvec

  def cart_to_cryst(self, kvec):
#crystal to cart BG MgB2
    at1 = np.array([ 1.000000,  -0.500000,   0.000000])
    at2 = np.array([ 0.000000,   0.866025,   0.000000])
    at3 = np.array([ 0.000000,   0.000000,   1.142069])
    at  = np.array([at1, at2, at3])
    outvec = np.dot(kvec,at)
    return outvec

  def pull_fermi(self,f):
    fermi_regex  = re.compile(r'k\s=\s?(\-?[0-9\.]+)\s?(\-?[0-9\.]+)\s?(\-?[0-9\.]+).*?:\n\n\s+([0-9\.\-\s]+)')
    print len(fermi_regex.findall(f))
    for a, b, c, d in fermi_regex.findall(f):
      a = float(a)
      b = float(b)
      c = float(c)
      kvec = np.array([a,b,c])
      d = map(float, d.split())
      kvec = self.cryst_to_cart(kvec)
  
      #Fold into first brillouin zone:
      for i, a in enumerate(kvec):
        if (a< -0.001): kvec[i] = kvec[i] + 1.0
      index = [round(a) for a in np.multiply(kvec, self.dimvec)]

      # Rescale the energy so that the Fermi level = 0
      d[:] = [x - 7.4272 for x in d]

      for i, a in enumerate(index):
        if index[i] == 61.0: 
          index[i] = 0.0
      print index
      self.fermixyz[tuple(index)] = d

  def pull_fermi_xy(self,f):
    fermi_regex  = re.compile(r'k\s=\s?(\-?[0-9\.]+)\s?(\-?[0-9\.]+)\s?(\-?[0-9\.]+).*?:\n\n\s+([0-9\.\-\s]+)')
    print len(fermi_regex.findall(f))
    for a, b, c, d in fermi_regex.findall(f):
      a    = float(a)
      b    = float(b)
      c    = float(c)
      kvec = np.array([a,b,c])
      d    = map(float, d.split())
      # Turn kpoint coordinates into integer indices
      # and fold back into the first Brillouin zone.
      kvec = self.cryst_to_cart(kvec)
      # Fold into first Brillouin zone:
      for i, a in enumerate(kvec):
        if (a<0.0): kvec[i] = kvec[i] + 1.0
      index = [round(a) for a in np.multiply(kvec, self.dimvec)]
      for i, a in enumerate(index):
        if index[i] == 61.0:
          index[i] = 0.0     

#returns dictionary keys:xyz coordinates, values:eigenvalues.
  def print_xsf(self, surf, title='band', band1=1):
    for ibnd in range(band1):
      f1 = open('{0}.col.band{1}.xsf'.format(self.prefix, ibnd), 'w')
      print >>f1, "BEGIN_BLOCK_DATAGRID_3D" 
      print >>f1, "{0}_band_{1}".format(self.prefix, ibnd)     
      print >>f1, " BEGIN_DATAGRID_3D_{0}".format(self.prefix) 
      print >>f1, " {0}  {1}  {2} ".format(self.nx, self.ny, self.nz)
      print >>f1, "0.000000  0.000000  0.000000"   
    #MgB2:
      print >>f1, "1.000000  0.577350  0.000000"
      print >>f1, "0.000000  1.154701  0.000000"
      print >>f1, "0.000000  0.000000  0.875604"
      print >>f1, ""
      
      total = 0
      for z in range(self.nz):
        for y in range(self.ny):
          for x in range(self.nx):
            try:
              print>>f1, surf[x,y,z], " ",
              #print>>f1, "0.05", " ",
              total = total+ 1 
            except TypeError:
              print>>f1, surf[x,y,z], " ",
              print 'Missing key'
              print>>f1, "0.0", " ",
            except KeyError:
              print 'Missing key' 
              print>>f1, "0.0", " ",
          print >> f1, ""
        print >> f1, ""
      print >>f1, "END_DATAGRID_3D"  
      print >>f1, "END_BLOCK_DATAGRID_3D"  
      f1.close()
      print 'Total number of data ',total

  def pull_muk(self, f):
    total = 0
    for line in f.split('\n'):
      try:
        a,b,c,d,e,f = map(float, line.split())
        # Do one band d manually
        if d == 2:
          kvec = np.array([a,b,c])
          kvec = self.cart_to_cryst(kvec)

          index = [round(ii) for ii in np.multiply(kvec, self.dimvec)]
          print index
          self.gap[tuple(index)] = f
          total += 1
      except:
        print "Couldn't read the following line:"
        print line
    print 'Total number of lines extracted ',total  

if __name__=="__main__":
# run as: 
# python --fs y ./nscf.out to parse band file and make fermiplot
# else run as:
# python --gap y name to parse gap plot.
  extra, vars = parse_args(sys.argv[1:])
  vars_values = []

  vars = split_vars(vars)
  print vars, extra

  f = open(extra[0]).read()
  fs = FermiSurface()

  if 'fs' in vars.keys():
    fs.pull_fermi(f)
    fs.print_xsf(fs.fermixyz, band1=2, band2=3, band3=4)

  if 'gap' in vars.keys():
    fs.pull_muk(f)
    fs.print_xsf(fs.gap, 'gap')
