from ase.calculators.siesta import *
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.siesta.parameters import format_fdf



class MySiesta(Siesta):
    def set_mixer(self,
                  method='pulay',
                  weight=0.05,
                  history=10,
                  restart=25,
                  restart_save=4,
                  linear_after=0,
                  linear_after_weight=0.1):
        pass

    def add_Hubbard_U(self, specy, n=3, l=2, U=0, J=0, rc=0.0, Fermi_cut=0.0):
        if not 'Udict' in self.__dict__:
            self.Udict=dict()
        self.Udict[specy]={'n':n, 'l':l, 'U':U, 'J':J, 'rc':rc,'Fermi_cut':Fermi_cut  }

    def set_Hubbard_U(self, Udict):
        """
        Udict: {'Fe': {'n':n, 'l':l, 'U':U, 'J', J, 'rc':rc, 'Fermi_cut':Fermi_cut }}
        """
        self.Udict=Udict



    def write_Hubbard_block(self, f):
        text="%block LDAU.Proj\n" 
        for key, val in self.Udict.items():
            text += '  %s %s \n'%(key, 1)
            if val['n'] is None:
                text += '  n=%s %s\n'%(n, l)
            else:
                text += '%s\n'%(l)
            text += '  %s  %s\n'%(val['U'], val['J'])
            text += '  %s  %s\n'%(val['rc'], val['Fermi_cut'])
        text="%endblock LDAU.Proj\n" 
        f.write(text)


    def relax(
        self,
        atoms,
        TypeOfRun='cg',
        VariableCell=True,
        ConstantVolume=False,
        RelaxCellOnly=False,
        MaxForceTol=0.04,
        MaxStressTol=1,
        NumCGSteps=40,
    ):
        pbc=atoms.get_pbc()
        initial_magnetic_moments=atoms.get_initial_magnetic_moments()
        self.set_fdf_arguments({
            'MD.TypeOfRun': TypeOfRun,
            'MD.VariableCell': VariableCell,
            'MD.ConstantVolume': ConstantVolume,
            'MD.RelaxCellOnly': RelaxCellOnly,
            'MD.MaxForceTol': "%s eV/Ang"%MaxForceTol,
            'MD.MaxStressTol': "%s GPa"%MaxStressTol,
            'MD.NumCGSteps': NumCGSteps,
        })
        self.calculate(atoms)
        self.read(self.label+'.XV')
        self.atoms.set_pbc(pbc)
        self.atoms.set_initial_magnetic_moments(initial_magnetic_moments)
        atoms=self.atoms
        self.set_fdf_arguments({
            'MD.NumCGSteps': 0,
        })
        return self.atoms

