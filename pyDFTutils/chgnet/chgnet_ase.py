from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from pymatgen.core import Structure

def get_chgnet_calculator():
    chgnet=CHGNet.load()
    calc=CHGNetCalculator(chgnet)
    return calc

#get_chgnet_calculator()
