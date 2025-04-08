import ase.io
from ase.calculators.emt import EMT
from mace.calculators import MACECalculator

from make_interface import Surfaces, Interfaces

Lmax = 7.0
strain_max = 0.06

bulkAu = ase.io.read("../bulk_structures/Au.xsf")
surfacesAu = Surfaces(bulkAu, (0, 0, 0), Lmax)  # all surfaces
print(surfacesAu)

bulkTiO2 = ase.io.read("../bulk_structures/TiO2.xsf")
surfacesTiO2 = Surfaces(bulkTiO2, (0, 0, 0), Lmax)  # all surfaces
print(surfacesTiO2)

interfaces = Interfaces(surfacesAu, surfacesTiO2, strain_max)
print(interfaces)

i_interface = interfaces.strain_max.argmin()  # pick minimum strain within selections
calc = MACECalculator(model_paths=["../mace.model"], device="cuda")
slab = interfaces.make_slab(i_interface, 14.0, 10.0, calc, 3)
ase.io.write("interface_smallest.xsf", slab)
