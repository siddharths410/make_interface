import ase.io
from ase.calculators.emt import EMT
from mace.calculators import MACECalculator

from make_interface import Surfaces, Interfaces

Lmax = 7.0
strain_max = 0.08

bulkTiO2 = ase.io.read("../bulk_structures/TiO2.xsf")
surfacesTiO2 = Surfaces(bulkTiO2, (1, 1, 0), Lmax)
print(surfacesTiO2)

bulkIrO2 = ase.io.read("../bulk_structures/IrO2.xsf")
surfacesIrO2 = Surfaces(bulkIrO2, (1, 1, 0), Lmax)
print(surfacesIrO2)

interfaces = Interfaces(surfacesTiO2, surfacesIrO2, strain_max)
print(interfaces)

i_interface = 0  # pick smallest area interface
calc = MACECalculator(model_paths=["../mace.model"], device="cuda")
slab = interfaces.make_slab(i_interface, 14.0, 10.0, calc, 3)
ase.io.write("interface110.xsf", slab)
