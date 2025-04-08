import ase.io
from ase.calculators.emt import EMT

from make_interface import Surfaces


bulkAg = ase.io.read("../bulk_structures/Ag.xsf")
surfaces = Surfaces(bulkAg, (0, 0, 0), 5.0)  # search for all surface directions
print(surfaces)

slab = surfaces.make_slab(0, 10.0, 10.0, EMT())
ase.io.write("surface_smallest.xsf", slab)
