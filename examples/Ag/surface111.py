import ase.io
from ase.calculators.emt import EMT

from make_interface import Surfaces


bulkAg = ase.io.read("../bulk_structures/Ag.xsf")
surfaces = Surfaces(bulkAg, (1, 1, 1), 6.0)  # search for 111 surfaces
print(surfaces)

slab = surfaces.make_slab(0, 10.0, 10.0, EMT())
ase.io.write("surface111.xsf", slab)
