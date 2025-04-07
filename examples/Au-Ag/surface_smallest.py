import ase.io
from ase.calculators.emt import EMT

from make_interface.surface import Surfaces


bulkAg = ase.io.read("Ag.xsf")
surfaces = Surfaces(bulkAg, (0, 0, 0), 5.0)  # search for all surface directions
print(surfaces)

slab = surfaces.make_slab(0, 10.0, 10.0, EMT())
ase.io.write("out-Ag-smallest.xsf", slab)
