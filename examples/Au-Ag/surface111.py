import ase.io

from make_interface.surface import Surfaces


bulkAg = ase.io.read("Ag.xsf")
surfaces = Surfaces(bulkAg, (1, 1, 1), 6.0)
print(surfaces)

slab = surfaces.make_slab(0, 10.0, 10.0, None)
ase.io.write("test.xsf", slab)

#surfaces = Surfaces(bulkAg, (0, 0, 0), 5.0)
#print(surfaces)


