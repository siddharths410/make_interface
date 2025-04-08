import ase.io
from ase.calculators.emt import EMT

from make_interface import Surfaces, Interfaces

Lmax = 6.0
strain_max = 0.02

bulkAg = ase.io.read("../bulk_structures/Ag.xsf")
surfacesAg = Surfaces(bulkAg, (1, 1, 1), Lmax)  # search for 111 surfaces
print(surfacesAg)

bulkAu = ase.io.read("../bulk_structures/Au.xsf")
surfacesAu = Surfaces(bulkAu, (1, 1, 1), Lmax)  # search for 111 surfaces
print(surfacesAu)

interfaces = Interfaces(surfacesAg, surfacesAu, strain_max)
print(interfaces)

#slab = surfaces.make_slab(0, 10.0, 10.0, EMT())
#ase.io.write("interface111.xsf", slab)
