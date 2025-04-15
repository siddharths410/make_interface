import numpy as np
import ase.io
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator

from make_interface import Surfaces, Interfaces

Lmax = 7.0
strain_max = 0.06

calc = MACECalculator(model_paths=["../mace.model"], device="cuda")

print("\n---- Au ----\n")
bulkAu = ase.io.read("../bulk_structures/Au.xsf")
bulkAu.calc = calc
LBFGS(FrechetCellFilter(bulkAu), logfile="-").run(fmax=0.01)
surfacesAu = Surfaces(bulkAu, (0, 0, 0), Lmax)  # all surfaces
print(surfacesAu)

print("\n---- TiO2 ----\n")
bulkTiO2 = ase.io.read("../bulk_structures/TiO2.xsf")
bulkTiO2.calc = calc
LBFGS(FrechetCellFilter(bulkTiO2), logfile="-").run(fmax=0.01)
surfacesTiO2 = Surfaces(bulkTiO2, (0, 0, 0), Lmax)  # all surfaces
print(surfacesTiO2)

print("\n---- Au - TiO2 interface ----\n")
interfaces = Interfaces(surfacesAu, surfacesTiO2, strain_max)
print(interfaces)

i_interface = interfaces.strain_max.argmin()  # pick minimum strain within selections
slab, slabAu, slabTiO2 = interfaces.make_slab(i_interface, 14.0, 10.0, calc, 3)
n_bulk_Au = len(slabAu) / len(bulkAu)
n_bulk_TiO2 = len(slabTiO2) / len(bulkTiO2)
slab.calc = calc
LBFGS(FrechetCellFilter(slab), logfile="-").run(fmax=0.01)
base_area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
interface_energy = (
    slab.get_potential_energy() - (
        bulkAu.get_potential_energy() * n_bulk_Au
        + bulkTiO2.get_potential_energy() * n_bulk_TiO2
    )
) / (2 * base_area)
print(f"{interface_energy = :.4f} eV/A^2")
ase.io.write("interface_with_opt.xsf", slab)
