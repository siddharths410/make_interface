import ase.io

from make_interface.surface import Surfaces


bulkAg = ase.io.read("Ag.xsf")
surfaces = Surfaces(bulkAg, (1, 1, 1), 6.0)
print(surfaces)

#unit_cells, unit_cell_lengths, cos_theta = candidates
