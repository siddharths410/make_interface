import ase.io

from make_interface import surface


bulkAg = ase.io.read("Ag.xsf")
candidates = surface.unit_cells(bulkAg, (1, 1, 1), 5.0)
surface.print_candidates(*candidates)
