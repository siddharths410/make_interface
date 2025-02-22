import numpy as np
from ase import Atoms


def surface_lattice_vectors(
    atoms: Atoms, surface_index: np.ndarray, Lmax: float
) -> np.ndarray:
    """Get lattice vectors for surface with Miller indices `surface_index`
    of bulk structure `atoms`. Return (N x 3) linear combination coefficients
    for each lattice vector up to maximum length Lmax (in Angstroms)."""
    
    # Find n_max for each dimension to cover all lattice vectors with length upto Lmax
    RT = atoms.cell[:]
    RTinv = np.linalg.inv(RT)
    n_max = np.ceil(Lmax * np.linalg.norm(RTinv, axis=0)).astype(int)
    
    # List all linear combinations with coefficients up to n_max
    grids1d =  [ np.arange(-n, n+1) for n in n_max ]
    vectors = np.stack(*grids1d, axis=-1).reshape(-1, 3)
    vector_lengths = np.linalg.norm(vectors @ RT, axis=1)
    
    # Select non-zero vectors hsorter than Lmax and perpendicular to surface_index
    sel = np.where(
        (vector_lengths > 0)
        & (vector_lengths <= Lmax)
        & (vectors @ surface_index == 0)
    )[0]
    return vectors[sel]
