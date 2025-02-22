import numpy as np
from ase import Atoms


def unit_cells(
    bulk: Atoms,
    surface_index: np.ndarray,
    Lmax: float,
    theta_max=np.radians(135),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find surface unit cells of `bulk` with miller index `surface_index`,
    such that both in-plane lattice vectors are shorter than `Lmax` 
    and have a reasonable angle between 90 degrees and `theta_max`.
    Returns the surface linear combinations (N x 2 x 3), along with the
    corresponding lattice vector lengths (N x 2) and cos(angle)'s (N).
    The two lattice vectors are in ascending length order for each case.
    """
    
    # Get lattice vectors and compute their lengths:
    vectors = lattice_vectors(bulk, surface_index, Lmax)
    RT = bulk.cell[:]
    vectors_cart = vectors @ RT
    vector_lengths = np.linalg.norm(vectors @ RT, axis=1)
    
    # Compute cos(angle) between all pairs of vectors
    cos_theta = (
        (vectors_cart @ vectors_cart.T) / np.outer(vector_lengths, vector_lengths)
    )

    # Select pairs with v1 shorter than v2 and angle in range:
    cos_theta_min = np.cos(theta_max)
    cos_theta_max = 1E-12  # 90 degrees upto round-off threshold
    sel1, sel2 = np.where(
        (vector_lengths[:, None] <= vector_lengths[None, :])  # sort v1 shorter than v2
        & (vectors[:, None, 0] >= 0)  # one of each pair of inversion partners WLOG
        & (cos_theta >= cos_theta_min)
        & (cos_theta <= cos_theta_max)
    )
    if not len(sel1):
        raise KeyError(
            f"No surface unit cells with lattice vector lengths <= {Lmax}"
            "and a reasonable angle"
        )
    
    # Collect selected pairs:
    unit_cells = np.stack((vectors[sel1], vectors[sel2]), axis=1)
    unit_cell_lengths = np.stack((vector_lengths[sel1], vector_lengths[sel2]), axis=1)
    cos_theta = cos_theta[sel1, sel2]

    # Sort by unit cell area and find unique lengths and angles:
    cell_area = np.prod(unit_cell_lengths, axis=-1) * np.sqrt(1 - cos_theta**2)
    
    class SortKey:
        """Indexed sorter of unit cells by area, angle and then individual lengths."""
        def __init__(self, i):
            self.index = i  # index into relevant arrays
            self.keys = np.array(
                [
                    cell_area[i],
                    np.arccos(cos_theta[i]),
                    unit_cell_lengths[i, 0],
                    unit_cell_lengths[i, 1],
                ]
            )

        def compare(self, other):
            TOL = 1e-4
            key_diff = self.keys - other.keys
            for d in key_diff:
                if np.abs(d) > TOL:
                    return int(np.copysign(1, d))
            return 0

        def __lt__(self, other): return self.compare(other) < 0
        def __gt__(self, other): return self.compare(other) > 0
        def __eq__(self, other): return self.compare(other) == 0
        def __le__(self, other): return self.compare(other) <= 0
        def __ge__(self, other): return self.compare(other) >= 0
        def __ne__(self, other): return self.compare(other) != 0

    sorted_keys = sorted([SortKey(i) for i in range(len(sel1))])
    sel = [sorted_keys[0].index]
    for prev_key, key in zip(sorted_keys, sorted_keys[1:]):
        if prev_key != key:
            sel.append(key.index)
    return unit_cells[sel], unit_cell_lengths[sel], cos_theta[sel]


def print_candidates(
    unit_cells: np.ndarray, unit_cell_lengths: np.ndarray, cos_theta: np.ndarray
) -> None:
    """Neatly summarize results of `unit_cells` in tabular form."""
    areas = np.prod(unit_cell_lengths, axis=-1) * np.sqrt(1. - cos_theta**2)
    angles_deg = np.degrees(np.arccos(cos_theta))
    print(
        f"Index | {'Area':>6s} {'a':>6s} {'b':>6s} {'theta':>6s} |  vector1 |  vector2"
    )
    print('-'*6 + '+' + '-'*29 + '+' + '-'*10 + '+' + '-'*10)
    for i, (area, (a, b), theta, ((a0, a1, a2), (b0, b1, b2))) in enumerate(
        zip(areas, unit_cell_lengths, angles_deg, unit_cells)
    ):
        print(
            f"{i:5d} | {area:6.2f} {a:6.3f} {b:6.3f} {theta:6.2f}"
            f" | {a0:2d} {a1:2d} {a2:2d} | {b0:2d} {b1:2d} {b2:2d}"
        )


def lattice_vectors(
    bulk: Atoms, surface_index: np.ndarray, Lmax: float
) -> np.ndarray:
    """
    Find surface lattice vectors for `bulk` with Miller indices `surface_index`.
    Return (N x 3) linear combination coefficients for each lattice vector
    up to maximum length `Lmax` (in Angstroms).
    """
    
    # Find n_max for each dimension to cover all lattice vectors with length upto Lmax
    RT = bulk.cell[:]
    RTinv = np.linalg.inv(RT)
    n_max = np.ceil(Lmax * np.linalg.norm(RTinv, axis=0)).astype(int)
    
    # List all linear combinations with coefficients up to n_max
    grids1d =  [np.arange(-n, n+1) for n in n_max]
    vectors = np.stack(np.meshgrid(*grids1d, indexing="ij"), axis=-1).reshape(-1, 3)
    vector_lengths = np.linalg.norm(vectors @ RT, axis=1)
    
    # Select non-zero vectors hsorter than Lmax and perpendicular to surface_index
    sel = np.where(
        (vector_lengths > 0)
        & (vector_lengths <= Lmax)
        & (vectors @ surface_index == 0)
    )[0]
    if not len(sel):
        raise KeyError(f"No surface vectors with lengths <= {Lmax}")
    return vectors[sel]
