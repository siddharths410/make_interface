import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.calculators.calculator import Calculator


class Surfaces:
    """List of possible surface supercells, constrained by surface size."""
    bulk: Atoms  #: Reference bulk structure
    sup: np.ndarray  #: N x 3 x 3 linear combinations defining each supercell
    a: np.ndarray  #: N lengths (Angstroms) of shorter in-plane lattice vector
    b: np.ndarray  #: N lengths (Angstroms) of longer in-plane lattice vector
    theta: np.ndarray  #: N angles (radians) between in-plane lattice vectors
    area: np.ndarray  #: N areas of the surface supercells

    def __init__(
        self,
        bulk: Atoms,
        surface_index: np.ndarray,
        Lmax: float,
        theta_max=np.radians(135),
    ) -> None:
        """
        Find surface unit cells of `bulk` with miller index `surface_index`,
        such that both in-plane lattice vectors are shorter than `Lmax` 
        and have a reasonable angle between 90 degrees and `theta_max`.
        
        If `surface_index` is (0, 0, 0), this results in all surface supercells
        with in-plane lengths less than `Lmax`, regardless of surface direction.
        """ 

        # Get lattice vectors and compute their lengths:
        vectors = lattice_vectors(bulk, surface_index, Lmax)
        RT = bulk.cell[:]
        vectors_cart = vectors @ RT
        vector_lengths = np.linalg.norm(vectors_cart, axis=1)
        
        # Compute angle between all pairs of vectors:
        theta = np.arccos(
            (vectors_cart @ vectors_cart.T) / np.outer(vector_lengths, vector_lengths)
        )
        
        # Select pairs with v1 shorter than v2 and angle in range:
        TOL = 1e-6
        theta_min = 0.5 * np.pi - TOL  # 90 degrees (with round-off margin)
        sel1, sel2 = np.where(
            (vector_lengths[:, None] <= vector_lengths[None, :] + TOL)  # |v1| <= |v2|
            & (theta >= theta_min)
            & (theta <= theta_max)
        )
        if not len(sel1):
            raise KeyError(
                f"No surface unit cells with lattice vector lengths <= {Lmax}"
                "and a reasonable angle"
            )
        
        # Compute and reduce normals:
        normals = np.cross(vectors[sel1], vectors[sel2])
        normals_gcd = np.gcd(normals[..., 0], np.gcd(normals[..., 1], normals[..., 2]))
        normals //= normals_gcd[..., None]
        
        # Collect and sort selected pairs:
        sup = np.stack((vectors[sel1], vectors[sel2], normals), axis=2)
        a = vector_lengths[sel1]
        b = vector_lengths[sel2]
        theta = theta[sel1, sel2]
        area = a * b * np.sin(theta)

        # Determine keys for sorting output surfaces (most to least important):
        sort_properties = [area, theta, a, b]
        sort_properties.append(np.abs(normals).sum(axis=-1))  # simplest of equiv. specs
        if np.linalg.norm(surface_index):
            sort_properties.append(-normals @ surface_index)  # fix sign when possible
        sort_properties.extend(-normals.T)  # sort descending by normal components
        sort_properties.extend(-vectors[sel1].T)  # sort descending by v1 components
        sort_properties.extend(-vectors[sel2].T)  # sort descending by v2 components
        sort_properties = np.stack(sort_properties, axis=1)

        class SortKey:
            """Indexed sorter of unit cells by area, theta, a, b."""
            def __init__(self, i):
                self.index = i  # index into relevant arrays

            def compare(self, other) -> int:
                key_diff = sort_properties[self.index] - sort_properties[other.index]
                for d in key_diff:
                    if np.abs(d) > TOL:
                        return int(np.copysign(1, d))
                return 0
            
            def equivalent(self, other) -> bool:
                """Only check area for equivalence."""
                return np.abs(area[self.index] - area[other.index]) <= TOL

            def __lt__(self, other): return self.compare(other) < 0
            def __gt__(self, other): return self.compare(other) > 0
            def __eq__(self, other): return self.compare(other) == 0
            def __le__(self, other): return self.compare(other) <= 0
            def __ge__(self, other): return self.compare(other) >= 0
            def __ne__(self, other): return self.compare(other) != 0

        sorted_keys = sorted([SortKey(i) for i in range(len(sel1))])
        sel = [sorted_keys[0].index]
        for prev_key, key in zip(sorted_keys, sorted_keys[1:]):
            if not prev_key.equivalent(key):  # pick lowest-angle of equivalent cells
                sel.append(key.index)
        
        # Set properties:
        self.bulk = bulk
        self.sup = sup[sel]
        self.a = a[sel]
        self.b = b[sel]
        self.theta = theta[sel]
        self.area = area[sel]

    def __str__(self) -> str:
        """Neatly summarize in tabular form."""
        lines = [
            f"Index | {'Area':>6s} {'a':>6s} {'b':>6s} {'theta':>6s}"
            " | surface1 | surface2 |  normal",
            '-'*6 + '+' + '-'*29 + '+' + '-'*10 + '+' + '-'*10 + '+' + '-'*10
        ]
        theta_deg = np.degrees(self.theta)
        for i, (area, a, b, theta, (a0, b0, n0, a1, b1, n1, a2, b2, n2)) in enumerate(
            zip(self.area, self.a, self.b, theta_deg, self.sup.reshape(-1, 9))
        ):
            lines.append(
                f"{i:5d} | {area:6.2f} {a:6.3f} {b:6.3f} {theta:6.2f}"
                f" | {a0:2d} {a1:2d} {a2:2d} | {b0:2d} {b1:2d} {b2:2d}"
                f" | {n0:2d} {n1:2d} {n2:2d}"
            )
        return "\n".join(lines)

    def make_slab(
        self,
        i_surface: int,
        minimum_thickness: float,
        vacuum_spacing: float,
        calculator: Calculator,
    ) -> Atoms:
        """Make slab corresponding to index `i_surface` from search results,
        with specified `minimum_thickness` and `vacuum_spacing` in Angstroms,
        using the specified `calculator` to find the lowest-energy cut."""
        # Construct smallest supercell units needed to reach thickness
        sup_unit = self.sup[i_surface]
        RTsup_unit = sup_unit.T @ self.bulk.cell[:]
        volume_unit = abs(np.linalg.det(RTsup_unit))
        base_area = np.linalg.norm(np.cross(RTsup_unit[0], RTsup_unit[1]))
        thickness_unit = volume_unit / base_area
        n_units = int(np.ceil(minimum_thickness / thickness_unit))
        sup = sup_unit @ np.diag([1, 1, n_units])
        supercell = make_supercell(self.bulk, sup.T, wrap=True, order="atom-major")
        return supercell


def lattice_vectors(bulk: Atoms, surface_index: np.ndarray, Lmax: float) -> np.ndarray:
    """
    Find surface lattice vectors for `bulk` with Miller indices `surface_index`.
    Return (N x 3) linear combination coefficients for each lattice vector
    up to maximum length `Lmax` (in Angstroms).
    
    If `surface_index` is (0, 0, 0), return all lattice vectors with length <= `Lmax`.
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
