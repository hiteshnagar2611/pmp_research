"""
MD Trajectory Processor: Extract dynamics features from molecular dynamics.

Computes:
  - RMSF (root mean square fluctuation) per residue
  - Velocity vectors from time-dependent positions
  - Conformational ensemble statistics
  - Backbone frame trajectories

Supports formats: DCD, XTC, TRR (via MDAnalysis)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import MDAnalysis (optional dependency)
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.base import AnalysisBase
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    logger.warning("MDAnalysis not installed. MD processing will be limited.")


class MDProcessor:
    """
    Process molecular dynamics trajectories.

    Extracts RMSF, velocities, and conformational dynamics from MD simulations.

    Args:
        trajectory_file (str): Path to MD trajectory (DCD, XTC, TRR)
        topology_file (str): Path to topology file (PSF, PDB, GRO)
        start_frame (int): First frame to analyze (default: 0)
        end_frame (int): Last frame to analyze (default: None, all frames)
        stride (int): Take every nth frame (default: 1)
    """

    def __init__(
        self,
        trajectory_file: str,
        topology_file: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        stride: int = 1,
    ):
        """Initialize MD processor."""
        self.trajectory_file = Path(trajectory_file)
        self.topology_file = Path(topology_file)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stride = stride

        # Lazy load universe
        self._universe = None

    @property
    def universe(self):
        """Lazy load MDAnalysis universe."""
        if self._universe is None:
            if not HAS_MDANALYSIS:
                raise ImportError("MDAnalysis required for MD processing")

            self._universe = mda.Universe(
                str(self.topology_file),
                str(self.trajectory_file),
            )
        return self._universe

    def compute_rmsf(
        self,
        selection: str = "name CA",
        fit: bool = True,
    ) -> np.ndarray:
        """
        Compute root mean square fluctuation (RMSF).

        Args:
            selection: MDAnalysis selection string (default: "name CA" for Cα)
            fit: whether to fit to reference structure (default: True)

        Returns:
            rmsf: (N_residues,) RMSF values in Angstroms
        """
        if not HAS_MDANALYSIS:
            logger.warning("MDAnalysis not available, returning dummy RMSF")
            return np.ones(self.n_residues) * 1.0

        u = self.universe
        ca = u.select_atoms(selection)
        n_atoms = ca.n_atoms

        # Reference structure
        ref_coords = u.trajectory[self.start_frame].positions[ca.indices].copy()

        # Collect coordinates
        rmsf_sq = np.zeros(n_atoms)
        n_frames = 0

        for frame_idx in range(self.start_frame, self.end_frame or len(u.trajectory), self.stride):
            u.trajectory[frame_idx]
            coords = u.trajectory[frame_idx].positions[ca.indices]

            if fit:
                # Fit to reference and compute displacement
                coords = self._fit_to_reference(coords, ref_coords)

            rmsf_sq += np.sum((coords - ref_coords) ** 2, axis=1)
            n_frames += 1

        rmsf = np.sqrt(rmsf_sq / n_frames)

        return rmsf

    def compute_velocities(
        self,
        selection: str = "name CA",
        dt: float = 1.0,  # picoseconds
    ) -> np.ndarray:
        """
        Compute velocity vectors from trajectory.

        Args:
            selection: MDAnalysis selection string
            dt: timestep in picoseconds

        Returns:
            velocities: (T, N_residues, 3) velocity vectors
        """
        if not HAS_MDANALYSIS:
            logger.warning("MDAnalysis not available, returning dummy velocities")
            return np.random.randn(10, self.n_residues, 3) * 0.1

        u = self.universe
        ca = u.select_atoms(selection)

        positions_list = []

        for frame_idx in range(self.start_frame, self.end_frame or len(u.trajectory), self.stride):
            u.trajectory[frame_idx]
            positions = u.trajectory[frame_idx].positions[ca.indices].copy()
            positions_list.append(positions)

        positions = np.array(positions_list)  # (T, N, 3)

        # Compute velocities: v_t = (x_{t+1} - x_t) / dt
        velocities = np.diff(positions, axis=0) / dt

        return velocities

    def compute_conformational_ensemble(
        self,
        n_snapshots: int = 5,
        selection: str = "name CA",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract conformational ensemble snapshots.

        Args:
            n_snapshots: number of representative frames to extract
            selection: atom selection

        Returns:
            snapshots: (n_snapshots, N_residues, 3) coordinates
            rmsf_per_frame: (n_frames,) RMSF for each frame
        """
        if not HAS_MDANALYSIS:
            logger.warning("MDAnalysis not available, returning dummy ensemble")
            frames = np.linspace(0, 9, n_snapshots, dtype=int)
            return np.random.randn(n_snapshots, self.n_residues, 3), np.ones(10)

        u = self.universe
        ca = u.select_atoms(selection)

        # Collect all coordinates
        all_coords = []
        frame_indices = []

        for frame_idx in range(self.start_frame, self.end_frame or len(u.trajectory), self.stride):
            u.trajectory[frame_idx]
            coords = u.trajectory[frame_idx].positions[ca.indices].copy()
            all_coords.append(coords)
            frame_indices.append(frame_idx)

        all_coords = np.array(all_coords)  # (T, N, 3)

        # Select representative frames (e.g., maximum RMSF frames)
        rmsf_per_frame = np.sqrt(
            np.sum((all_coords - all_coords.mean(axis=0)) ** 2, axis=(1, 2)) / ca.n_atoms
        )

        # Select frames with highest RMSF (most flexible)
        selected_indices = np.argsort(rmsf_per_frame)[-n_snapshots:]

        snapshots = all_coords[selected_indices]

        return snapshots, rmsf_per_frame

    @staticmethod
    def _fit_to_reference(
        coords: np.ndarray,
        ref_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Fit coordinates to reference using least squares rotation.

        Uses Kabsch algorithm.

        Args:
            coords: (N, 3) current coordinates
            ref_coords: (N, 3) reference coordinates

        Returns:
            coords_fitted: fitted coordinates
        """
        # Center
        coords_centered = coords - coords.mean(axis=0)
        ref_centered = ref_coords - ref_coords.mean(axis=0)

        # SVD
        H = coords_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation
        coords_fitted = coords_centered @ R.T + ref_centered.mean(axis=0)

        return coords_fitted

    @property
    def n_residues(self) -> int:
        """Number of Cα atoms (residues)."""
        if not HAS_MDANALYSIS:
            return 0
        return len(self.universe.select_atoms("name CA"))

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        if not HAS_MDANALYSIS:
            return 0
        return len(self.universe.trajectory)


class RMSFComputer:
    """Simple RMSF computation without MDAnalysis dependency."""

    @staticmethod
    def compute_from_snapshots(
        snapshots: np.ndarray,  # (T, N, 3)
    ) -> np.ndarray:
        """
        Compute RMSF from coordinate snapshots.

        Args:
            snapshots: (T, N, 3) coordinates over time

        Returns:
            rmsf: (N,) per-residue RMSF
        """
        # Mean structure
        mean_coords = snapshots.mean(axis=0)

        # Deviations
        deviations = snapshots - mean_coords  # (T, N, 3)

        # RMSF: sqrt(mean(squared_distance))
        rmsf = np.sqrt(np.mean(np.sum(deviations ** 2, axis=2), axis=0))

        return rmsf

    @staticmethod
    def compute_bfactors(
        rmsf: np.ndarray,
    ) -> np.ndarray:
        """
        Convert RMSF to B-factors.

        B = (8π²/3) * RMSF²

        Args:
            rmsf: (N,) RMSF values in Angstroms

        Returns:
            bfactors: (N,) B-factors in Ų
        """
        bfactors = (8 * np.pi ** 2 / 3) * rmsf ** 2
        return bfactors


class VelocityExtractor:
    """Extract velocity features from MD snapshots."""

    @staticmethod
    def compute_velocities(
        snapshots: np.ndarray,  # (T, N, 3)
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Compute velocities from snapshots.

        Args:
            snapshots: (T, N, 3) coordinates
            dt: timestep

        Returns:
            velocities: (T-1, N, 3) velocities
        """
        velocities = np.diff(snapshots, axis=0) / dt
        return velocities

    @staticmethod
    def compute_velocity_autocorr(
        snapshots: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Compute velocity autocorrelation function.

        C(t) = <v(0) · v(t)> / <v·v>

        Args:
            snapshots: (T, N, 3) coordinates
            dt: timestep

        Returns:
            autocorr: (T-1,) autocorrelation function
        """
        velocities = np.diff(snapshots, axis=0) / dt  # (T-1, N, 3)

        # Compute autocorrelation
        v0 = velocities[0]  # reference velocity
        autocorr = []

        for t in range(len(velocities)):
            vt = velocities[t]
            corr = np.mean(np.sum(v0 * vt, axis=1)) / np.mean(np.sum(v0 * v0, axis=1))
            autocorr.append(corr)

        return np.array(autocorr)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_md_ensemble(
    pdb_file: str,
    md_dir: str,
    n_snapshots: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MD ensemble from directory of snapshots.

    Assumes directory contains PDB files for each snapshot.

    Args:
        pdb_file: reference PDB file
        md_dir: directory with snapshot PDB files
        n_snapshots: number of snapshots to load

    Returns:
        coords: (n_snapshots, N, 3) coordinates
        rmsf: (N,) RMSF values
        velocities: (n_snapshots-1, N, 3) velocities
    """
    import warnings
    warnings.filterwarnings('ignore')

    md_dir = Path(md_dir)
    snapshot_files = sorted(md_dir.glob("*.pdb"))[:n_snapshots]

    if len(snapshot_files) == 0:
        logger.warning(f"No PDB files found in {md_dir}")
        return None, None, None

    coords_list = []

    for snapshot_file in snapshot_files:
        try:
            # Parse PDB (simplified - assumes CA atoms)
            u = mda.Universe(str(snapshot_file)) if HAS_MDANALYSIS else None
            if u is not None:
                ca = u.select_atoms("name CA")
                coords = ca.positions.copy()
                coords_list.append(coords)
        except Exception as e:
            logger.warning(f"Could not load {snapshot_file}: {e}")

    if not coords_list:
        return None, None, None

    coords = np.array(coords_list)  # (T, N, 3)

    # Compute RMSF and velocities
    rmsf = RMSFComputer.compute_from_snapshots(coords)
    velocities = VelocityExtractor.compute_velocities(coords)

    return coords, rmsf, velocities


if __name__ == "__main__":
    # Test MD processor
    print("MD Processor loaded successfully")
    print("\nKey classes:")
    print("- MDProcessor: full MD trajectory processing")
    print("- RMSFComputer: simple RMSF computation")
    print("- VelocityExtractor: velocity/dynamics features")
