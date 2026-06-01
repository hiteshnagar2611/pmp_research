"""
Feature extraction for protein residues.

Extracts:
  - Scalar node features: backbone dihedrals, physicochemical, exposure, PMP-specific
  - Vector node features: backbone frame, Cβ direction, membrane normal, MD displacement
  - Edge features (delegated to graph_builder)

Usage:
    extractor = FeatureExtractor(opm_normal, rmsf)
    s_node, V_node = extractor.extract(chain, pdb_structure)
"""

from __future__ import annotations
import numpy as np
from Bio.PDB import Chain, Structure, DSSP
import freesasa


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

KYTE_DOOLITTLE = {
    'ALA':  1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS':  2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE':  4.5,
    'LEU':  3.8, 'LYS': -3.9, 'MET':  1.9, 'PHE':  2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL':  4.2,
}

AA_MAX_SASA = {
    'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 187.0, 'CYS': 148.0,
    'GLN': 214.0, 'GLU': 214.0, 'GLY': 97.0, 'HIS': 216.0, 'ILE': 195.0,
    'LEU': 191.0, 'LYS': 230.0, 'MET': 203.0, 'PHE': 240.0, 'PRO': 145.0,
    'SER': 143.0, 'THR': 163.0, 'TRP': 281.0, 'TYR': 263.0, 'VAL': 165.0,
}

pKa_TABLE = {
    'ASP': 3.9, 'GLU': 4.2, 'HIS': 6.0, 'LYS': 10.5, 'ARG': 12.5,
    'TYR': 10.1, 'CYS': 9.2, 'N': 9.6, 'C': 3.6,
}


class FeatureExtractor:
    """Extract scalar and vector features per residue."""

    def __init__(
        self,
        opm_normal: np.ndarray = None,    # (3,) membrane normal
        rmsf: np.ndarray = None,          # (N,) RMSF from MD
        compute_sasa: bool = True,
    ):
        self.opm_normal = opm_normal or np.array([0.0, 0.0, 1.0])
        self.rmsf = rmsf
        self.compute_sasa = compute_sasa

    def extract(
        self,
        chain: Chain,
        structure: Structure = None,
        md_displacement: np.ndarray = None,  # (N, 3) mean MD displacement
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a protein chain.

        Returns:
            s_node (N, 19): scalar features
            V_node (N, 6, 3): vector features
        """
        residues = list(chain.get_residues())
        N = len(residues)

        # Extract coordinates
        CA_coords, N_coords, C_coords = self._get_backbone_coords(residues)

        # ── Scalar features ───────────────────────────────────────────────────

        # 1. Backbone dihedrals: φ, ψ, ω (sin+cos) — 6 dims
        backbone_feats = self._compute_dihedrals(residues)  # (N, 6)

        # 2. Physicochemical: KD, charge, HBD, HBA — 4 dims
        physico_feats = self._compute_physicochemical(residues)  # (N, 4)

        # 3. Solvent exposure: SASA, rel_SASA, HSE_up, HSE_down, depth — 5 dims
        exposure_feats = self._compute_exposure(
            chain, structure, CA_coords, residues
        )  # (N, 5)

        # 4. PMP-specific: mem_depth, mem_sasa, amph_score, RMSF — 4 dims
        pmp_feats = self._compute_pmp_features(
            residues, CA_coords, exposure_feats
        )  # (N, 4)

        s_node = np.concatenate([
            backbone_feats,   # 6
            physico_feats,    # 4
            exposure_feats,   # 5
            pmp_feats,        # 4
        ], axis=-1)  # (N, 19)

        # ── Vector features ───────────────────────────────────────────────────

        # 1. Backbone frame: u1, u2, u3 — 3 vectors
        frame = self._build_backbone_frame(N_coords, CA_coords, C_coords)  # (N, 3, 3)

        # 2. Virtual Cβ direction — 1 vector
        cb_dir = self._compute_virtual_cbeta(N_coords, CA_coords, C_coords)  # (N, 3)

        # 3. Membrane normal — 1 vector (same for all)
        n_hat = np.tile(self.opm_normal, (N, 1))  # (N, 3)

        # 4. MD displacement mean direction — 1 vector
        if md_displacement is not None:
            md_dir = md_displacement / (np.linalg.norm(md_displacement, axis=-1, keepdims=True) + 1e-8)
        else:
            md_dir = np.zeros((N, 3))

        V_node = np.stack([
            frame[:, 0, :],   # u1 (N→Cα direction)
            frame[:, 1, :],   # u2 (Cα→C direction)
            frame[:, 2, :],   # u3 (frame normal)
            cb_dir,           # Cα→Cβ direction
            n_hat,            # membrane normal
            md_dir,           # MD displacement direction
        ], axis=1)  # (N, 6, 3)

        return s_node.astype(np.float32), V_node.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────

    def _get_backbone_coords(self, residues) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract N, Cα, C coordinates."""
        N_pos, CA_pos, C_pos = [], [], []
        for res in residues:
            if 'N' in res and 'CA' in res and 'C' in res:
                N_pos.append(res['N'].get_vector().get_array())
                CA_pos.append(res['CA'].get_vector().get_array())
                C_pos.append(res['C'].get_vector().get_array())
        return (
            np.array(CA_pos, dtype=np.float32),
            np.array(N_pos, dtype=np.float32),
            np.array(C_pos, dtype=np.float32),
        )

    def _compute_dihedrals(self, residues) -> np.ndarray:
        """Compute φ, ψ, ω angles (sin+cos encoding)."""
        N = len(residues)
        dihedral_feats = np.zeros((N, 6), dtype=np.float32)

        for i, res in enumerate(residues):
            try:
                # φ = (C_i-1, N_i, Cα_i, C_i)
                if i > 0 and 'C' in residues[i-1]:
                    p0 = residues[i-1]['C'].get_vector().get_array()
                    p1 = res['N'].get_vector().get_array()
                    p2 = res['CA'].get_vector().get_array()
                    p3 = res['C'].get_vector().get_array()
                    phi = self._dihedral(p0, p1, p2, p3)
                    dihedral_feats[i, 0] = np.sin(phi)
                    dihedral_feats[i, 1] = np.cos(phi)

                # ψ = (N_i, Cα_i, C_i, N_i+1)
                if i < N - 1 and 'N' in residues[i+1]:
                    p0 = res['N'].get_vector().get_array()
                    p1 = res['CA'].get_vector().get_array()
                    p2 = res['C'].get_vector().get_array()
                    p3 = residues[i+1]['N'].get_vector().get_array()
                    psi = self._dihedral(p0, p1, p2, p3)
                    dihedral_feats[i, 2] = np.sin(psi)
                    dihedral_feats[i, 3] = np.cos(psi)

                # ω = (Cα_i, C_i, N_i+1, Cα_i+1) — usually ~180° (planar peptide bond)
                if i < N - 1 and 'CA' in residues[i+1]:
                    p0 = res['CA'].get_vector().get_array()
                    p1 = res['C'].get_vector().get_array()
                    p2 = residues[i+1]['N'].get_vector().get_array()
                    p3 = residues[i+1]['CA'].get_vector().get_array()
                    omega = self._dihedral(p0, p1, p2, p3)
                    dihedral_feats[i, 4] = np.sin(omega)
                    dihedral_feats[i, 5] = np.cos(omega)
            except Exception:
                pass  # missing atoms, leave zeros

        return dihedral_feats

    def _dihedral(self, p0, p1, p2, p3) -> float:
        """Compute dihedral angle (radians) from 4 points."""
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        sin_angle = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))

        return np.arctan2(sin_angle, cos_angle)

    def _compute_physicochemical(self, residues) -> np.ndarray:
        """Compute KD hydrophobicity, net charge, HBD, HBA."""
        N = len(residues)
        feats = np.zeros((N, 4), dtype=np.float32)

        for i, res in enumerate(residues):
            aa = res.get_resname()
            feats[i, 0] = KYTE_DOOLITTLE.get(aa, 0.0)  # Kyte-Doolittle
            feats[i, 1] = self._net_charge_ph7(aa)
            feats[i, 2] = float(self._hbond_donors(aa))
            feats[i, 3] = float(self._hbond_acceptors(aa))

        return feats

    def _net_charge_ph7(self, aa: str) -> float:
        """Estimate net charge at pH 7 using Henderson-Hasselbalch."""
        charge = 0.0
        if aa == 'LYS':
            pKa = pKa_TABLE['LYS']
            charge += 1.0 / (1.0 + 10 ** (7 - pKa))  # protonated
        elif aa == 'ARG':
            pKa = pKa_TABLE['ARG']
            charge += 1.0 / (1.0 + 10 ** (7 - pKa))
        elif aa == 'HIS':
            pKa = pKa_TABLE['HIS']
            charge += 1.0 / (1.0 + 10 ** (7 - pKa))
        elif aa == 'ASP':
            pKa = pKa_TABLE['ASP']
            charge -= 1.0 / (1.0 + 10 ** (pKa - 7))  # deprotonated
        elif aa == 'GLU':
            pKa = pKa_TABLE['GLU']
            charge -= 1.0 / (1.0 + 10 ** (pKa - 7))
        return charge

    def _hbond_donors(self, aa: str) -> int:
        """Count HBond donors: N, S, O in side chain + backbone N."""
        donors = {'N': 1, 'S': 0, 'T': 1, 'Y': 1}  # rough estimate
        return donors.get(aa, 0) + 1  # +1 for backbone N

    def _hbond_acceptors(self, aa: str) -> int:
        """Count HBond acceptors: O, N in side chain + backbone O."""
        acceptors = {
            'D': 2, 'E': 2, 'N': 1, 'Q': 1, 'S': 1, 'T': 1, 'Y': 1,
        }
        return acceptors.get(aa, 0) + 1  # +1 for backbone O

    def _compute_exposure(
        self, chain: Chain, structure: Structure, CA_coords: np.ndarray, residues
    ) -> np.ndarray:
        """Compute SASA, relative SASA, HSE, depth."""
        N = len(residues)
        feats = np.zeros((N, 5), dtype=np.float32)

        if not self.compute_sasa or structure is None:
            return feats  # return zeros if SASA not available

        try:
            # Compute SASA using freesasa
            result = freesasa.calcBioPDB(structure)
            areas = result.residueAreas()

            for i, res in enumerate(residues):
                chain_id = chain.id
                res_num = str(res.id[1])
                try:
                    sasa = areas[chain_id][res_num].total
                    aa = res.get_resname()
                    max_sasa = AA_MAX_SASA.get(aa, 200.0)
                    feats[i, 0] = sasa / 200.0  # normalise
                    feats[i, 1] = sasa / max_sasa  # relative
                except Exception:
                    pass

            # Simplified depth: distance from surface (heuristic)
            r_com = CA_coords.mean(axis=0)
            for i in range(N):
                depth = np.linalg.norm(CA_coords[i] - r_com)
                feats[i, 4] = depth / 20.0  # normalise

        except Exception as e:
            # SASA computation failed, return zeros
            pass

        return feats

    def _compute_pmp_features(
        self, residues, CA_coords: np.ndarray, exposure_feats: np.ndarray
    ) -> np.ndarray:
        """Compute membrane depth, membrane SASA, amphipathic score, RMSF."""
        N = len(residues)
        feats = np.zeros((N, 4), dtype=np.float32)

        # Membrane depth: (r - COM) · n̂
        r_com = CA_coords.mean(axis=0)
        depth = (CA_coords - r_com) @ self.opm_normal
        feats[:, 0] = depth / 20.0  # normalise

        # Membrane-facing SASA: heuristic — SASA on membrane-normal side
        feats[:, 1] = exposure_feats[:, 0]  # use abs SASA as proxy

        # Amphipathic helix score: Eisenberg moment over 7-residue window
        for i in range(N):
            amph = self._amphipathic_score(residues, i)
            feats[i, 2] = amph

        # RMSF from MD (if provided)
        if self.rmsf is not None:
            feats[:, 3] = self.rmsf / 5.0  # normalise

        return feats

    def _amphipathic_score(self, residues, center_idx: int, window: int = 7) -> float:
        """Eisenberg hydrophobic moment for amphipathic helix detection."""
        start = max(0, center_idx - window // 2)
        end = min(len(residues), center_idx + window // 2 + 1)
        window_res = residues[start:end]

        if len(window_res) < 4:
            return 0.0

        # Hydrophobicity values
        H = np.array([KYTE_DOOLITTLE.get(r.get_resname(), 0.0) for r in window_res])

        # Helix wheel: 100° per residue for α-helix
        angles = np.arange(len(H)) * np.radians(100)
        moment_x = (H @ np.cos(angles)) ** 2
        moment_y = (H @ np.sin(angles)) ** 2
        moment = np.sqrt(moment_x + moment_y) / len(H)

        return min(1.0, moment / 3.0)  # normalise to ~[0, 1]

    def _build_backbone_frame(
        self, N_coords: np.ndarray, CA_coords: np.ndarray, C_coords: np.ndarray
    ) -> np.ndarray:
        """Build orthonormal backbone frame (N→CA, CA→C, normal)."""
        N = CA_coords.shape[0]
        frame = np.zeros((N, 3, 3), dtype=np.float32)

        for i in range(N):
            u1 = CA_coords[i] - N_coords[i]
            u2 = C_coords[i] - CA_coords[i]
            u1 = u1 / (np.linalg.norm(u1) + 1e-8)
            u2 = u2 / (np.linalg.norm(u2) + 1e-8)

            u3 = np.cross(u1, u2)
            u3 = u3 / (np.linalg.norm(u3) + 1e-8)

            # Re-orthogonalise u2
            u2 = np.cross(u3, u1)

            frame[i, 0, :] = u1
            frame[i, 1, :] = u2
            frame[i, 2, :] = u3

        return frame

    def _compute_virtual_cbeta(
        self, N_coords: np.ndarray, CA_coords: np.ndarray, C_coords: np.ndarray
    ) -> np.ndarray:
        """Compute virtual Cβ from backbone atoms (works for Gly)."""
        N = CA_coords.shape[0]
        cb_dir = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            b = CA_coords[i] - N_coords[i]
            c = C_coords[i] - CA_coords[i]
            a = np.cross(b, c)
            CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA_coords[i]
            direction = CB - CA_coords[i]
            cb_dir[i] = direction / (np.linalg.norm(direction) + 1e-8)

        return cb_dir
