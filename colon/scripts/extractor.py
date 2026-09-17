"""Mesh morphology feature extraction utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import trimesh
from tqdm.auto import tqdm


@dataclass
class MeshRecord:
    """Container for one mesh item."""

    cell_id: str
    channel: str
    path: Path


class Extractor:
    """Extract single-object and MEM-NUC paired morphology features.

    Parameters
    ----------
    mesh_dir : str or Path
        Directory containing mesh files named as ``<cell_id>_<channel>.obj``.
    mesh_scale_to_physical : float, default=20.0
        Uniform coordinate scale factor from mesh unit to physical unit.
    """

    def __init__(self, mesh_dir: Path | str, mesh_scale_to_physical: float = 20.0) -> None:
        self.mesh_dir = Path(mesh_dir)
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory does not exist: {self.mesh_dir}")
        if mesh_scale_to_physical <= 0:
            raise ValueError("mesh_scale_to_physical must be positive.")
        self.mesh_scale_to_physical = float(mesh_scale_to_physical)

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-12:
            return float("nan")
        return float(numerator / denominator)

    @staticmethod
    def _symmetry_ratio(a: float, b: float) -> float:
        """Compute symmetric ratio in [0, 1] when both values are positive."""
        if a <= 0 or b <= 0:
            return float("nan")
        return float(min(a, b) / max(a, b))

    def list_mesh_records(self) -> List[MeshRecord]:
        """List mesh files and parse ids/channels.

        Returns
        -------
        list of MeshRecord
            Parsed mesh metadata list.
        """
        records: List[MeshRecord] = []
        for path in sorted(self.mesh_dir.glob("*.obj")):
            stem = path.stem
            if "_" not in stem:
                continue
            cell_id, channel = stem.rsplit("_", 1)
            channel = channel.upper()
            if channel not in {"MEM", "NUC"}:
                continue
            records.append(MeshRecord(cell_id=cell_id, channel=channel, path=path))
        return records

    @staticmethod
    def _load_mesh(path: Path) -> trimesh.Trimesh:
        """Load mesh and ensure triangle mesh."""
        mesh = trimesh.load_mesh(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Unsupported mesh type: {type(mesh)} for {path}")
        mesh = mesh.copy()
        if not mesh.is_watertight:
            mesh.fill_holes()
        return mesh

    @staticmethod
    def _principal_axes(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PCA eigenvalues/eigenvectors for vertices."""
        centered = vertices - vertices.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        return eigvals, eigvecs

    def extract_single_features(
        self, path: Path | str, *, cell_id: str, channel: str
    ) -> Dict[str, float]:
        """Extract morphology features for one mesh object.

        Parameters
        ----------
        path : Path or str
            Mesh file path.
        cell_id : str
            Cell identifier.
        channel : str
            Channel identifier, expected ``MEM`` or ``NUC``.

        Returns
        -------
        dict
            Single-object feature dictionary.
        """
        # Load and clean mesh
        mesh = self._load_mesh(Path(path))
        mesh.apply_scale(self.mesh_scale_to_physical)
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()

        volume = float(abs(mesh.volume))
        area = float(mesh.area)
        centroid = mesh.centroid.astype(float)

        eigvals, eigvecs = self._principal_axes(mesh.vertices.view(np.ndarray))
        eigvals = np.clip(eigvals, 0.0, None)
        pca_lengths = 4.0 * np.sqrt(eigvals + 1e-12)
        major, middle, minor = [float(v) for v in pca_lengths]
        axis1 = eigvecs[:, 0].astype(float)

        hull = mesh.convex_hull
        hull_volume = float(abs(hull.volume))
        hull_area = float(hull.area)

        eq_diameter = (
            float((6.0 * volume / np.pi) ** (1.0 / 3.0)) if volume > 0 else float("nan")
        )
        sphere_area_same_volume = (
            float(np.pi * eq_diameter**2) if np.isfinite(eq_diameter) else float("nan")
        )
        sphericity = (
            self._safe_div(sphere_area_same_volume, area) if area > 0 else float("nan")
        )
        compactness = (
            self._safe_div(36.0 * np.pi * volume**2, area**3)
            if area > 0
            else float("nan")
        )
        asphericity = (
            self._safe_div(area, sphere_area_same_volume)
            if sphere_area_same_volume > 0
            else float("nan")
        )

        feature = {
            "cell_id": cell_id,
            "channel": channel,
            "mesh_path": str(path),
            "is_watertight": bool(mesh.is_watertight),
            # basic features
            "volume": volume,
            "surface_area": area,
            "area_to_volume": self._safe_div(area, volume),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            # spherical features
            "equivalent_sphere_diameter": eq_diameter,
            "sphericity": sphericity,
            "compactness": compactness,
            "asphericity": asphericity,
            # convex hull features
            "convex_hull_volume": hull_volume,
            "convex_hull_area": hull_area,
            "solidity": self._safe_div(volume, hull_volume),
            "surface_convexity": self._safe_div(hull_area, area),
            # PCA-based features
            "major": major,
            "middle": middle,
            "minor": minor,
            "elongation": self._safe_div(major, middle),
            "flatness": self._safe_div(middle, minor),
            "filamentarity": self._safe_div(major, minor),
            # major axis direction
            "axis1_x": float(axis1[0]),
            "axis1_y": float(axis1[1]),
            "axis1_z": float(axis1[2]),
        }
        return feature

    def extract_all_single_features(self) -> pd.DataFrame:
        """Extract single-object features from all mesh files.

        Returns
        -------
        pandas.DataFrame
            One row per mesh object.
        """
        rows: List[Dict[str, float]] = []
        records = self.list_mesh_records()
        iterator = tqdm(records, desc="Extracting single features") if tqdm else records
        for record in iterator:
            rows.append(
                self.extract_single_features(
                    record.path, cell_id=record.cell_id, channel=record.channel
                )
            )
        return pd.DataFrame(rows)

    def _build_pairs(
        self, single_df: pd.DataFrame
    ) -> Iterable[Tuple[str, pd.Series, pd.Series]]:
        """Build MEM-NUC pairs from single-object feature table."""
        grouped = single_df.groupby("cell_id", sort=True)
        for cell_id, group in grouped:
            mem = group[group["channel"] == "MEM"]
            nuc = group[group["channel"] == "NUC"]
            if len(mem) == 1 and len(nuc) == 1:
                yield cell_id, mem.iloc[0], nuc.iloc[0]

    def extract_pair_features(self, single_df: pd.DataFrame) -> pd.DataFrame:
        """Extract MEM-NUC paired comparison features.

        Parameters
        ----------
        single_df : pandas.DataFrame
            Single-object feature table from :meth:`extract_all_single_features`.

        Returns
        -------
        pandas.DataFrame
            One row per complete MEM-NUC pair.
        """
        required_cols = [
            "cell_id",
            "volume",
            "surface_area",
            "area_to_volume",
            "equivalent_sphere_diameter",
            "sphericity",
            "compactness",
            "solidity",
            "surface_convexity",
            "major",
            "middle",
            "minor",
            "elongation",
            "flatness",
            "filamentarity",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "axis1_x",
            "axis1_y",
            "axis1_z",
        ]
        missing_cols = [col for col in required_cols if col not in single_df.columns]
        if missing_cols:
            raise ValueError(
                "Cannot build pair features. Missing required single-object columns: "
                + ", ".join(missing_cols)
            )

        pos_scale_cols = [
            "volume",
            "surface_area",
            "area_to_volume",
            "equivalent_sphere_diameter",
            "major",
            "middle",
            "minor",
        ]
        bounded_shape_cols = [
            "sphericity",
            "compactness",
            "solidity",
            "surface_convexity",
            "elongation",
            "flatness",
            "filamentarity",
        ]

        rows: List[Dict[str, float]] = []
        for cell_id, mem, nuc in self._build_pairs(single_df):
            for col in pos_scale_cols:
                mem_val = float(mem[col])
                nuc_val = float(nuc[col])
                if mem_val <= 0 or nuc_val <= 0:
                    raise ValueError(
                        f"Positive scale feature `{col}` must be > 0 for log transform. "
                        f"Found mem={mem_val}, nuc={nuc_val} at cell_id={cell_id}."
                    )

            mem_center = np.array(
                [mem["centroid_x"], mem["centroid_y"], mem["centroid_z"]], dtype=float
            )
            nuc_center = np.array(
                [nuc["centroid_x"], nuc["centroid_y"], nuc["centroid_z"]], dtype=float
            )
            center_distance = float(np.linalg.norm(mem_center - nuc_center))
            norm_distance = self._safe_div(center_distance, float(mem["major"]))

            axis_mem = np.array(
                [mem["axis1_x"], mem["axis1_y"], mem["axis1_z"]], dtype=float
            )
            axis_nuc = np.array(
                [nuc["axis1_x"], nuc["axis1_y"], nuc["axis1_z"]], dtype=float
            )
            axis_mem_norm = np.linalg.norm(axis_mem)
            axis_nuc_norm = np.linalg.norm(axis_nuc)
            if axis_mem_norm < 1e-12 or axis_nuc_norm < 1e-12:
                cos_sim = float("nan")
            else:
                cos_sim = float(
                    abs(np.dot(axis_mem / axis_mem_norm, axis_nuc / axis_nuc_norm))
                )
                cos_sim = min(1.0, max(0.0, cos_sim))
            angle_deg = float(np.degrees(np.arccos(cos_sim)))

            row = {
                "cell_id": cell_id,
                "center_distance": center_distance,
                "normalized_center_distance": norm_distance,
                "major_axis_alignment_cosine": cos_sim,
                "major_axis_angle_deg": angle_deg,
            }

            # 1) absolute channel features
            absolute_cols = pos_scale_cols + bounded_shape_cols
            for col in absolute_cols:
                mem_val = float(mem[col])
                nuc_val = float(nuc[col])
                row[f"{col}_mem"] = mem_val
                row[f"{col}_nuc"] = nuc_val

            # 2) shared morphology features
            for col in pos_scale_cols:
                row[f"mean_log_{col}"] = float(
                    0.5 * (np.log(float(mem[col])) + np.log(float(nuc[col])))
                )
            for col in bounded_shape_cols:
                row[f"mean_{col}"] = float(0.5 * (float(mem[col]) + float(nuc[col])))

            # 3) channel imbalance features
            for col in pos_scale_cols:
                row[f"delta_log_{col}"] = float(
                    np.log(float(mem[col])) - np.log(float(nuc[col]))
                )
            for col in bounded_shape_cols:
                row[f"delta_{col}"] = float(float(mem[col]) - float(nuc[col]))

            rows.append(row)
        return pd.DataFrame(rows)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full extraction pipeline.

        Returns
        -------
        tuple of pandas.DataFrame
            ``(single_features, pair_features)``.
        """
        single_df = self.extract_all_single_features()
        pair_df = self.extract_pair_features(single_df)
        return single_df, pair_df
