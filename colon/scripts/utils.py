"""Point-cloud and HDF5 utilities for the colon dataset."""

from __future__ import annotations

import csv
import zlib
from pathlib import Path

import h5py
import numpy as np
import trimesh


CHANNELS = ("MEM", "NUC")
LABEL_MAP = {"cancerous": 0, "normal": 1}


def _sample_seed(base_seed: int, cell_id: str, channel: str) -> int:
    key = f"{cell_id}_{channel}".encode("utf-8")
    return (base_seed + zlib.crc32(key)) % (2**32)


def sample_mesh(
    mesh_path: Path, num_points: int, seed: int, mesh_scale: float = 1.0
) -> np.ndarray:
    """Sample points uniformly by surface area from an OBJ mesh."""
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Invalid triangle mesh: {mesh_path}")
    if mesh_scale <= 0:
        raise ValueError("mesh_scale must be positive")
    if mesh_scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(mesh_scale)
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return np.asarray(points, dtype=np.float32)


def save_ply(points: np.ndarray, output_path: Path) -> None:
    """Save an XYZ point array as an ASCII PLY file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float32))
    cloud.export(output_path, file_type="ply", encoding="ascii")


def load_ply(path: Path) -> np.ndarray:
    """Load XYZ coordinates from a PLY point cloud."""
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point cloud shape {points.shape}: {path}")
    return points


def read_metadata(path: Path) -> list[dict[str, str]]:
    """Read and validate the colon sample metadata."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "group"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Metadata must contain {sorted(required)}: {path}")
    unknown = sorted({row["group"] for row in rows} - LABEL_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown groups in metadata: {unknown}")
    return sorted(rows, key=lambda row: row["sample_id"])


def generate_plys(
    mesh_dir: Path,
    pointcloud_dir: Path,
    rows: list[dict[str, str]],
    num_points: int = 2048,
    seed: int = 42,
    mesh_scale: float = 1.0,
) -> None:
    """Generate one PLY file per cell and channel."""
    for row in rows:
        cell_id = row["sample_id"]
        for channel in CHANNELS:
            mesh_path = mesh_dir / f"{cell_id}_{channel}.obj"
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            points = sample_mesh(
                mesh_path,
                num_points=num_points,
                seed=_sample_seed(seed, cell_id, channel),
                mesh_scale=mesh_scale,
            )
            save_ply(points, pointcloud_dir / channel / f"{cell_id}.ply")


def build_hdf5(
    pointcloud_dir: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    num_points: int = 2048,
    mesh_scale: float = 1.0,
) -> None:
    """Aggregate MEM and NUC PLY files into separate HDF5 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([row["sample_id"].encode("utf-8") for row in rows], dtype="S9")
    labels = np.asarray(
        [[LABEL_MAP[row["group"]]] for row in rows], dtype=np.int64
    )

    for channel in CHANNELS:
        clouds = []
        for row in rows:
            path = pointcloud_dir / channel / f'{row["sample_id"]}.ply'
            points = load_ply(path)
            if points.shape != (num_points, 3):
                raise ValueError(
                    f"Expected {(num_points, 3)}, found {points.shape}: {path}"
                )
            clouds.append(points)

        data = np.stack(clouds).astype(np.float32, copy=False)
        output_path = output_dir / f"{channel}.hdf5"
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset(
                "data", data=data, compression="gzip", compression_opts=4
            )
            handle.create_dataset("id", data=ids)
            handle.create_dataset("label", data=labels)
            handle.attrs["channel"] = channel
            handle.attrs["num_points"] = num_points
            handle.attrs["label_0"] = "cancerous"
            handle.attrs["label_1"] = "normal"
            handle.attrs["mesh_scale"] = mesh_scale
            handle.attrs["coordinate_system"] = "scaled OBJ coordinates"
