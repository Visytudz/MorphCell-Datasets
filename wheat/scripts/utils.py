import os
import warnings
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import plotly.graph_objects as go

from typing import List, Tuple


def save_ply(points: np.ndarray, filename: str) -> None:
    """Save a point cloud to a PLY file."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_points = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, points, fmt="%.10f")


def load_ply(file_path: str) -> np.ndarray:
    """Load a .ply file into a numpy array."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Please install open3d to load .ply files: pip install open3d"
        )
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points).astype(np.float32)


def plot_ply(points: np.ndarray, title: str = "Point Cloud"):
    """Plot 3D point cloud using Plotly."""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=points[:, 2], colorscale="Viridis", showscale=True
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        height=700,
    )

    fig.show()


def gather_plys(root: Path, label_map: dict) -> List[Tuple[Path, str, int, int]]:
    """Collect PLY files whose parent directory is present in ``label_map``."""
    plys = list(root.glob("**/*.ply"))
    items = []
    skipped = {}
    for path in plys:
        class_name = path.parent.name
        if class_name not in label_map:
            skipped[class_name] = skipped.get(class_name, 0) + 1
            continue
        numeric_id = int(path.stem)
        items.append((path, class_name, label_map[class_name], numeric_id))

    if skipped:
        summary = ", ".join(
            f"{class_name} ({count})"
            for class_name, count in sorted(skipped.items())
        )
        warnings.warn(
            f"Skipped PLY files from classes absent from label_map: {summary}",
            stacklevel=2,
        )

    items.sort(key=lambda x: (x[2], x[3]))
    return items


def build_hdf5(
    input_root: Path,
    output_path: Path,
    label_map: dict,
    id_length: int = 4,
) -> None:
    """Build HDF5 file from PLY files."""
    items = gather_plys(input_root, label_map)
    if not items:
        raise ValueError(
            f"No PLY files under {input_root} match the classes in label_map."
        )
    count = len(items)
    num_points = len(load_ply(items[0][0]))

    # Preallocate arrays
    data = np.zeros((count, num_points, 3), dtype=np.float32)
    labels = np.zeros((count, 1), dtype=np.int64)
    ids = np.empty((count,), dtype=f"S{id_length}")

    # Load data
    for idx, (path, class_name, label, numeric_id) in enumerate(
        tqdm(items, desc="Loading PLY", unit="cell")
    ):
        points = load_ply(path)
        data[idx] = points
        labels[idx, 0] = label
        ids[idx] = str(numeric_id).encode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("id", data=ids)
        f.create_dataset("label", data=labels)

    print(f"Written {count} samples to {output_path}")
