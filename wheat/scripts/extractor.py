import trimesh
import tifffile
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from skimage import measure

from scripts.utils import save_ply


class ShapeExtractor:
    """
    Extract point clouds from binary TIFF volumes.

    Examples
    --------
    >>> extractor = ShapeExtractor()
    >>> result = extractor.extract("sample.tif", target_points=2048)
    >>> save_ply(result["points"], "sample.ply")
    """

    def __init__(
        self, seed=42, min_volume=1000, min_shape=(2, 2, 2), keep_largest_component=True
    ):
        """
        Initialize the extractor.

        Parameters
        ----------
        seed : int, optional
            Random seed for FPS sampling (default: 42)
        min_volume : int, optional
            Minimum foreground voxel count required for processing (default: 1000)
        min_shape : tuple of int, optional
            Minimum shape (z, y, x) required for processing (default: (2, 2, 2))
        keep_largest_component : bool, optional
            If True, keeps only the largest connected component from
            the binary mask (default: True)
        """
        self.seed = seed
        self.min_volume = min_volume
        self.min_shape = min_shape
        self.keep_largest_component = keep_largest_component

    def extract(self, tiff_path, target_points=None):
        """
        Extract point cloud from a binary TIFF file.

        Parameters
        ----------
        tiff_path : str or Path
            Path to TIFF file.
        target_points : int, optional
            If provided, resample vertices to this number:
            - downsample with FPS when points are more than target
            - upsample with smooth interpolation when points are fewer

        Returns
        -------
        dict
            Result dictionary containing:
            - ok: whether extraction succeeded
            - skip_reason: reason when skipped
            - points: ndarray of shape (N, 3)
        """
        tiff_path = Path(tiff_path)

        binary = self._load_binary_volume(tiff_path)
        mesh, skip_reason = self._extract_mesh(binary, name=tiff_path.stem)

        if skip_reason is not None:
            return {
                "ok": False,
                "skip_reason": skip_reason,
                "points": None,
                "binary": binary,
                "mesh": None,
            }

        points = np.asarray(mesh.vertices)

        if target_points is not None:
            if len(points) > target_points:
                points = self._fps_downsample(points, target_points)
            elif len(points) < target_points:
                points = self._upsample_points(points, target_points)

        return {
            "ok": True,
            "skip_reason": None,
            "points": points,
            "binary": binary,
            "mesh": mesh,
        }

    def _process_single(
        self,
        tiff_file: Path,
        input_path: Path,
        ply_output_path: Path,
        npy_output_path: Path,
        target_points: int,
    ):
        """Process a single TIFF and save both point cloud (PLY) and binary (NPY)."""
        result = self.extract(tiff_file, target_points=target_points)
        binary_shape = result["binary"].shape

        # Determine output paths based on input structure
        relative_path = tiff_file.relative_to(input_path)
        ply_file = ply_output_path / relative_path.with_suffix(".ply")
        npy_file = npy_output_path / relative_path.with_suffix(".npy")
        ply_file_str = ""
        npy_file_str = ""

        # save binary
        npy_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_file, result["binary"])
        npy_file_str = str(npy_file)

        # save point cloud if extraction succeeded
        if result["ok"]:
            ply_file.parent.mkdir(parents=True, exist_ok=True)
            save_ply(result["points"], ply_file)
            ply_file_str = str(ply_file)

        return {
            "id": tiff_file.stem,
            "class": tiff_file.parent.name,
            "shape_z": int(binary_shape[0]),
            "shape_y": int(binary_shape[1]),
            "shape_x": int(binary_shape[2]),
            "volume": int(np.sum(result["binary"])),
            "saved": bool(result["ok"]),
            "skipped": bool(not result["ok"]),
            "skip_reason": result["skip_reason"] or "",
            "source": str(tiff_file),
            "output_ply": ply_file_str,
            "output_npy": npy_file_str,
        }

    def batch_process(
        self,
        input_dir,
        output_dir,
        metadata_path="pointcloud_index.csv",
        target_points=2048,
        pattern="**/*.tif",
    ):
        """
        Process all TIFF files in a directory tree.

        Parameters
        ----------
        input_dir : str or Path
            Root directory containing TIFF files.
        output_dir : str or Path
            Root directory to save PLY and NPY outputs. Will create
            'pointclouds' and 'arrays' subdirectories.
        metadata_path : str or Path, optional
            CSV path for processing index (default: pointcloud_index.csv)
        target_points : int, optional
            Target point count after FPS downsampling for PLY (default: 2048).
        pattern : str, optional
            TIFF matching pattern (default: **/*.tif)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        ply_output_path = output_path / "pointclouds"
        npy_output_path = output_path / "arrays"
        metadata_path = Path(metadata_path)

        tiff_files = sorted(
            list(input_path.glob(pattern)),
            key=lambda path: (path.parent.as_posix(), int(path.stem)),
        )
        if not tiff_files:
            print(f"No files found matching pattern: {pattern}")
            return

        print(f"Found {len(tiff_files)} files")
        print(f"Processing with target_points={target_points}\n")

        csv_exists = metadata_path.exists()
        skipped_count = 0

        for tiff_file in tqdm(tiff_files, total=len(tiff_files), desc="Processing"):
            metadata_row = self._process_single(
                tiff_file,
                input_path,
                ply_output_path,
                npy_output_path,
                target_points,
            )

            if metadata_row["skipped"]:
                skipped_count += 1

            df_row = pd.DataFrame([metadata_row])

            if not csv_exists:
                df_row.to_csv(metadata_path, mode="w", header=True, index=False)
                csv_exists = True
            else:
                df_row.to_csv(metadata_path, mode="a", header=False, index=False)

        print(f"\n✓ Metadata saved to: {metadata_path}")
        print(f"✓ Skipped samples: {skipped_count}")
        print(f"✓ PLY saved under: {ply_output_path}")
        print(f"✓ NPY saved under: {npy_output_path}")

    def _load_binary_volume(self, tiff_path):
        """
        Load TIFF and convert to binary mask.

        Since data is already binarized, conversion is done with `> 0` only.
        """
        volume = tifffile.imread(str(tiff_path))
        binary = volume > 0

        if self.keep_largest_component:
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            if regions:
                largest = max(regions, key=lambda region: region.area)
                binary = labeled == largest.label

        return binary

    def _extract_mesh(self, binary, name=None):
        """Extract mesh from binary volume using marching cubes."""
        # Check if foreground voxel volume is large enough
        foreground_volume = int(np.count_nonzero(binary))
        if foreground_volume < self.min_volume:
            msg_prefix = f"[{name}] " if name else ""
            print(
                f"{msg_prefix}Warning: Foreground volume {foreground_volume} < min_volume {self.min_volume}. Skipping sample."
            )
            return None, "volume_too_small"

        # Check if shape is large enough
        if any(dim < min_dim for dim, min_dim in zip(binary.shape, self.min_shape)):
            msg_prefix = f"[{name}] " if name else ""
            print(
                f"{msg_prefix}Warning: Shape {binary.shape} has dimension smaller than min_shape {self.min_shape}. Skipping sample."
            )
            return None, "shape_too_small"

        # Extract mesh with marching cubes
        verts, faces, _, _ = measure.marching_cubes(
            binary.astype(float), level=0.5, spacing=(1.0, 1.0, 1.0)
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()

        # if not mesh.is_watertight:
        #     msg_prefix = f"[{name}] " if name else ""
        #     print(f"{msg_prefix}Warning: Extracted mesh is not watertight.")

        return mesh, None

    def _fps_downsample(self, points, target_count):
        """
        Farthest Point Sampling for uniform point distribution.

        Parameters
        ----------
        points : ndarray
            Input points, shape (N, 3)
        target_count : int
            Number of points to sample

        Returns
        -------
        ndarray
            Sampled points, shape (target_count, 3)
        """
        n = len(points)
        if target_count >= n:
            return points

        sampled_indices = np.zeros(target_count, dtype=int)
        distances = np.full(n, np.inf)

        rng = np.random.RandomState(self.seed)
        current_idx = rng.randint(n)
        sampled_indices[0] = current_idx

        for i in range(1, target_count):
            last_point = points[current_idx]
            dists = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dists)
            current_idx = np.argmax(distances)
            sampled_indices[i] = current_idx

        return points[sampled_indices]

    def _upsample_points(self, points, target_count):
        """
        Upsample points by smooth interpolation between nearby points.

        Parameters
        ----------
        points : ndarray
            Input points, shape (N, 3)
        target_count : int
            Number of points after upsampling

        Returns
        -------
        ndarray
            Upsampled points, shape (target_count, 3)
        """
        n = len(points)
        if n == 0 or target_count <= n:
            return points

        if n == 1:
            return np.repeat(points, target_count, axis=0)

        rng = np.random.RandomState(self.seed)
        extra_count = target_count - n

        anchor_idx = rng.randint(0, n, size=extra_count)
        anchors = points[anchor_idx]

        # Pick a nearby partner from random candidates for each anchor
        candidate_idx = rng.randint(0, n, size=(extra_count, 8))
        candidates = points[candidate_idx]
        distances = np.linalg.norm(candidates - anchors[:, None, :], axis=2)
        nearest_pos = np.argmin(distances, axis=1)
        partner_idx = candidate_idx[np.arange(extra_count), nearest_pos]
        partners = points[partner_idx]

        # Interpolate on local line segments to generate smooth new points
        alpha = rng.uniform(0.25, 0.75, size=(extra_count, 1))
        extra_points = alpha * anchors + (1.0 - alpha) * partners

        return np.vstack([points, extra_points])
