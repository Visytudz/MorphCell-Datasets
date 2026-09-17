import trimesh
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import filters, morphology, measure
from scipy.ndimage import gaussian_filter, binary_fill_holes
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.utils import save_ply


class ShapeExtractor:
    """
    Extract 3D shape features from TIFF volumes.

    Examples
    --------
    >>> extractor = ShapeExtractor()
    >>> result = extractor.extract('cell.tif', target_points=2048)
    >>> save_ply(result['points'], 'output.ply')
    """

    def __init__(self, smooth=1.0, seed=42, roughness_smooth_iter=10):
        """
        Initialize the ShapeExtractor.

        Parameters
        ----------
        smooth : float, optional
            Gaussian smoothing factor for volume preprocessing (default: 1.0)
        seed : int, optional
            Random seed for FPS sampling (default: 42)
        roughness_smooth_iter : int, optional
            Smoothing iterations for roughness computation.
            Higher values filter out more noise but may shrink the mesh (default: 10)
        """
        self.smooth = smooth
        self.seed = seed
        self.roughness_smooth_iter = roughness_smooth_iter

    def extract(
        self,
        tiff_path,
        target_points=None,
        save_geo=False,
        save_texture=False,
        save_binary=False,
        binary_output_path=None,
    ):
        """
        Extract point cloud and shape features from TIFF file.

        Parameters
        ----------
        tiff_path : str or Path
            Path to TIFF file
        target_points : int, optional
            If specified, downsample to this many points using FPS
        save_geo : bool, optional
            Whether to calculate and return geometric shape features (default: False)
        save_texture : bool, optional
            Whether to calculate and return texture features (default: False)
        save_binary : bool, optional
            Whether to save the preprocessed binary mask (default: False)
        binary_output_path : str or Path, optional
            Output path for the binary mask when save_binary is True

        Returns
        -------
        result : dict
            Dictionary containing points and comprehensive shape features
        """
        tiff_path = Path(tiff_path)

        # Step 1: Load and preprocess volume
        binary = self._preprocess_volume(tiff_path)
        if save_binary:
            if binary_output_path is None:
                raise ValueError("binary_output_path must be provided when save_binary=True")
            binary_output_path = Path(binary_output_path)
            binary_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(binary_output_path, binary)

        # Step 2: Extract mesh
        # Mesh is extracted at original resolution to preserve micro-textures
        mesh = self._extract_mesh(binary, name=tiff_path.stem)

        # Step 3: Compute features
        features = {
            "points": self._sample_points(mesh, target_points),
        }

        if save_geo:
            geom_features = self._compute_geometric_features(mesh, binary)
            features.update(geom_features)

        if save_texture:
            texture_features = self._compute_texture_features(
                mesh, smooth_iter=self.roughness_smooth_iter
            )
            features.update(texture_features)

        return features

    def process_single(
        self,
        tiff_file,
        input_path,
        ply_output_path,
        target_points,
        save_geo,
        save_texture,
        save_binary,
        binary_output_path,
    ):
        """Helper for batch processing a single file."""
        relative_path = tiff_file.relative_to(input_path)
        binary_output_file = None
        if save_binary:
            binary_output_file = (
                Path(binary_output_path) / relative_path.with_suffix(".npy")
            )

        result = self.extract(
            tiff_file,
            target_points=target_points,
            save_geo=save_geo,
            save_texture=save_texture,
            save_binary=save_binary,
            binary_output_path=binary_output_file,
        )

        # Save point cloud
        output_file = ply_output_path / relative_path.with_suffix(".ply")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_ply(result["points"], output_file)

        # Return metadata if needed
        if save_geo or save_texture:
            return {
                "id": tiff_file.stem,
                "class": tiff_file.parent.name,
                **{k: v for k, v in result.items() if k != "points"},
            }
        return None

    def batch_process(
        self,
        input_dir,
        ply_output_dir,
        metadata_path="morph.csv",
        target_points=2048,
        pattern="**/*.tif",
        save_geo=False,
        save_texture=False,
        save_binary=False,
        binary_output_dir="binary",
        num_workers=4,
    ):
        """
        Process all TIFF files in a directory tree.

        Parameters
        ----------
        input_dir : str or Path
            Root directory with TIFF files
        ply_output_dir : str or Path
            Output directory for PLY files (preserves structure)
        metadata_path : str or Path, optional
            Output path for metadata CSV (default: "morph.csv")
        target_points : int, optional
            Target point count (default: 2048)
        pattern : str, optional
            File pattern to match (default: **/*.tif)
        save_geo : bool, optional
            Whether to calculate and save geometric shape features (default: False)
        save_texture : bool, optional
            Whether to calculate and save texture features (default: False)
        save_binary : bool, optional
            Whether to save the preprocessed binary masks (default: False)
        binary_output_dir : str or Path, optional
            Output directory for binary masks (default: "binary")
        num_workers : int, optional
            Number of parallel workers (default: 4)
        """
        input_path = Path(input_dir)
        ply_output_path = Path(ply_output_dir)
        binary_output_path = Path(binary_output_dir) if save_binary else None

        # Find all TIFF files
        tiff_files = list(input_path.glob(pattern))
        if not tiff_files:
            print(f"No files found matching pattern: {pattern}")
            return
        print(f"Found {len(tiff_files)} files")
        print(
            f"Processing with target_points={target_points}, num_workers={num_workers}\n"
        )

        # Create metadata CSV path
        save_metadata = save_geo or save_texture
        csv_exists = False
        if save_metadata:
            csv_exists = Path(metadata_path).exists()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self.process_single,
                    tiff_file,
                    input_path,
                    ply_output_path,
                    target_points,
                    save_geo,
                    save_texture,
                    save_binary,
                    binary_output_path,
                ): tiff_file
                for tiff_file in tiff_files
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                metadata_row = future.result()

                # Save metadata
                if save_metadata and metadata_row:
                    # Create dataframe for this row
                    df_row = pd.DataFrame([metadata_row])

                    # Write header on first write, append afterwards
                    if not csv_exists:
                        df_row.to_csv(
                            Path(metadata_path), mode="w", header=True, index=False
                        )
                        csv_exists = True
                    else:
                        df_row.to_csv(
                            Path(metadata_path), mode="a", header=False, index=False
                        )

        if save_metadata:
            print(f"\n✓ Metadata saved to: {metadata_path}")
        if save_binary:
            print(f"✓ Binary masks saved to: {binary_output_path}")

        print(f"\n✓ Done! PLY files saved to: {ply_output_path}")

    def _preprocess_volume(self, tiff_path):
        """Preprocess TIFF volume to binary mask."""
        stack = tifffile.imread(str(tiff_path))
        stack_smooth = gaussian_filter(stack.astype(float), sigma=self.smooth)

        # Adaptive thresholding
        thresh = (
            filters.threshold_otsu(stack_smooth) + filters.threshold_li(stack_smooth)
        ) / 2
        binary = stack_smooth > thresh

        # Morphological operations
        binary = binary_fill_holes(binary)
        binary = morphology.remove_small_objects(binary, min_size=100)

        # Keep largest component
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            binary = labeled == largest.label

        return binary

    def _extract_mesh(self, binary, name=None):
        """Extract mesh from binary volume using marching cubes."""
        verts, faces, _, _ = measure.marching_cubes(
            binary.astype(float), level=0.5, spacing=(1.0, 1.0, 1.0)
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()

        if not mesh.is_watertight:
            msg_prefix = f"[{name}] " if name else ""
            print(f"{msg_prefix}Warning: Extracted mesh is not watertight.")

        return mesh

    def _sample_points(self, mesh, target_points):
        """Sample points from mesh surface."""
        if target_points is None:
            return np.array(mesh.vertices)

        points = np.array(mesh.vertices)
        if len(points) > target_points:
            return self._fps_downsample(points, target_points)
        return points

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
        sampled_points : ndarray
            Sampled points, shape (target_count, 3)
        """
        n = len(points)
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

    def _compute_geometric_features(self, mesh, binary):
        """
        Compute basic geometric shape features using accurate mesh metrics and PCA.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh object (with fixed normals)
        binary : ndarray
            Binary volume mask

        Returns
        -------
        dict
            Geometric features including volume, sphericity, convexity,
            elongation, and flatness.
        """
        # Volume and Area from Mesh (more accurate than voxel sum)
        volume = abs(mesh.volume)
        surface_area = mesh.area

        # Sphericity: 1.0 for perfect sphere
        # Formula: (pi^(1/3) * (6V)^(2/3)) / A
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area

        # Convexity: ratio of mesh volume to convex hull volume
        convex_hull = mesh.convex_hull
        convexity = (
            volume / float(convex_hull.volume) if convex_hull.volume > 0 else 0.0
        )

        # --- PCA based Geometric Features ---
        vertices = mesh.vertices - mesh.centroid
        cov_matrix = np.dot(vertices.T, vertices) / len(vertices)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)
        eigenvalues = np.clip(eigenvalues, 0, None)
        L_min = np.sqrt(eigenvalues[0])
        L_mid = np.sqrt(eigenvalues[1])
        L_max = np.sqrt(eigenvalues[2])

        # Elongation: Ratio of Medium to Largest axis
        elongation = L_mid / L_max if L_max > 0 else 0.0

        # Flatness: Ratio of Smallest to Medium axis
        flatness = L_min / L_mid if L_mid > 0 else 0.0

        return {
            "volume": float(volume),
            "surface_area": float(surface_area),
            "sphericity": float(sphericity),
            "convexity": float(convexity),
            "elongation": float(elongation),
            "flatness": float(flatness),
            "axis_length_max": float(L_max),
            "axis_length_mid": float(L_mid),
            "axis_length_min": float(L_min),
        }

    def _compute_texture_features(self, mesh, smooth_iter=10):
        """
        Compute surface texture features including Roughness and Multi-scale Curvature.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The input triangular mesh.
        smooth_iter : int, optional
            Number of iterations for Laplacian smoothing (default: 10)

        Returns
        -------
        dict
            Multi-scale texture features.
        """
        # --- 1. Roughness Features ---
        # Compute displacement between original surface and smoothed surface
        mesh_smooth = mesh.copy()
        trimesh.smoothing.filter_laplacian(mesh_smooth, iterations=smooth_iter)
        displacements = np.linalg.norm(mesh.vertices - mesh_smooth.vertices, axis=1)

        # Roughness Max: Indicates the length of the longest spike (key for Echinocytes)
        # Roughness RMS: Root Mean Square, sensitive to outliers/spikes
        roughness_mean = np.mean(displacements)
        roughness_max = np.max(displacements)
        roughness_rms = np.sqrt(np.mean(displacements**2))

        # --- 2. Multi-scale Curvature Features ---
        # Fixed physical radii based on cell biology scales
        # Small (3.0): Captures fine spikes and micro-villi
        # Medium (8.0): Captures local bumps and structural protrusions
        # Large (15.0): Captures global shape curvature (e.g., concavity of discocytes)
        radii = {
            "small": 3.0,
            "medium": 8.0,
            "large": 15.0,
        }

        features = {
            "roughness_mean": float(roughness_mean),
            "roughness_max": float(roughness_max),
            "roughness_rms": float(roughness_rms),
        }

        # Compute curvature at each defined scale
        for scale_name, radius in radii.items():
            scale_features = self._compute_curvature_at_scale(mesh, radius, scale_name)
            features.update(scale_features)

        return features

    def _compute_curvature_at_scale(self, mesh, radius, scale_name):
        """
        Compute curvature features at a specific scale.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh object
        radius : float
            Neighborhood radius for curvature computation
        scale_name : str
            Scale identifier (e.g., 'small', 'medium', 'large')

        Returns
        -------
        dict
            Curvature features with scale prefix
        """
        # Mean Curvature (H)
        H = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=radius
        )

        # Gaussian Curvature (K)
        K = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=radius
        )

        # Shape Index calculation
        # S = (2/pi) * arctan((k1 + k2) / (k1 - k2))
        # Derived from H and K: k1 - k2 = 2 * sqrt(H^2 - K)
        discriminant = np.clip(H**2 - K, 0, None)
        k_diff = 2 * np.sqrt(discriminant)

        # Avoid division by zero
        safe_k_diff = np.where(k_diff < 1e-8, 1e-8, k_diff)
        shape_index = (2 / np.pi) * np.arctan((2 * H) / safe_k_diff)

        # Return statistical descriptors
        # Abs Mean H: Total bending energy proxy
        # Std H: Variation of surface curvature (chaos)
        # Std Shape Index: Entropy of surface topology types
        return {
            f"curvature_mean_{scale_name}": float(np.mean(np.abs(H))),
            f"curvature_std_{scale_name}": float(np.std(H)),
            f"gaussian_mean_{scale_name}": float(np.mean(np.abs(K))),
            f"shape_index_std_{scale_name}": float(np.nanstd(shape_index)),
        }
