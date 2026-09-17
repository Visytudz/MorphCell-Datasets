# Wheat root cell dataset

This workspace contains 1,127 3D wheat root cells.

## Structure

```text
data/masks/         TIFF masks
data/arrays/        binary NumPy arrays generated from the masks
data/pointclouds/   2,048-point PLY files generated from the masks
scripts/            mask-to-point-cloud and HDF5 utilities
notebooks/          data-processing workflow
```

Processing order: `TIFF masks → NumPy arrays and PLY point clouds → HDF5`.

## Scale

The TIFF masks have ZYX order and isotropic spacing of 0.7 µm/voxel. NPY masks, PLY point clouds and HDF5 point clouds retain voxel-scale coordinates; no physical scaling is applied. Accordingly, shape descriptors are in voxel-derived units.
