# Colon cell dataset

Paired membrane (MEM) and nuclear (NUC) segmentations from 70 human colon epithelial cells, including 47 cancerous cells and 23 normal cells.

## Structure

```text
data/slices/    per-slice segmentation images
data/masks/     two-channel TIFF masks and metadata
data/meshes/    MEM and NUC meshes
data/pointclouds/  2,048-point MEM and NUC PLY files generated from the masks
scripts/        feature and point-cloud extraction code
notebooks/      data preprocessing and morphology analysis
```

Processing order: `slices → TIFF masks → meshes → morphology features and point clouds`.

## Scale

`data/slices/` contains the original-scale per-slice segmentation images without spatial resizing. Their XY image dimensions are 20 times those of the TIFF masks. These slices are not included in the public release.

The TIFF masks use ZCYX order with MEM and NUC channels. They retain the slice count in Z and reduce XY by a factor of 20. To keep Z aligned with the scaled XY coordinates, Z spacing is divided by 20, from 50 nm to 2.5 nm. The stored ZYX spacing is `(2.5, 5.61523, 5.61523)` nm for cancerous cells and `(2.5, 7.324, 7.324)` nm for normal cells.

The OBJ meshes use the same scaled coordinate system as the TIFF masks and are stored at 1/20 of physical linear scale. `scripts/extractor.py` multiplies mesh coordinates by 20 before calculating morphology features.

PLY and HDF5 point clouds are sampled directly from the OBJ meshes and retain the same 1/20-scale coordinates. The point-cloud section in `notebooks/main.ipynb` generates `data/pointclouds/MEM/`, `data/pointclouds/NUC/`, `outputs/MEM.hdf5` and `outputs/NUC.hdf5`. Each HDF5 file contains `data`, `id` and `label`, with `cancerous=0` and `normal=1`. Set `mesh_scale=20` only when physical-scale coordinates are required.
