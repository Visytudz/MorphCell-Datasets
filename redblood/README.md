# Red blood cell dataset

This repository contains the red blood cell data and preprocessing code used in MorphCell. The dataset includes 620 three-dimensional TIFF volumes from seven morphological classes.

The original data are available from [Zenodo](https://doi.org/10.5281/zenodo.4670205). Additional information is provided by the [cytoShapeNet repository](https://github.com/kgh-85/cytoShapeNet).

## Structure

```text
redblood/
├── data/
│   ├── volumes/        TIFF volumes grouped by class
│   ├── binary/         binary masks derived from the TIFF volumes
│   └── pointclouds/    point clouds derived from the binary masks
├── notebooks/          preprocessing workflow
├── scripts/            point-cloud extraction and HDF5 utilities
```
