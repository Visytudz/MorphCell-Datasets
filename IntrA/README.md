# IntrA

IntrA is a 3D intracranial aneurysm dataset used for vessel and aneurysm point-cloud classification. The original data and description are available from the [official IntrA repository](https://github.com/intra3d2019/IntrA).

## Structure

```text
data/complete/       103 complete brain-vessel OBJ meshes
data/annotated/      116 manually annotated aneurysm segments
data/generated/      1,694 vessel and 215 aneurysm segments
data/geodesic/       geodesic-distance matrices for the annotated segments
scripts/             HDF5 generation code
outputs/intrA.h5     combined point-cloud dataset
```

## HDF5

Run `python scripts/build_hdf5.py` from this directory to sample 2,048 surface points from each mesh in `data/generated/` and write `outputs/intrA.h5`.

The HDF5 file contains `data` with shape `(1909, 2048, 3)`, `id`, and `label`, with `vessel=0` and `aneurysm=1`. Coordinates are sampled directly from the OBJ meshes. MorphCell normalizes each point cloud to the unit sphere when loading it.
