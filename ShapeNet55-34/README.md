# ShapeNet55-34

ShapeNet55-34 is used as the point-cloud pretraining dataset for MorphCell. This copy comes from the processed dataset distributed by [Point-BERT](https://github.com/Julie-tang00/Point-BERT).

## ShapeNet-55 and ShapeNet-34

- **ShapeNet-55** uses all 55 object categories for training and testing.
- **ShapeNet-34** uses 34 categories as seen categories; the remaining 21 categories are reserved for testing generalization to unseen categories.

They use the same point-cloud collection but different split files. MorphCell uses **ShapeNet-55**.

## Download

Follow the [Point-BERT data instructions](https://github.com/Julie-tang00/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md) to download the processed dataset.

## Downloaded structure

Point-BERT uses the following files:

```text
ShapeNet55-34/
├── shapenet_pc/
│   ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│   └── ...
└── ShapeNet-55/
    ├── train.txt
    └── test.txt
```

Each `.npy` file stores one point cloud as an `(8192, 3)` XYZ array. File names follow `{synset_id}-{model_id}.npy`, and the split files contain these complete file names.
