import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample IntrA generated meshes and combine them into HDF5."
    )
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "intrA.h5"
    )
    return parser.parse_args()


def sample_meshes(folder, label, num_points, data, labels, ids):
    for path in sorted(folder.glob("*.obj")):
        mesh = trimesh.load_mesh(path, process=False)
        data.append(mesh.sample(num_points))
        labels.append(label)
        ids.append(path.stem)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    generated = ROOT / "data" / "generated"
    data, labels, ids = [], [], []
    sample_meshes(
        generated / "vessel" / "obj", 0, args.num_points, data, labels, ids
    )
    sample_meshes(
        generated / "aneurysm" / "obj", 1, args.num_points, data, labels, ids
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("data", data=np.asarray(data))
        handle.create_dataset(
            "label", data=np.asarray(labels, dtype=np.int64).reshape(-1, 1)
        )
        handle.create_dataset(
            "id", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8"))
        )

    print(f"Saved {len(data)} point clouds to {args.output}")


if __name__ == "__main__":
    main()
