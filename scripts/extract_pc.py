#!/usr/bin/env python3
import numpy as np
import struct
import argparse

def main():
    p = argparse.ArgumentParser(
        description="Extract point-cloud from .npz into simple binary")
    p.add_argument("npz", help="ShapeNet .npz file (keys: 'pc', 'part_label')")
    p.add_argument("out", help="Output binary file, e.g. cloud.bin")
    args = p.parse_args()

    data = np.load(args.npz)
    pc = data['pc']                     # shape (M,3), dtype float32/float64
    labels = data['part_label'].astype(np.int32)  # shape (M,)

    M = pc.shape[0]
    with open(args.out, 'wb') as f:
        # 1) число точек
        f.write(struct.pack('<I', M))
        # 2) для каждой точки: x,y,z (float32) + label (int32)
        for i in range(M):
            x,y,z = pc[i]
            lbl   = labels[i]
            f.write(struct.pack('<3fI', float(x), float(y), float(z), int(lbl)))

    print(f"Wrote {M} points to {args.out}")

if __name__ == "__main__":
    main()
