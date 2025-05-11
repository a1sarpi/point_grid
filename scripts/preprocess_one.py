#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import numpy as np

def make_bin(npz_path, bin_path):
    """Вызывает extract_pc.py, чтобы из .npz получить .bin"""
    cmd = [
        sys.executable,        # путь к python
        os.path.join("scripts","extract_pc.py"),
        npz_path,
        bin_path
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_voxelizer(bin_path, voxelizer_exe, ply_out):
    """Запускает voxelize_bin.exe на cloud.bin → .ply"""
    cmd = [voxelizer_exe, bin_path, ply_out]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def save_cls(npz_path, cls_out):
    """Считывает из .npz глобальный класс и пишет его в cls_out"""
    data = np.load(npz_path, allow_pickle=True)
    # в .npz может быть 'label_cls', 'category' или 'cls'
    for key in ("label_cls","category","cls"):
        if key in data.files:
            val = int(data[key])
            with open(cls_out, "w") as f:
                f.write(f"{val}\n")
            print(f"-> wrote class {val} to {cls_out}")
            return
    print("⚠️ no classification key in .npz", data.files)

def save_seg(npz_path, seg_out, vox_n=32):
    """
    Берёт part_label из .npz, делает majority‐vote по вокселям
    (те же вычисления, что в Voxelizer.cpp),
    и сохраняет D*D*D строк с метками.
    """
    data = np.load(npz_path, allow_pickle=True)
    pts = data["pc"]
    seg = data.get("part_label", data.get("labels"))
    if seg is None:
        print("⚠️ no segmentation labels in .npz")
        return

    # find bounds
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = maxs - mins + 1e-8

    # assign each point to a voxel index
    voxmap = {}
    N = vox_n
    for (x,y,z), lbl in zip(pts, seg):
        i = min(int((x-mins[0])/spans[0]*N), N-1)
        j = min(int((y-mins[1])/spans[1]*N), N-1)
        k = min(int((z-mins[2])/spans[2]*N), N-1)
        key = (i,j,k)
        voxmap.setdefault(key, []).append(int(lbl))

    # majority vote
    grid = np.zeros((N,N,N), dtype=int)
    for (i,j,k), labels in voxmap.items():
        # most common
        counts = np.bincount(labels)
        grid[i,j,k] = int(counts.argmax())

    # save flat
    with open(seg_out, "w") as f:
        for v in grid.flatten():
            f.write(f"{v}\n")
    print(f"-> wrote {N**3} seg labels to {seg_out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",        required=True,  help=".npz input")
    p.add_argument("--voxelizer",  required=True,  help="path to voxelize_bin.exe")
    p.add_argument("--out_ply",    required=True,  help="output .ply")
    p.add_argument("--out_cls",    required=True,  help="output cls.txt")
    p.add_argument("--out_seg",    required=True,  help="output seg.txt")
    args = p.parse_args()

    # 1) .npz → cloud.bin
    bin_tmp = "_tmp_pc.bin"
    make_bin(args.npz, bin_tmp)

    # 2) cloud.bin → .ply
    run_voxelizer(bin_tmp, args.voxelizer, args.out_ply)

    # удаляем временный .bin
    os.remove(bin_tmp)

    # 3) Сохраняем классификац. метку
    save_cls(args.npz, args.out_cls)

    # 4) Сохраняем сегментационные метки
    save_seg(args.npz, args.out_seg)

    print("DONE.")

if __name__ == "__main__":
    main()
