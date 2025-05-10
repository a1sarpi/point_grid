#!/usr/bin/env python3
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import product

def load_npz(path):
    """Load ShapeNet .npz with arrays 'pc' and 'part_label'."""
    data = np.load(path)
    pc = data['pc']                   # (M,3)
    labels = data['part_label'].astype(int)  # (M,)
    return pc, labels

def load_color_map(path):
    """
    Load part_color_mapping.json.
    The file is a LIST of [r,g,b], indexed by label.
    Returns dict[label] -> (r,g,b).
    """
    arr = json.load(open(path))
    return {i:tuple(arr[i]) for i in range(len(arr))}

def load_ply_centers(path):
    """
    Read ASCII PLY, return array of centers (x,y,z) 
    for all vertices with color very close to pure red.
    """
    centers = []
    with open(path,'r') as f:
        # skip header
        for line in f:
            if line.strip() == 'end_header':
                break
        for line in f:
            toks = line.strip().split()
            if len(toks) < 6: continue
            x,y,z,r,g,b = map(float, toks[:6])
            # normalized to 0–1
            if r/255.0 > 0.9 and g/255.0 < 0.1 and b/255.0 < 0.1:
                centers.append((x,y,z))
    return np.array(centers)

def make_wireframe_edges(center, size):
    """
    Given center (x,y,z) of a voxel and size=(sx,sy,sz),
    return the 12 line segments as pairs of points.
    """
    cx, cy, cz = center
    sx, sy, sz = size
    # half‐sizes
    hs = np.array([sx/2, sy/2, sz/2])
    # 8 offsets
    offsets = np.array(list(product([-1,1], repeat=3)))
    corners = offsets * hs + center
    edges_idx = [
        (0,1),(1,3),(3,2),(2,0),
        (4,5),(5,7),(7,6),(6,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return [[corners[a], corners[b]] for a,b in edges_idx]

def visualize(npz_path, ply_path, cmap_json, N, pt_size, edge_lw, edge_alpha):
    # 1) load point cloud + labels
    pc, labels = load_npz(npz_path)
    cmap = load_color_map(cmap_json)
    colors = np.array([cmap.get(l,(0.5,0.5,0.5)) for l in labels])

    # 2) load voxel centers from PLY
    centers = load_ply_centers(ply_path)
    if centers.size == 0:
        print("Warning: no centers found in PLY. Check output.ply generation.")

    # 3) compute origin and voxel size from pc
    mn = pc.min(axis=0)
    mx = pc.max(axis=0)
    spans = mx - mn
    size = spans / N

    # 4) build wireframe segments
    lines = []
    for c in centers:
        lines += make_wireframe_edges(c, size)

    # 5) plot
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    # full pointcloud
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],
               c=colors, s=pt_size, depthshade=False, label='Points')
    # voxel wireframe
    lc = Line3DCollection(lines, colors='black',
                          linewidths=edge_lw, alpha=edge_alpha)
    ax.add_collection3d(lc)

    ax.set_xlim(mn[0], mn[0]+size[0]*N)
    ax.set_ylim(mn[1], mn[1]+size[1]*N)
    ax.set_zlim(mn[2], mn[2]+size[2]*N)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="ShapeNet .npz (with 'pc' and 'part_label')")
    p.add_argument("ply", help="output.ply from voxelize.exe")
    p.add_argument("--N", type=int, default=16, help="Voxel grid size N")
    p.add_argument("--ptsz", type=float, default=1.0, help="Point size")
    p.add_argument("--edge_lw", type=float, default=0.5, help="Voxel edge linewidth")
    p.add_argument("--edge_alpha", type=float, default=0.3, help="Voxel edge opacity")
    p.add_argument("--cmap", default="part_color_mapping.json", help="Color map JSON")
    args = p.parse_args()
    visualize(args.npz, args.ply, args.cmap,
              args.N, args.ptsz, args.edge_lw, args.edge_alpha)

