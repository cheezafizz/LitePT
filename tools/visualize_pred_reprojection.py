import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

def instance_palette(n_classes, seed=0):
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(n_classes, 3), dtype=np.uint8)
    return colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="bemealbakeryyedang_007098")
    parser.add_argument("--pred-npz", default="viz/maskclust_strong/strong-filter/bemealbakeryyedang_007098_pred.npz")
    parser.add_argument("--source-root", default="/home/fai/workspace/jhp/dataset/val_sample_output")
    parser.add_argument("--out-dir", default="viz/reprojected")
    parser.add_argument("--mask-occlusion-tol", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading prediction from {args.pred_npz}...")
    pred = np.load(args.pred_npz, allow_pickle=False)
    coord = pred["coord"].astype(np.float64) # (N, 3)
    inst_id = pred["inst_id"].astype(np.int32) # (N,)
    
    max_inst_id = inst_id.max() if inst_id.size > 0 else 0
    palette = instance_palette(max_inst_id + 1)
    
    loc, _, frame = args.scene.partition("_")
    data_npz = os.path.join(args.source_root, loc, frame, "data.npz")
    print(f"Loading data from {data_npz}...")
    data = np.load(data_npz, allow_pickle=True)
    
    rgb_all = data["rgb"] # (S, H, W, 3)
    if rgb_all.dtype != np.uint8:
        rgb_all = (np.clip(rgb_all, 0.0, 1.0) * 255).astype(np.uint8)
    
    K_all = data["intrinsic_aligned"].astype(np.float64)
    ext_all = data["extrinsic_pnp"].astype(np.float64)
    
    world_points = None
    for k in ["world_points", "world_points_from_depth", "points", "pts3d", "xyz"]:
        if k in data.files:
            world_points = data[k]
            break
    if world_points is None:
        raise ValueError("world_points not found in data.npz")
    
    world_points = world_points.astype(np.float64)
    S, H, W, _ = rgb_all.shape
    tol = args.mask_occlusion_tol
    
    for s in range(S):
        K = K_all[s]
        R = ext_all[s][:, :3]
        t = ext_all[s][:, 3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        surf_z = (world_points[s] @ R[2, :]) + t[2]
        
        Pc = coord @ R.T + t
        z = Pc[:, 2]
        front = z > 1e-6
        zc = np.where(front, z, 1.0)
        u = fx * Pc[:, 0] / zc + cx
        v = fy * Pc[:, 1] / zc + cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        
        in_b = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not in_b.any():
            continue
            
        idx = np.nonzero(in_b)[0]
        su = surf_z[vi[idx], ui[idx]]
        
        occluded = np.isfinite(su) & (z[idx] > su + tol)
        vis = idx[~occluded]
        
        if vis.size == 0:
            continue
            
        # Sort by depth (descending) so closer points are drawn last
        sort_idx = np.argsort(z[vis])[::-1]
        vis = vis[sort_idx]
        vis_valid = vis[inst_id[vis] >= 0]
        
        # Create an image to draw on
        img = Image.fromarray(rgb_all[s])
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Draw points
        for pt_idx in vis_valid:
            x, y = ui[pt_idx], vi[pt_idx]
            color = tuple(palette[inst_id[pt_idx]]) + (150,) # Add alpha
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            
        out_path = os.path.join(args.out_dir, f"{args.scene}_view{s}.png")
        img.convert("RGB").save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
