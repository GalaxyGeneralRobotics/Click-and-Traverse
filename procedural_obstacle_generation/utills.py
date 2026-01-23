import numpy as np
import torch
from skimage import measure
import trimesh
from trimesh.creation import cylinder, cone
from trimesh.visual.material import SimpleMaterial

def load_occupancy_pt(pt_path, key=None):
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        if key is None:
            for k in ["occupancy", "occ", "vox", "voxel", "grid"]:
                if k in obj:
                    key = k
                    break
        if key is None:
            raise ValueError("")
        tensor = obj[key]
    elif torch.is_tensor(obj):
        tensor = obj
    else:
        raise TypeError("")

    occ = tensor.detach().cpu().numpy()
    if occ.dtype != np.bool_:
        occ = occ.astype(np.uint8) > 0
    return occ  # (a,b,c) -> (X,Y,Z)

def load_occupancy_npy(npy_path, key=None):
    obj = np.load(npy_path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):  
        if key is None:
            for k in ["occupancy", "occ", "vox", "voxel", "grid"]:
                if k in obj.files:
                    key = k
                    break
        if key is None:
            raise ValueError("")
        arr = obj[key]
    elif isinstance(obj, np.ndarray): 
        arr = obj
    else:
        raise TypeError("")

    occ = np.array(arr)
    if occ.dtype != np.bool_:
        occ = occ.astype(np.uint8) > 0

    return occ  # (a,b,c) -> (X,Y,Z)


def marching_cubes_mesh(occ, spacing):
    if occ.sum() == 0:
        raise ValueError("occupancy is empty.")
    verts, faces, normals, _ = measure.marching_cubes(occ.astype(np.uint8), level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh

def make_arrow(axis_len, radius, tip_ratio=0.15, segments=24, axis='x', color=(1.0,0.0,0.0)):
    shaft_len = axis_len * (1.0 - tip_ratio)
    tip_len   = axis_len * tip_ratio
    shaft = cylinder(radius=radius, height=shaft_len, sections=segments)
    tip   = cone(radius=radius*1.8, height=tip_len, sections=segments)

    shaft.apply_translation([0, 0, shaft_len/2.0])
    tip.apply_translation([0, 0, shaft_len + tip_len/2.0])

    if axis == 'x':
        R = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0,1,0])  # z->x
    elif axis == 'y':
        R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1,0,0])   # z->y
    elif axis == 'z':
        R = np.eye(4)
    else:
        raise ValueError("axis  x/y/z")

    shaft.apply_transform(R)
    tip.apply_transform(R)

    mat = SimpleMaterial(name=f"axis_{axis}", diffuse=color)
    shaft.visual.material = mat
    tip.visual.material   = mat

    return trimesh.util.concatenate([shaft, tip])

def build_axes(lengths, radius, segments=24):
    Lx, Ly, Lz = lengths
    x_arrow = make_arrow(Lx, radius, segments=segments, axis='x', color=(1,0,0))   # 红
    y_arrow = make_arrow(Ly, radius, segments=segments, axis='y', color=(0,1,0))   # 绿
    z_arrow = make_arrow(Lz, radius, segments=segments, axis='z', color=(0,0,1))   # 蓝
    return trimesh.util.concatenate([x_arrow, y_arrow, z_arrow])

def occupancy_to_points(occ, voxel_size=0.05, origin=(0,0,0)):
    idx = np.argwhere(occ)  # N×3, (i,j,k)
    if idx.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    xyz = (idx + 0.5) * np.array(voxel_size, dtype=np.float32)
    xyz += np.array(origin, dtype=np.float32)
    return xyz.astype(np.float32)

def preview_matplotlib(points, sample=200000):
    if points.shape[0] == 0:
        print("[warn] 没有占用点可视化"); return
    if points.shape[0] > sample:
        sel = np.random.choice(points.shape[0], sample, replace=False)
        pts = points[sel]
    else:
        pts = points

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"PointCloud preview  (N={points.shape[0]})")
    plt.tight_layout()
    plt.show()

def make_wall():
    x_min, x_max = 0.0, 3.0
    y_min, y_max = 2.0, 2.2
    z_min, z_max = -0.1, 2.0

    verts = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3], 
        [4, 5, 6], [4, 6, 7], 
        [0, 1, 5], [0, 5, 4],  
        [1, 2, 6], [1, 6, 5],  
        [2, 3, 7], [2, 7, 6], 
        [3, 0, 4], [3, 4, 7], 
    ])

    normals = None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh

def combine_meshes(mesh_list):
    valid_meshes = [m for m in mesh_list if isinstance(m, trimesh.Trimesh)]
    if len(valid_meshes) == 0:
        raise ValueError("❌")

    combined = trimesh.util.concatenate(valid_meshes)
    return combined

def extract_rotated_subvolume_np(
    occ,                       
    voxel_size=0.04,           
    center=(0.0, 0.0, 0.0),    
    dir_xy=(1.0, 0.0),         
    box_size=(3.0, 2.0, 1.5),  
    origin=(0.0, 0.0, 0.0),    
    out_voxel_size=None,       
    threshold=None             
):
    if out_voxel_size is None:
        out_voxel_size = voxel_size

    occ_f = occ.astype(np.float32)
    X, Y, Z = occ_f.shape

    L, W, H = box_size
    nL = max(1, int(round(L / out_voxel_size)))
    nW = max(1, int(round(W / out_voxel_size)))
    nH = max(1, int(round(H / out_voxel_size)))

    l_vals = np.linspace(-L/2 + out_voxel_size/2, L/2 - out_voxel_size/2, nL, dtype=np.float32)
    w_vals = np.linspace(-W/2 + out_voxel_size/2, W/2 - out_voxel_size/2, nW, dtype=np.float32)
    h_vals = np.linspace(-H/2 + out_voxel_size/2, H/2 - out_voxel_size/2, nH, dtype=np.float32)

    Lg, Wg, Hg = np.meshgrid(l_vals, w_vals, h_vals, indexing='ij')

    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    vL = np.array([dx, dy, 0.0], dtype=np.float32)
    vLn = vL / (np.linalg.norm(vL) + 1e-9)      

    vW = np.array([-vLn[1], vLn[0], 0.0], dtype=np.float32)  
    vH = np.array([0.0, 0.0, 1.0], dtype=np.float32)       

    C = np.array(center, dtype=np.float32)  
    O = np.array(origin, dtype=np.float32)  

    wx = C[0] + Lg * vLn[0] + Wg * vW[0] + Hg * vH[0]
    wy = C[1] + Lg * vLn[1] + Wg * vW[1] + Hg * vH[1]
    wz = C[2] + Lg * vLn[2] + Wg * vW[2] + Hg * vH[2]

    ix = (wx - O[0]) / voxel_size
    iy = (wy - O[1]) / voxel_size
    iz = (wz - O[2]) / voxel_size

    ix0 = np.floor(ix).astype(np.int64); ix1 = ix0 + 1
    iy0 = np.floor(iy).astype(np.int64); iy1 = iy0 + 1
    iz0 = np.floor(iz).astype(np.int64); iz1 = iz0 + 1

    fx = ix - ix0; wx0 = 1.0 - fx; wx1 = fx
    fy = iy - iy0; wy0 = 1.0 - fy; wy1 = fy
    fz = iz - iz0; wz0 = 1.0 - fz; wz1 = fz

    def inb(i, lo, hi):  # lo<=i<hi
        return (i >= lo) & (i < hi)

    m000 = inb(ix0,0,X) & inb(iy0,0,Y) & inb(iz0,0,Z)
    m100 = inb(ix1,0,X) & inb(iy0,0,Y) & inb(iz0,0,Z)
    m010 = inb(ix0,0,X) & inb(iy1,0,Y) & inb(iz0,0,Z)
    m110 = inb(ix1,0,X) & inb(iy1,0,Y) & inb(iz0,0,Z)
    m001 = inb(ix0,0,X) & inb(iy0,0,Y) & inb(iz1,0,Z)
    m101 = inb(ix1,0,X) & inb(iy0,0,Y) & inb(iz1,0,Z)
    m011 = inb(ix0,0,X) & inb(iy1,0,Y) & inb(iz1,0,Z)
    m111 = inb(ix1,0,X) & inb(iy1,0,Y) & inb(iz1,0,Z)

    ix0c = np.clip(ix0, 0, X-1); ix1c = np.clip(ix1, 0, X-1)
    iy0c = np.clip(iy0, 0, Y-1); iy1c = np.clip(iy1, 0, Y-1)
    iz0c = np.clip(iz0, 0, Z-1); iz1c = np.clip(iz1, 0, Z-1)

    v000 = occ_f[ix0c, iy0c, iz0c] * m000
    v100 = occ_f[ix1c, iy0c, iz0c] * m100
    v010 = occ_f[ix0c, iy1c, iz0c] * m010
    v110 = occ_f[ix1c, iy1c, iz0c] * m110
    v001 = occ_f[ix0c, iy0c, iz1c] * m001
    v101 = occ_f[ix1c, iy0c, iz1c] * m101
    v011 = occ_f[ix0c, iy1c, iz1c] * m011
    v111 = occ_f[ix1c, iy1c, iz1c] * m111

    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    sub = (v000*w000 + v100*w100 + v010*w010 + v110*w110 +
           v001*w001 + v101*w101 + v011*w011 + v111*w111).astype(np.float32)  #  (nL,nW,nH)

    if threshold is not None:
        sub = (sub >= float(threshold))
        if occ.dtype == np.bool_:
            sub = sub.astype(np.bool_)
        else:
            sub = sub.astype(occ.dtype)

    return sub  # (nL, nW, nH)

def fill_upward_from_threshold(sub, voxel_size=0.04, center_z=0.0, z_threshold=1.2 ,ground_th=0.2):
    sub = sub.astype(np.float32)
    nL, nW, nH = sub.shape
    H = nH * voxel_size

    h_vals = np.linspace(0, H, nH, dtype=np.float32)
    z_world = center_z + h_vals 

    mask_above = z_world >= z_threshold            # (nH,)
    if not mask_above.any():
        return sub 

    occ_above = sub[:, :, mask_above]             # (nL, nW, nH_above)
    filled_above = np.maximum.accumulate(occ_above, axis=2)

    occ = sub.copy()
    occ[:, :, mask_above] = filled_above

    # mask_floor = z_world < 0.06           # (nH,)
    # sub_filled[:, :, mask_floor] = 1.
    below_mask = z_world < ground_th
    occ[:, :, below_mask] = 0

    range_mask = (z_world >= ground_th) & (z_world < ground_th + 0.05)
    if not range_mask.any():
        return occ

    occ_in_range = occ[:, :, range_mask]             # (X,Y,K)
    col_mask = occ_in_range.any(axis=2)              # (X,Y) 

    occ[col_mask, :][:, below_mask] = 1
    return occ

import numpy as np

def expand_occupancy_lr(occ: np.ndarray, y_center: int = 0) -> np.ndarray:
    occ = occ.astype(bool)
    X, Y, Z = occ.shape
    expanded = occ.copy()

    right_region = occ[:, y_center:, :]
    filled_right = np.cumsum(right_region[:, ::1, :], axis=1) > 0
    expanded[:, y_center:, :] |= filled_right

    left_region = occ[:, :y_center+1, :][:, ::-1, :]  
    filled_left = np.cumsum(left_region, axis=1) > 0
    filled_left = filled_left[:, ::-1, :]
    expanded[:, :y_center+1, :] |= filled_left

    return expanded

