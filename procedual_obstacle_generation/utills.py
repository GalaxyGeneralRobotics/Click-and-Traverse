import argparse, os, sys
import numpy as np
import torch
from skimage import measure
import trimesh
from trimesh.creation import cylinder, cone
from trimesh.visual.material import SimpleMaterial
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing

def load_occupancy_pt(pt_path, key=None):
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        if key is None:
            for k in ["occupancy", "occ", "vox", "voxel", "grid"]:
                if k in obj:
                    key = k
                    break
        if key is None:
            raise ValueError("输入是 dict，但未找到常见的 occupancy 字段，请用 --key 指定。")
        tensor = obj[key]
    elif torch.is_tensor(obj):
        tensor = obj
    else:
        raise TypeError("pt 文件既不是 tensor 也不是包含 tensor 的 dict。")

    occ = tensor.detach().cpu().numpy()
    if occ.dtype != np.bool_:
        occ = occ.astype(np.uint8) > 0
    return occ  # (a,b,c) -> (X,Y,Z)

def load_occupancy_npy(npy_path, key=None):
    """
    加载 occupancy 格网数据 (.npy 或 .npz 格式)

    Args:
        npy_path: 文件路径
        key: 若文件是 .npz，可指定键名（如 "occupancy"）

    Returns:
        occ: np.ndarray, dtype=bool, shape=(X, Y, Z)
    """
    # === Step 1. 读取文件 ===
    obj = np.load(npy_path, allow_pickle=True)

    # === Step 2. 判断类型 ===
    if isinstance(obj, np.lib.npyio.NpzFile):  # 多键压缩 npz
        if key is None:
            # 尝试常见字段名
            for k in ["occupancy", "occ", "vox", "voxel", "grid"]:
                if k in obj.files:
                    key = k
                    break
        if key is None:
            raise ValueError("输入是 .npz 文件，但未找到常见的 occupancy 字段，请用 key 参数指定。")
        arr = obj[key]
    elif isinstance(obj, np.ndarray):  # 直接是数组
        arr = obj
    else:
        raise TypeError("文件既不是 ndarray，也不是包含 ndarray 的 .npz。")

    # === Step 3. 转为 bool 格式 ===
    occ = np.array(arr)
    if occ.dtype != np.bool_:
        occ = occ.astype(np.uint8) > 0

    return occ  # (a,b,c) -> (X,Y,Z)


def marching_cubes_mesh(occ, spacing):
    if occ.sum() == 0:
        raise ValueError("occupancy 全为 0，无法抽取表面。")
    verts, faces, normals, _ = measure.marching_cubes(occ.astype(np.uint8), level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh

def make_arrow(axis_len, radius, tip_ratio=0.15, segments=24, axis='x', color=(1.0,0.0,0.0)):
    """
    生成从原点指向 +axis 的箭头（圆柱 + 圆锥），并返回合并后的 mesh。
    """
    shaft_len = axis_len * (1.0 - tip_ratio)
    tip_len   = axis_len * tip_ratio
    shaft = cylinder(radius=radius, height=shaft_len, sections=segments)
    tip   = cone(radius=radius*1.8, height=tip_len, sections=segments)

    # 缺省 cylinder/cone 的轴向是 +Z，高度从 -h/2 到 +h/2
    # 先把它们平移到 0..length，再旋转到目标轴
    shaft.apply_translation([0, 0, shaft_len/2.0])
    tip.apply_translation([0, 0, shaft_len + tip_len/2.0])

    # 旋转到 X 或 Y 方向
    if axis == 'x':
        R = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0,1,0])  # z->x
    elif axis == 'y':
        R = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1,0,0])   # z->y
    elif axis == 'z':
        R = np.eye(4)
    else:
        raise ValueError("axis 必须是 x/y/z")

    shaft.apply_transform(R)
    tip.apply_transform(R)

    # 设定材质颜色（写到 MTL）
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
    idx = np.argwhere(occ)  # N×3, 每行为(i,j,k)
    if idx.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    xyz = (idx + 0.5) * np.array(voxel_size, dtype=np.float32)
    xyz += np.array(origin, dtype=np.float32)
    return xyz.astype(np.float32)

def preview_matplotlib(points, sample=200000):
    # 轻量预览：随机采样避免卡顿
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

    # 固定坐标轴范围
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
    # 坐标范围
    x_min, x_max = 0.0, 3.0
    y_min, y_max = 2.0, 2.2
    z_min, z_max = -0.1, 2.0

    # 8 个顶点（矩形墙体）
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

    # 12 个三角面（每个面两个三角形）
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 5, 6], [4, 6, 7],  # 顶面
        [0, 1, 5], [0, 5, 4],  # 前面
        [1, 2, 6], [1, 6, 5],  # 右面
        [2, 3, 7], [2, 7, 6],  # 后面
        [3, 0, 4], [3, 4, 7],  # 左面
    ])

    # 法线（可由 trimesh 自动算）
    normals = None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh

def combine_meshes(mesh_list):
    """
    将多个 trimesh.Trimesh 合并成一个整体 mesh
    """
    # 确保都是 Trimesh 实例
    valid_meshes = [m for m in mesh_list if isinstance(m, trimesh.Trimesh)]
    if len(valid_meshes) == 0:
        raise ValueError("❌ 没有可合并的 mesh 对象。")

    # 合并
    combined = trimesh.util.concatenate(valid_meshes)
    return combined

def extract_rotated_subvolume_np(
    occ,                         # numpy array (X, Y, Z)
    voxel_size=0.04,             # 输入体素大小 (m)
    center=(0.0, 0.0, 0.0),      # 子盒中心 (cx,cy,cz) in meters
    dir_xy=(1.0, 0.0),           # 盒子“长度”在XY平面内的方向
    box_size=(3.0, 2.0, 1.5),    # (L, W, H) in meters
    origin=(0.0, 0.0, 0.0),      # 源体素(0,0,0)的世界坐标 (m)
    out_voxel_size=None,         # 输出体素大小，默认等于输入
    threshold=None               # 若给定阈值(如0.5)，则返回布尔占用
):
    """
    返回 sub_occ，形状 (nL, nW, nH)，对应盒子局部坐标 (L, W, H) 三轴。
    假定 occ[i,j,k] 对应世界坐标 (x=i*vs + ox, y=j*vs + oy, z=k*vs + oz)。
    三线性插值，越界按0填充（零填充）。
    """
    if out_voxel_size is None:
        out_voxel_size = voxel_size

    occ_f = occ.astype(np.float32)
    X, Y, Z = occ_f.shape

    # ---- 盒子输出体素数 ----
    L, W, H = box_size
    nL = max(1, int(round(L / out_voxel_size)))
    nW = max(1, int(round(W / out_voxel_size)))
    nH = max(1, int(round(H / out_voxel_size)))

    # 在每个输出体素中心采样（避免半格偏移问题）
    l_vals = np.linspace(-L/2 + out_voxel_size/2, L/2 - out_voxel_size/2, nL, dtype=np.float32)
    w_vals = np.linspace(-W/2 + out_voxel_size/2, W/2 - out_voxel_size/2, nW, dtype=np.float32)
    h_vals = np.linspace(-H/2 + out_voxel_size/2, H/2 - out_voxel_size/2, nH, dtype=np.float32)

    # 局部坐标网格 (nL, nW, nH)
    Lg, Wg, Hg = np.meshgrid(l_vals, w_vals, h_vals, indexing='ij')

    # ---- 构造局部正交基：长度方向 vL，宽度方向 vW，高度方向 vH ----
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    vL = np.array([dx, dy, 0.0], dtype=np.float32)
    vLn = vL / (np.linalg.norm(vL) + 1e-9)        # 单位向量（XY平面）

    vW = np.array([-vLn[1], vLn[0], 0.0], dtype=np.float32)  # 与vL正交（XY内）
    vH = np.array([0.0, 0.0, 1.0], dtype=np.float32)         # 高度对齐Z轴

    C = np.array(center, dtype=np.float32)  # 世界坐标中心
    O = np.array(origin, dtype=np.float32)  # 源体素原点

    # ---- 局部 -> 世界 ----
    wx = C[0] + Lg * vLn[0] + Wg * vW[0] + Hg * vH[0]
    wy = C[1] + Lg * vLn[1] + Wg * vW[1] + Hg * vH[1]
    wz = C[2] + Lg * vLn[2] + Wg * vW[2] + Hg * vH[2]

    # ---- 世界 -> 源体素连续索引 (ix,iy,iz) ----
    ix = (wx - O[0]) / voxel_size
    iy = (wy - O[1]) / voxel_size
    iz = (wz - O[2]) / voxel_size

    # ---- 三线性插值（零填充）----
    # 下取整/上取整索引
    ix0 = np.floor(ix).astype(np.int64); ix1 = ix0 + 1
    iy0 = np.floor(iy).astype(np.int64); iy1 = iy0 + 1
    iz0 = np.floor(iz).astype(np.int64); iz1 = iz0 + 1

    # 插值权重
    fx = ix - ix0; wx0 = 1.0 - fx; wx1 = fx
    fy = iy - iy0; wy0 = 1.0 - fy; wy1 = fy
    fz = iz - iz0; wz0 = 1.0 - fz; wz1 = fz

    # 每个角的有效性（是否在界内）
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

    # 为索引安全进行裁剪（用于读取），随后用 mask 把越界角置0
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

    # 权重乘积
    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    sub = (v000*w000 + v100*w100 + v010*w010 + v110*w110 +
           v001*w001 + v101*w101 + v011*w011 + v111*w111).astype(np.float32)  # 形状 (nL,nW,nH)

    if threshold is not None:
        sub = (sub >= float(threshold))
        # 若希望返回与原 occ 相同 dtype：
        if occ.dtype == np.bool_:
            sub = sub.astype(np.bool_)
        else:
            sub = sub.astype(occ.dtype)

    return sub  # (nL, nW, nH)

def fill_upward_from_threshold(sub, voxel_size=0.04, center_z=0.0, z_threshold=1.2 ,ground_th=0.2):
    """
    sub: (nL, nW, nH) 的占用体素（bool/0-1/float都可）
    规则：若某 (i,j) 列在 z > z_threshold 处存在占用，则该列该高度及其之上全部置占用。
    返回同形状 sub_filled（float32，若需 bool 可再转）。
    """
    sub = sub.astype(np.float32)
    nL, nW, nH = sub.shape
    H = nH * voxel_size

    # 子块高度轴在局部坐标的采样中心（与之前裁剪一致：采样在体素中心）
    h_vals = np.linspace(0, H, nH, dtype=np.float32)
    z_world = center_z + h_vals  # 每个高度层对应的世界 z

    # 仅在 z >= 阈值 区间内做“向上填充”
    mask_above = z_world >= z_threshold            # (nH,)
    if not mask_above.any():
        return sub  # 没有层高超过阈值，直接返回

    occ_above = sub[:, :, mask_above]             # (nL, nW, nH_above)
    # 沿高度方向做前缀最大：一旦出现1，后面全为1（实现“往上全填”）
    filled_above = np.maximum.accumulate(occ_above, axis=2)

    occ = sub.copy()
    occ[:, :, mask_above] = filled_above

    # mask_floor = z_world < 0.06           # (nH,)
    # sub_filled[:, :, mask_floor] = 1.
    below_mask = z_world < ground_th
    occ[:, :, below_mask] = 0

    # 2. 找到 [ground_th, ground_th+0.05) 范围的层索引
    range_mask = (z_world >= ground_th) & (z_world < ground_th + 0.05)
    if not range_mask.any():
        return occ

    # 哪些 (x,y) 列在这个范围内有占用
    occ_in_range = occ[:, :, range_mask]             # (X,Y,K)
    col_mask = occ_in_range.any(axis=2)              # (X,Y) 布尔

    # 3. 把这些列在 ground_th 以下全部置 1
    occ[col_mask, :][:, below_mask] = 1
    return occ

import numpy as np

def expand_occupancy_lr(occ: np.ndarray, y_center: int = 0) -> np.ndarray:
    """
    从 y = y_center 出发，沿 y 轴左右扩张 occupancy：
    - 若从中心向左或向右方向上遇到 occupancy，则该侧后续所有 voxel 都变为 occupancy。
    （类似“光线”遇到障碍后的填充效果）

    Args:
        occ (np.ndarray): 3D occupancy, shape (X, Y, Z), dtype=bool 或 uint8
        y_center (int): 起始 y 索引（默认 0）

    Returns:
        np.ndarray: 扩张后的 occupancy (dtype=bool)
    """
    occ = occ.astype(bool)
    X, Y, Z = occ.shape
    expanded = occ.copy()

    # --- 向右方向 (y_center → +y)
    right_region = occ[:, y_center:, :]
    filled_right = np.cumsum(right_region[:, ::1, :], axis=1) > 0
    expanded[:, y_center:, :] |= filled_right

    # --- 向左方向 (y_center → -y)
    left_region = occ[:, :y_center+1, :][:, ::-1, :]  # 反向
    filled_left = np.cumsum(left_region, axis=1) > 0
    filled_left = filled_left[:, ::-1, :]  # 再翻回来
    expanded[:, :y_center+1, :] |= filled_left

    return expanded

