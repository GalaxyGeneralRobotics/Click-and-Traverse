# pf_modular.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import skfmm
# =========================
# 配置
# =========================
# @dataclass
class PFConfig:
    voxel: float = 0.04           # 分辨率(米)
    Lx: float = 3.0               # x 尺度(米) 前
    Ly: float = 2.0               # y 尺度(米) 左
    Lz: float = 1.5               # z 尺度(米) 上
    origin_w: np.ndarray = np.array([-0.5, -1.0, 0.0], dtype=np.float32)
    start_w:  np.ndarray = np.array([0.0,  0.0, 0.75], dtype=np.float32)
    goal_w:   np.ndarray = np.array([2.0,  0.0, 0.75], dtype=np.float32)

    # 导航速度
    v_max: float = 0.6          # 远处最大速度 m/s
    k_decay: float = 0.6        # 近 goal 衰减半径(米)
    goal_seed_r: float = 0.12   # goal 负区半径(米)

# =========================
# 网格工具
# =========================
# def make_axes(L, voxel):
#     n = int(np.round(L / voxel))
#     return np.linspace(0.0, L, n + 1, dtype=np.float32)

# def make_grid(cfg: PFConfig):
#     xv = make_axes(cfg.Lx, cfg.voxel)
#     yv = make_axes(cfg.Ly, cfg.voxel)
#     zv = make_axes(cfg.Lz, cfg.voxel)
#     X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
#     return (xv, yv, zv), (X, Y, Z)

def world_to_local(p_w, cfg: PFConfig):
    return p_w - cfg.origin_w

# =========================
# 场生成：SDF / ∇SDF / 导航场 GF
# =========================
def make_sdf(obs_mask: np.ndarray, voxel: float) -> np.ndarray:
    phi_obs = np.ones(obs_mask.shape, dtype=float)
    phi_obs[obs_mask] = -1.0
    sdf = skfmm.distance(phi_obs, dx=voxel).astype(np.float32)  # 有符号距离(米)
    return sdf

def grad3(scalar_field: np.ndarray, voxel: float):
    dfx, dfy, dfz = np.gradient(scalar_field, voxel, voxel, voxel, edge_order=2)
    return np.stack([dfx, dfy, dfz], axis=-1).astype(np.float32)


def make_raw_guidance_field(cfg, grids, obs_mask, goal_local, r_proj=None):
    """
    cfg 需要: voxel, v_max, k_decay
    T        : np.ndarray, 距离势
    obs_mask : bool ndarray, True=障碍物内部
    bf       : (...,3) 法向场（建议为SDF外法向）
    sdf      : np.ndarray, 与障碍物的有符号距离(外正内负). 若None则用obs_mask构造
    r_proj   : 法向消去的影响半径（米）。若None默认 2*voxel ~ 3*voxel
    返回:
      T, gf  : 指导速度场（已归一化并做速度衰减）
    """
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # 可按机器人尺寸调大/调小

    X, Y, Z = grids
    # 目标负区（小球）
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T（到φ=0的最短距离，障碍绕行）
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    return T

def make_guidance_field_progressive(cfg, grids, obs_mask, goal_local, bf, sdf, r_proj=None):
    """
    cfg 需要: voxel, v_max, k_decay
    T        : np.ndarray, 距离势
    obs_mask : bool ndarray, True=障碍物内部
    bf       : (...,3) 法向场（建议为SDF外法向）
    sdf      : np.ndarray, 与障碍物的有符号距离(外正内负). 若None则用obs_mask构造
    r_proj   : 法向消去的影响半径（米）。若None默认 2*voxel ~ 3*voxel
    返回:
      T, gf  : 指导速度场（已归一化并做速度衰减）
    """
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # 可按机器人尺寸调大/调小

    X, Y, Z = grids
    # 目标负区（小球）
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T（到φ=0的最短距离，障碍绕行）
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    # 2) -∇T（不做归一化、不做速度缩放）
    dTx, dTy, dTz = np.gradient(T, voxel, voxel, voxel, edge_order=2)
    g = np.stack([-dTx, -dTy, -dTz], axis=-1).astype(np.float32)     # (...,3)

    # 3) 单位化法向 b̂；零范数处跳过
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    bunit = np.zeros_like(bf, dtype=np.float32)
    valid_b = bnorm[..., 0] > eps
    bunit[valid_b] = bf[valid_b] / bnorm[valid_b]

    # 4) 去法向分量: g_perp = g - (g·b̂) b̂
    proj = np.sum(g * bunit, axis=-1, keepdims=True)
    # proj = np.clip(proj, -1.0, 0.0)
    g_perp = g - proj * bunit

    # 5) 距离权重 w(d): d>=0 取外侧距离；w=1(贴边) -> w=0(远离)
    d_out = np.maximum(sdf, 0.0)
    # 使用smoothstep: s = clamp((d/r)^2 * (3-2*d/r), 0,1)
    t = np.clip(d_out / (r_proj + eps), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    # w = 2 * (1.0 - t)[..., None]  # (...,1) 0-2 0.5-1 1-0
    w = (1.0 - smooth)[..., None]  # (...,1)
    # w = 0

    # 6) 组合: 贴边更“切向”，远处还原原始导航
    g_mix = (1.0 - w) * g + w * g_perp
    # g_mix[g_mix[...,0]<0] *= -1

    # 7) 障碍物内部：用法向（通常用外法向把场推离障碍；若想指向内部可取 -bunit）
    # from scipy.ndimage import binary_dilation
    # obs_mask = binary_dilation(obs_mask, iterations=1)
    g_mix[obs_mask] = bunit[obs_mask]

    # 8) 统一归一化（最后一步）+ 速度标量
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    dir_unit = np.zeros_like(g_mix, dtype=np.float32)
    nz = (mag[..., 0] > eps)
    dir_unit[nz] = g_mix[nz] / mag[nz]

    # 速度：远处 vmax，近 goal 衰减到 0
    Tpos = np.maximum(T, 0.0)
    # speed = cfg.v_max * (Tpos / (Tpos + cfg.k_decay))
    # speed = np.clip(speed, 0.0, cfg.v_max).astype(np.float32)[..., None]
    T_thresh = 0.3
    p = 3.0  # >1 越大下降越陡
    goal_dist = (((X - goal_local[0])**2 + (Y - goal_local[1])**2))**0.5

    speed = np.where(
        goal_dist > T_thresh,
        cfg.v_max,
        cfg.v_max * (goal_dist / T_thresh) ** p
    )
    speed = speed.astype(np.float32)[..., None]


    gf = (dir_unit * speed).astype(np.float32)
    return T, gf

# =========================
# 保存
# =========================
def save_all(cfg: PFConfig, sdf, bf, gf, obs_mask, meta_extra=None):
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "sdf.npy", sdf)
    np.save(outdir / "bf.npy",  bf)
    np.save(outdir / "gf.npy",  gf)
    np.save(outdir / "obs.npy", obs_mask.astype(np.uint8))
    meta = {
        "voxel": cfg.voxel,
        "origin": cfg.origin_w,
        "shape_xyz": np.array(sdf.shape, dtype=np.int32),
        "start_w": cfg.start_w,
        "goal_w": cfg.goal_w,
        "scene": cfg.scene
    }
    if meta_extra:
        meta.update(meta_extra)
    np.save(outdir / "meta.npy", meta)
    print(f"[OK] Saved to {outdir}")

# =========================
# 可视化（三视图 + 导航矢量）
# =========================
def visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, start_l, goal_l, title_prefix=""):
    # z≈start_z 的顶视图 (xy)
    kz = int(np.argmin(np.abs(zv - start_l[2])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, :, kz].T, origin='lower',
                    extent=[xv[0], xv[-1], yv[0], yv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xy = obs_mask[:, :, kz].T
    plt.contour(obs_xy, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], yv[0], yv[-1]])
    step = 3
    X2, Y2 = np.meshgrid(xv[::step], yv[::step], indexing='ij')
    U = gf[::step, ::step, kz, 0]; V = gf[::step, ::step, kz, 1]
    plt.quiver(X2, Y2, U, V, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[0]],[start_l[1]], c='w', s=50, edgecolors='k', label='start')
    plt.scatter([goal_l[0]],[goal_l[1]], c='r', s=60, edgecolors='k', marker='*', label='goal')
    plt.title(f"{title_prefix} Top view (z≈{zv[kz]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.legend(); plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_top.png", dpi=300)
    plt.close()

    # y≈start_y 的侧视图 (xz)
    ky = int(np.argmin(np.abs(yv - start_l[1])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, ky, :].T, origin='lower',
                    extent=[xv[0], xv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xz = obs_mask[:, ky, :].T
    plt.contour(obs_xz, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], zv[0], zv[-1]])
    X2, Z2 = np.meshgrid(xv[::step], zv[::step], indexing='ij')
    U = gf[::step, ky, ::step, 0]; W = gf[::step, ky, ::step, 2]
    plt.quiver(X2, Z2, U, W, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[0]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[0]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Side view (y≈{yv[ky]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("z (m)")
    plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_side.png", dpi=300)
    plt.close()

    # x≈中线 的正视图 (yz)
    kx = int(np.argmin(np.abs(xv - 1.)))  # 取中线（可按需改）
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[kx, :, :].T, origin='lower',
                    extent=[yv[0], yv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_yz = obs_mask[kx, :, :].T
    plt.contour(obs_yz, levels=[0.5], colors='k',
                extent=[yv[0], yv[-1], zv[0], zv[-1]])
    Y2, Z2 = np.meshgrid(yv[::step], zv[::step], indexing='ij')
    V = gf[kx, ::step, ::step, 1]; W = gf[kx, ::step, ::step, 2]
    plt.quiver(Y2, Z2, V, W, pivot='mid', scale=30, color='w')
    plt.scatter([start_l[1]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[1]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Front view (x≈{xv[kx]:.2f} m)")
    plt.xlabel("y (m)"); plt.ylabel("z (m)")
    plt.tight_layout(); #plt.show()
    plt.savefig(f"{title_prefix}_front.png", dpi=300)
    plt.close()

import os
import numpy as np
import trimesh
from trimesh.creation import box

# =========================
# 主流程
# =========================
def main(cfg: PFConfig):
    # 网格
    (xv, yv, zv), (X, Y, Z) = make_grid(cfg)
    start_l = world_to_local(cfg.start_w, cfg)
    goal_l  = world_to_local(cfg.goal_w,  cfg)

    # 障碍体素
    obs_mask = build_obstacles(cfg, (X, Y, Z))

    # SDF 与导数
    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)

    # Eikonal 导航场
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, goal_l, bf, sdf)

    # gf[obs_mask] = bf[obs_mask]
    # from scipy.ndimage import binary_dilation
    # halo = binary_dilation(obs_mask, iterations=1) & (~obs_mask)
    # gf[halo] = bf[halo]

    # 保存
    save_all(cfg, sdf, bf, gf, obs_mask)

    # 可视化
    visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, start_l, goal_l,
                  title_prefix=f"[{cfg.scene}]")
    # export_obstacles(X, Y, Z, obs_mask, out_dir=cfg.outdir, voxel_size=cfg.voxel)

if __name__ == "__main__":
    # 选择场景： "threshold" | "pillar" | "door" | "two-pillars" | "shin_bar" | "ceiling"
    scene = "sphere"
    cfg = PFConfig(scene=scene, outdir=f"./{scene}")
    main(cfg)

