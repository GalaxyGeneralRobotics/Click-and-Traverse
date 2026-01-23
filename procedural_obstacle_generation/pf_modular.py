# pf_modular.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import skfmm
# @dataclass
class PFConfig:
    voxel: float = 0.04           # resolution (m)
    Lx: float = 3.0               # x-axis (m) forward
    Ly: float = 2.0               # y-axis (m) left
    Lz: float = 1.5               # z-axis (m) up
    origin_w: np.ndarray = np.array([-0.5, -1.0, 0.0], dtype=np.float32)
    start_w:  np.ndarray = np.array([0.0,  0.0, 0.75], dtype=np.float32)
    goal_w:   np.ndarray = np.array([2.0,  0.0, 0.75], dtype=np.float32)

    v_max: float = 0.6          # max velocity far from goal (m/s)
    k_decay: float = 0.6        # decay radius near goal (m)
    goal_seed_r: float = 0.12   # goal negative region radius (m)

def world_to_local(p_w, cfg: PFConfig):
    return p_w - cfg.origin_w

def make_sdf(obs_mask: np.ndarray, voxel: float) -> np.ndarray:
    phi_obs = np.ones(obs_mask.shape, dtype=float)
    phi_obs[obs_mask] = -1.0
    sdf = skfmm.distance(phi_obs, dx=voxel).astype(np.float32)  # signed distance (m)
    return sdf

def grad3(scalar_field: np.ndarray, voxel: float):
    dfx, dfy, dfz = np.gradient(scalar_field, voxel, voxel, voxel, edge_order=2)
    return np.stack([dfx, dfy, dfz], axis=-1).astype(np.float32)


def make_raw_guidance_field(cfg, grids, obs_mask, goal_local, r_proj=None): # HumanoidPF
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # can be adjusted according to robot size

    X, Y, Z = grids
    # goal negative region (small sphere)
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T (geodesic distance)
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    return T

def make_guidance_field_progressive(cfg, grids, obs_mask, goal_local, bf, sdf, r_proj=None):
    """
    cfg      : PFConfig voxel, v_max, k_decay
    obs_mask : bool ndarray, True=inside obstacle
    bf       : (...,3) normal field (recommended to use SDF outward normal)
    sdf      : np.ndarray, signed distance to obstacles (positive outside, negative inside). If None, constructed from obs_mask
    r_proj   : radius of influence for normal projection (m). If None, defaults to 2*voxel ~ 3*voxel
    returns:
      T, gf  : HumanoidPF and gradient field
    """
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # can be adjusted according to robot size

    X, Y, Z = grids
    # goal negative region (small sphere)
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T (geodesic distance)
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    # 2) -∇T
    dTx, dTy, dTz = np.gradient(T, voxel, voxel, voxel, edge_order=2)
    g = np.stack([-dTx, -dTy, -dTz], axis=-1).astype(np.float32)     # (...,3)

    # 3) b̂ norm
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    bunit = np.zeros_like(bf, dtype=np.float32)
    valid_b = bnorm[..., 0] > eps
    bunit[valid_b] = bf[valid_b] / bnorm[valid_b]

    # 4) g_perp = g - (g·b̂) b̂
    proj = np.sum(g * bunit, axis=-1, keepdims=True)
    # proj = np.clip(proj, -1.0, 0.0)
    g_perp = g - proj * bunit

    # 5) distance weight w(d): d>=0 outside distance; w=1 (close to edge) -> w=0 (far away)
    d_out = np.maximum(sdf, 0.0)
    t = np.clip(d_out / (r_proj + eps), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    w = (1.0 - smooth)[..., None]  # (...,1)

    # 6) combine: more "tangential" near edges, original navigation farther away
    g_mix = (1.0 - w) * g + w * g_perp
    # g_mix[g_mix[...,0]<0] *= -1

    # 7) inside obstacles: use normal (usually outward normal to push field away; use -bunit to point inward)
    # from scipy.ndimage import binary_dilation
    # obs_mask = binary_dilation(obs_mask, iterations=1)
    g_mix[obs_mask] = bunit[obs_mask]

    # 8) normalize (final step) + speed scalar
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    dir_unit = np.zeros_like(g_mix, dtype=np.float32)
    nz = (mag[..., 0] > eps)
    dir_unit[nz] = g_mix[nz] / mag[nz]

    T_thresh = 0.3
    p = 3.0 
    goal_dist = (((X - goal_local[0])**2 + (Y - goal_local[1])**2))**0.5

    speed = np.where(
        goal_dist > T_thresh,
        cfg.v_max,
        cfg.v_max * (goal_dist / T_thresh) ** p
    )
    speed = speed.astype(np.float32)[..., None]


    gf = (dir_unit * speed).astype(np.float32)
    return T, gf

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

def visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, start_l, goal_l, title_prefix=""):
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

    kx = int(np.argmin(np.abs(xv - 1.))) 
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


