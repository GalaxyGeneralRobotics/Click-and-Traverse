import numpy as np
import torch
from dataclasses import dataclass
from scipy.ndimage import binary_closing, binary_opening, binary_erosion, rotate as ndi_rotate
import os
@dataclass
class Cfg:
    Lx: float = 3.0; Ly: float = 2.0; Lz: float = 1.5
    voxel: float = 0.04
    origin_w: tuple = (-0.5, -1.0, 0.0)
    start_w:  tuple = (0.0,  0.0, 0.75)
    goal_w:   tuple = (2.0,  0.0, 0.75)

    difficulty: float = 0.9
    seed: int = 42

    rect_factor: float = 1.0
    density_bias: float = 0.25

    block_dx_L: tuple = (0.3, 0.50)     # lateral left
    block_dz_L: tuple = (0.6, 0.80)

    block_dx_R: tuple = (0.3, 0.50)     # lateral right
    block_dz_R: tuple = (0.6, 0.80)

    block_dx_F: tuple = (0.10, 0.25)    # floor
    block_dy_F: tuple = (0.4, 1.60) 

    block_dx_C: tuple = (0.18, 0.60)    # ceiling
    block_dy_C: tuple = (0.4, 1.60)  

    n_rect_L: int = 9 # 0-9
    n_rect_R: int = 9 # 0-9
    n_rect_F: int = 3 # 0-3
    n_rect_C: int = 3 # 0-3

    y_center: float = 0.0
    gap_half_min: float = 0.16
    gap_half_max: float = 0.3
    gap_margin:   float = 0.05
    curve_knots_per_m: float = 0.9
    curve_smooth_vox: int = 9

    lr_thick_ratio_min: float = 0.3
    lr_thick_ratio_max: float = 1.0

    gate_segments_per_meter: float = 0.8
    gate_min_len: float = 0.30
    gate_max_len: float = 0.40
    gate_join_gap: float = 0.2
    gate_dir_jitter: float = 0.10

    floor_max_h: float = 0.25
    ceil_min_z:  float = 1.0

    rot_max_deg_L: float = 10.0
    rot_max_deg_R: float = 10.0
    rot_max_deg_F: float = 8.0
    rot_max_deg_C: float = 10.0

    closing_iters: int = 1
    closing_kernel: int = 3

    narrow_thresh: float = 0.45
    widen_extra:   float = 0.2
    top_band_z:    float = 0.30
    bot_band_z:    float = 0.20


def _lerp(a, b, t): 
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def make_axes(cfg: Cfg):
    Nx = int(round(cfg.Lx / cfg.voxel))
    Ny = int(round(cfg.Ly / cfg.voxel))
    Nz = int(round(cfg.Lz / cfg.voxel))
    ox, oy, oz = cfg.origin_w
    xv = ox + (np.arange(Nx) + 0.5) * cfg.voxel
    yv = oy + (np.arange(Ny) + 0.5) * cfg.voxel
    zv = oz + (np.arange(Nz) + 0.5) * cfg.voxel
    return xv, yv, zv

def _smooth1d(a, k):
    k = max(3, int(k) | 1)
    w = np.ones(k, np.float32) / k
    p = k // 2
    ap = np.pad(a, (p, p), mode='edge')
    return np.convolve(ap, w, mode='valid')

def _value_noise_1d(n, knots, rng):
    xs = np.linspace(0, n-1, max(2, knots)).astype(np.float32)
    vs = rng.uniform(0.0, 1.0, size=xs.shape[0]).astype(np.float32)
    x  = np.arange(n, dtype=np.float32)
    return np.interp(x, xs, vs)

def _perlin1d_0_1(n, knots, smooth_vox, rng):
    v = _value_noise_1d(n, knots, rng)
    v = _smooth1d(v, smooth_vox)
    v = (v - v.min()) / (np.ptp(v) + 1e-6)
    return v

def rect_mask_xz(Nx, Nz, x_min, x_max, n_rect, min_wx, max_wx, min_wz, max_wz, rng):
    M = np.zeros((Nx, Nz), np.uint8)
    for _ in range(n_rect):
        wx = int(rng.integers(min_wx, max_wx+1))
        wz = int(rng.integers(min_wz, max_wz+1))
        wx = max(wx, 1); wz = max(wz, 1)
        if wx > Nx: wx = Nx
        if wz > Nz: wz = Nz
        x0 = int(rng.integers(x_min, x_max - wx + 1))
        z0 = int(rng.integers(0, Nz - wz + 1))
        M[x0:x0+wx, z0:z0+wz] = 1
    return M

def rect_mask_xy(Nx, Ny, x_min, x_max, n_rect, min_wx, max_wx, min_wy, max_wy, difficulty, rng):
    M = np.zeros((Nx, Ny), np.uint8)
    for _ in range(n_rect):
        wx = int(rng.integers(min_wx, max_wx+1))
        wy = int(_lerp(min_wy, max_wy, difficulty)) 
        wx = max(wx, 1); wy = max(wy, 1)
        if wx > Nx: wx = Nx
        if wy > Ny: wy = Ny
        x0 = int(rng.integers(x_min, x_max - wx + 1))
        y0 = int(rng.integers(0, Ny - wy + 1))
        M[x0:x0+wx, y0:y0+wy] = 1
    return M

def closing_opening_padded(occ, iters=3, kernel=3):
    assert kernel in (3,5,7)
    rad = (kernel//2) * iters
    pad = max(1, rad)
    se = np.ones((kernel, kernel, kernel), dtype=bool)

    occ_pad = np.pad(
        occ.astype(bool),
        ((pad,pad),(pad,pad),(pad,pad)),
        mode='constant', constant_values=True
    )

    occ_closed = binary_closing(occ_pad, structure=se, iterations=iters)

    occ_pad[:pad] = True
    occ_pad[-pad:] = True
    occ_pad[:,:pad] = True
    occ_pad[:,-pad:] = True
    occ_pad[:,:,:pad] = True
    occ_pad[:,:,-pad:] = True
    occ_closed_opened = binary_opening(occ_closed, structure=se, iterations=1)

    occ_crop = occ_closed_opened[pad:-pad, pad:-pad, pad:-pad]
    return occ_crop.astype(bool)

def get_elevation(obs_mask: np.ndarray):
    nonzero_mask = obs_mask == 0

    ground_idx = np.argmax(nonzero_mask, axis=2)
    ground_idx[~np.any(nonzero_mask, axis=2)] = -1 

    ceil_idx = obs_mask.shape[2] - 1 - np.argmax(nonzero_mask[..., ::-1], axis=2)
    ceil_idx[~np.any(nonzero_mask, axis=2)] = -1
    return ground_idx, ceil_idx

def extract_surface_voxels(occ: np.ndarray, structure=None) -> np.ndarray:
    structure = np.ones((3,3,3), dtype=bool)
    eroded = binary_erosion(occ, structure=structure, border_value=1)
    eroded[0,:,:] = occ[0,:,:]
    eroded[-1,:,:] = occ[-1,:,:]
    eroded[:,0,:] = occ[:,0,:]
    eroded[:,-1,:] = occ[:,-1,:]
    eroded[:,:,0] = occ[:,:,0]
    eroded[:,:,-1] = occ[:,:,-1]
    surface = occ & (~eroded)

    return surface

def make_y_center_curve(cfg: Cfg, xv, rng, anchor_len=0.30, margin=0.05):
    Nx = len(xv)
    base = _perlin1d_0_1(Nx, int(cfg.curve_knots_per_m * cfg.Lx)+2, cfg.curve_smooth_vox, rng)
    base = 2.0*base - 1.0
    max_dev = (cfg.Ly/2 - cfg.gap_half_max - margin)
    y_curve = cfg.y_center + max_dev * base * 0.5
    s_x, g_x = cfg.start_w[0], cfg.goal_w[0]
    w_s = np.clip(1.0 - np.abs(xv - s_x)/anchor_len, 0.0, 1.0)
    w_s = 0.5 - 0.5*np.cos(np.pi*w_s)
    w_g = np.clip(1.0 - np.abs(xv - g_x)/anchor_len, 0.0, 1.0)
    w_g = 0.5 - 0.5*np.cos(np.pi*w_g)
    y_curve = y_curve*(1-w_s) + cfg.y_center*w_s
    y_curve = y_curve*(1-w_g) + cfg.y_center*w_g
    return y_curve.astype(np.float32)

def make_gap_half_curve_jumpy(cfg: Cfg, xv, rng, min_seg=0.18, max_seg=0.60, smooth=2):
    Nx = len(xv)
    min_seg_v = max(1, int(np.ceil(min_seg / cfg.voxel)))
    max_seg_v = max(min_seg_v+1, int(np.ceil(max_seg / cfg.voxel)))
    g = np.empty(Nx, np.float32); i = 0
    while i < Nx:
        L = int(rng.integers(min_seg_v, max_seg_v+1))
        if rng.random() < 0.13 * (1 + cfg.difficulty):
            target = cfg.gap_half_min + 0.04 * rng.random()
        else:
            target = cfg.gap_half_min + (cfg.gap_half_max - cfg.gap_half_min) * rng.random()
        g[i:min(Nx, i+L)] = target; i += L
    if smooth >= 3:
        g = _smooth1d(g, smooth)
    return np.clip(g, cfg.gap_half_min, min(cfg.gap_half_max, cfg.Ly/2 - cfg.gap_margin)).astype(np.float32)

def passband_indices_from_curve(yv, y_curve, gap_half_curve):
    y_left  = y_curve - gap_half_curve
    y_right = y_curve + gap_half_curve
    j_left  = np.searchsorted(yv, y_left,  side='right')
    j_right = np.searchsorted(yv, y_right, side='left')
    return j_left.astype(np.int32), j_right.astype(np.int32)

def build_occ_from_masks_thick_xyxz(cfg: Cfg,
    L_mask_xz, R_mask_xz,    # (Nx, Nz)
    F_mask_xy, C_mask_xy,    # (Nx, Ny)
    rng, j_left_cut, j_right_cut
):
    xv, yv, zv = make_axes(cfg)
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    vox = cfg.voxel

    tL_max = j_left_cut.copy()
    tR_max = (Ny - j_right_cut).copy()
    # lr_min, lr_max = cfg.lr_thick_ratio_min, cfg.lr_thick_ratio_max

    # L_th = np.zeros((Nx, Nz), np.int32)
    # if L_mask_xz.any():
    #     ratios_L = _lerp(lr_min, lr_max, cfg.difficulty) # TODO
    #     L_th = np.minimum((ratios_L * tL_max[:,None]).astype(np.int32), tL_max[:,None])
    #     L_th[L_mask_xz==0] = 0

    # R_th = np.zeros((Nx, Nz), np.int32)
    # if R_mask_xz.any():
    #     ratios_R = _lerp(lr_min, lr_max, cfg.difficulty) # TODO
    #     R_th = np.minimum((ratios_R * tR_max[:,None]).astype(np.int32), tR_max[:,None])
    #     R_th[R_mask_xz==0] = 0
    ratio_det = _lerp(cfg.lr_thick_ratio_min, cfg.lr_thick_ratio_max, cfg.difficulty)  # ✅

    L_th = (ratio_det * tL_max[:,None]).astype(np.int32)
    # L_th[L_mask_xz==0] = 0
    L_th = np.where(L_mask_xz==1, np.minimum(L_th, tL_max[:,None]), 0).astype(np.int32)

    R_th = (ratio_det * tR_max[:,None]).astype(np.int32)
    # R_th[R_mask_xz==0] = 0
    R_th = np.where(R_mask_xz==1, np.minimum(R_th, tR_max[:,None]), 0).astype(np.int32)
    # breakpoint()
    # 地/顶厚度（沿 z）
    floor_min_vox = max(2, int(np.ceil(0.08 / vox)))
    floor_max_vox = max(floor_min_vox, int(np.floor(cfg.floor_max_h / vox)))
    f_min = _lerp(floor_min_vox, floor_max_vox, cfg.difficulty * 0.6)
    f_max = _lerp(floor_min_vox, floor_max_vox, cfg.difficulty)
    f_max = min(f_max, floor_max_vox)

    k_min = int(np.searchsorted(zv, cfg.ceil_min_z, side='right'))
    ceil_allowed = max(0, Nz - k_min)
    ceil_min_vox = max(2, int(np.ceil(0.16 / vox)))
    ceil_max_vox = max(ceil_min_vox, int(np.floor(min(0.5, ceil_allowed*vox)/vox)))
    c_min = _lerp(ceil_min_vox, ceil_max_vox, cfg.difficulty * 0.6)
    c_max = _lerp(ceil_min_vox, ceil_max_vox, cfg.difficulty)
    c_max = min(c_max, ceil_max_vox)

    floor_v = np.zeros((Nx, Ny), np.int32)
    if F_mask_xy.any() and f_max > 0:
        sel = np.where(F_mask_xy==1)
        floor_v[sel] = np.random.default_rng(cfg.seed+321).integers(f_min, f_max+1, size=sel[0].size) # TODO

    ceil_v  = np.zeros((Nx, Ny), np.int32)
    if C_mask_xy.any() and c_max > 0:
        sel = np.where(C_mask_xy==1)
        ceil_v[sel]  = np.random.default_rng(cfg.seed+654).integers(c_min,  c_max+1, size=sel[0].size) # TODO
    J = np.arange(Ny, dtype=np.int32)[None, :, None]
    K = np.arange(Nz, dtype=np.int32)[None, None, :]

    occ_left  = (L_mask_xz[:,None,:]==1) & (J <  L_th[:,None,:])
    occ_right = (R_mask_xz[:,None,:]==1) & ((Ny - 1 - J) < R_th[:,None,:])

    occ_floor = (K <  floor_v[:,:,None])
    k_start   = np.maximum(k_min, Nz - np.clip(ceil_v, 0, Nz))
    occ_ceil  = (K >= k_start[:,:,None])

    return occ_left | occ_right | occ_floor | occ_ceil

def rotate_mask_2d(mask: np.ndarray, max_deg: float, rng) -> np.ndarray:
    if max_deg <= 0: return mask
    ang = float(rng.uniform(-max_deg, max_deg))
    R = ndi_rotate(mask.astype(np.uint8), angle=ang, reshape=False,
                   order=0, mode='constant', cval=0.0, prefilter=False)
    return (R > 0).astype(np.uint8)

def sample_gate_segments(
    Nx, p_x, rng,
    min_len_vox, max_len_vox,
    nseg, gap_min_vox,
    jitter=0.0
):
    # gap_min_vox = max(1, int(np.ceil(gap_min_m / cfg.voxel)))
    keep = np.zeros(Nx, np.uint8)

    w = np.asarray(p_x, np.float32)
    if jitter > 1e-6:
        w = np.clip(w * (1.0 + jitter * rng.standard_normal(Nx).astype(np.float32)), 1e-6, None)
    w /= w.sum()

    placed = []
    attempts = 0
    while len(placed) < max(1, nseg) and attempts < nseg * 20:
        attempts += 1
        i0 = rng.choice(Nx, p=w)
        L  = int(rng.integers(min_len_vox, max_len_vox+1))
        i1 = min(Nx, i0 + L)

        # ok = True
        # for (s, e) in placed:
        #     if not (i1 + gap_min_vox <= s or i0 >= e + gap_min_vox):
        #         ok = False
        #         break
        # if not ok:
        #     continue

        placed.append((i0, i1))
        keep[i0:i1] = 1

    placed.sort()
    return keep


def generate_and_save(cfg: Cfg, prefix="occ", save=True):
    rng = np.random.default_rng(cfg.seed)
    xv, yv, zv = make_axes(cfg)
    Nx, Ny, Nz = len(xv), len(yv), len(zv)

    y_curve = make_y_center_curve(cfg, xv, np.random.default_rng(cfg.seed+1))
    gap_half_curve = make_gap_half_curve_jumpy(cfg, xv, np.random.default_rng(cfg.seed+2))

    dx = float(np.mean(np.diff(xv)))  # 一般等距，≈ cfg.voxel
    ypp = np.zeros_like(y_curve, dtype=np.float32)
    ypp[1:-1] = (y_curve[2:] - 2*y_curve[1:-1] + y_curve[:-2]) / (dx*dx)
    bend = np.abs(ypp)

    den = np.percentile(bend, 90) + 1e-6
    bend01 = np.clip(bend / den, 0.0, 1.0)
    # gap_half_curve = gap_half_curve * (1.0 + 30.8 * bend01 * bend01)
    
    j_left_cut, j_right_cut = passband_indices_from_curve(yv, y_curve, gap_half_curve)

    def vox_range(r_m):
        lo = max(2, int(np.ceil(r_m[0] / cfg.voxel)))
        hi = max(lo, int(np.ceil(r_m[1] / cfg.voxel)))
        return lo, hi

    min_wx_L, max_wx_L = vox_range(cfg.block_dx_L);  min_wz_L, max_wz_L = vox_range(cfg.block_dz_L)
    min_wx_R, max_wx_R = vox_range(cfg.block_dx_R);  min_wz_R, max_wz_R = vox_range(cfg.block_dz_R)

    min_wx_F, max_wx_F = vox_range(cfg.block_dx_F);  min_wy_F, max_wy_F = vox_range(cfg.block_dy_F)
    min_wx_C, max_wx_C = vox_range(cfg.block_dx_C);  min_wy_C, max_wy_C = vox_range(cfg.block_dy_C)

    def auto_nrect(min_wx, scale=1.0):
        base = max(1, int(cfg.rect_factor * (Nx / min_wx)))
        return max(1, int(np.round((cfg.density_bias + cfg.difficulty) * base * scale)))

    n_rect_L = cfg.n_rect_L if cfg.n_rect_L >= 0 else auto_nrect(min_wx_L, scale=1.0)
    n_rect_R = cfg.n_rect_R if cfg.n_rect_R >= 0 else auto_nrect(min_wx_R, scale=1.0)
    n_rect_F = cfg.n_rect_F if cfg.n_rect_F >= 0 else auto_nrect(min_wx_F, scale=0.9)
    n_rect_C = cfg.n_rect_C if cfg.n_rect_C >= 0 else auto_nrect(min_wx_C, scale=0.9)

    rngL, rngR, rngF, rngC = [np.random.default_rng(cfg.seed + s) for s in (11, 22, 33, 44)]

    x_origin = cfg.origin_w[0]
    x_start = cfg.start_w[0] + 0.2
    x_goal  = cfg.goal_w[0] - 0.2
    x_min, x_max = min(x_start, x_goal), max(x_start, x_goal)
    Nx_min = int(np.ceil((x_min - x_origin) / cfg.voxel))
    Nx_max = int(np.ceil((x_max - x_origin) / cfg.voxel))

    L_mask_xz = rect_mask_xz(Nx, Nz, Nx_min, Nx_max, n_rect_L, min_wx_L, max_wx_L, min_wz_L, max_wz_L, rngL)
    R_mask_xz = rect_mask_xz(Nx, Nz, Nx_min, Nx_max, n_rect_R, min_wx_R, max_wx_R, min_wz_R, max_wz_R, rngR)

    F_mask_xy = rect_mask_xy(Nx, Ny, Nx_min, Nx_max, n_rect_F, min_wx_F, max_wx_F, min_wy_F, max_wy_F, cfg.difficulty, rngF)
    C_mask_xy = rect_mask_xy(Nx, Ny, Nx_min, Nx_max, n_rect_C, min_wx_C, max_wx_C, min_wy_C, max_wy_C, cfg.difficulty, rngC)

    occ_S1 = build_occ_from_masks_thick_xyxz(cfg, L_mask_xz, R_mask_xz, F_mask_xy, C_mask_xy,
                                             rng, j_left_cut, j_right_cut)
    if save:
        torch.save(torch.from_numpy(occ_S1.astype(np.uint8)), f"{prefix}_01_blocks.pt")

    p_x = np.zeros(Nx, np.float32)
    mask = (xv >= x_min) & (xv <= x_max)
    p_x[mask] = 0.5

    nseg = max(1, int(round(cfg.gate_segments_per_meter * cfg.Lx * (0.5 + 0.5*cfg.difficulty))))
    min_len_vox = max(2, int(np.ceil(cfg.gate_min_len / cfg.voxel)))
    max_len_vox = max(min_len_vox, int(np.ceil(cfg.gate_max_len / cfg.voxel)))
    join_gap_vox = max(0, int(np.round(cfg.gate_join_gap / cfg.voxel)))

    keep_LR = sample_gate_segments(Nx, p_x, rngL, min_len_vox, max_len_vox, nseg, join_gap_vox, cfg.gate_dir_jitter)
    keep_C  = sample_gate_segments(Nx, p_x, rngC, min_len_vox, max_len_vox, nseg, join_gap_vox, cfg.gate_dir_jitter)

    L_mask2_xz = L_mask_xz #* keep_LR[:, None]
    R_mask2_xz = R_mask_xz #* keep_LR[:, None]
    F_mask2_xy = F_mask_xy 
    C_mask2_xy = C_mask_xy #* keep_C[:, None]

    occ_S2 = build_occ_from_masks_thick_xyxz(cfg, L_mask2_xz, R_mask2_xz, F_mask2_xy, C_mask2_xy,
                                             rng, j_left_cut, j_right_cut)
    if save:
        torch.save(torch.from_numpy(occ_S2.astype(np.uint8)), f"{prefix}_02_gate.pt")

    L_mask3_xz = rotate_mask_2d(L_mask2_xz, cfg.rot_max_deg_L, np.random.default_rng(cfg.seed+201))
    R_mask3_xz = rotate_mask_2d(R_mask2_xz, cfg.rot_max_deg_R, np.random.default_rng(cfg.seed+202))
    F_mask3_xy = rotate_mask_2d(F_mask2_xy, cfg.rot_max_deg_F, np.random.default_rng(cfg.seed+203))
    C_mask3_xy = rotate_mask_2d(C_mask2_xy, cfg.rot_max_deg_C, np.random.default_rng(cfg.seed+204))

    occ_S3 = build_occ_from_masks_thick_xyxz(cfg, L_mask3_xz, R_mask3_xz, F_mask3_xy, C_mask3_xy,
                                             rng, j_left_cut, j_right_cut)
    r = 0.4  
    r2 = r * r

    # dist2_start[j,i] = (x_i - x_s)^2 + (y_j - y_s)^2
    dx_s = xv[:, None] - cfg.start_w[0]   # (Nx,1)
    dy_s = yv[None, :] - cfg.start_w[1]   # (1,Ny)
    dist2_start = dx_s**2 + dy_s**2       # (Nx,Ny)

    dx_g = xv[:, None] - cfg.goal_w[0]
    dy_g = yv[None, :] - cfg.goal_w[1]
    dist2_goal = dx_g**2 + dy_g**2        # (Nx,Ny)

    mask_start_xy = (dist2_start <= r2)   # (Nx,Ny)
    mask_goal_xy  = (dist2_goal  <= r2)   # (Nx,Ny)

    occ_S3[mask_start_xy, :] = False      #  occ_S2[mask_start_xy[:,:,None]] = False
    occ_S3[mask_goal_xy,  :] = False
    if save:
        torch.save(torch.from_numpy(occ_S3.astype(np.uint8)), f"{prefix}_03_rot.pt")

    # occ_S4 = closing_padded(occ_S3, iters=cfg.closing_iters, kernel=cfg.closing_kernel)
    occ_S4 = closing_opening_padded(occ_S3, iters=cfg.closing_iters, kernel=cfg.closing_kernel)

    if save:
        torch.save(torch.from_numpy(occ_S4.astype(np.uint8)), f"{prefix}_04_closed.pt")

    occ_S5 = occ_S4.copy()


    narrow_vox = int(np.ceil(cfg.narrow_thresh / cfg.voxel))
    widen_vox  = int(np.ceil(cfg.widen_extra   / cfg.voxel))
    k_top = int(np.floor((cfg.Lz - cfg.top_band_z - cfg.origin_w[2]) / cfg.voxel))
    k_bot = int(np.ceil((cfg.bot_band_z - cfg.origin_w[2]) / cfg.voxel))
    k_top = np.clip(k_top, 0, Nz-1); k_bot = np.clip(k_bot, 0, Nz-1)
    j_center = np.searchsorted(yv, y_curve, side='left').astype(np.int32)

    for i in range(Nx):
        width = j_right_cut[i] - j_left_cut[i]
        if width >= narrow_vox:
            continue
        sl = occ_S5[i]  # (Ny, Nz)
        mid_band = sl[j_left_cut[i]:j_right_cut[i], :]

        top_has = mid_band[:, k_top:].any()
        bot_has = mid_band[:, :k_bot+1].any()
        if not (top_has or bot_has):
            continue

        # jl = max(0, j_center[i] - (width//2 + widen_vox))
        # jr = min(Ny, j_center[i] + (width//2 + widen_vox))
        alpha = 1.8 - cfg.difficulty
        widen_vox_local = int(np.ceil(widen_vox * np.exp(alpha * bend01[i] * bend01[i])))

        jl = max(0, j_center[i] - (width//2 + widen_vox_local))
        jr = min(Ny, j_center[i] + (width//2 + widen_vox_local))


        z_low = 0
        z_high = Nz - 1

        if top_has:
            idxs = np.argwhere(mid_band[:, k_top:])
            if idxs.size > 0:
                z_high = k_top + idxs[:,1].min()

        if bot_has:
            idxs = np.argwhere(mid_band[:, :k_bot+1])
            if idxs.size > 0:
                z_low = idxs[:,1].max()

        if z_low <= z_high:
            sl[jl:jr, z_low:z_high+1] = False
            occ_S5[i] = sl

    if save:
        torch.save(torch.from_numpy(occ_S5.astype(np.uint8)), f"{prefix}_05_final.pt")
    return occ_S5, xv, yv, zv



if __name__ == "__main__":
    difficulty = 0.2
    seed = 42
    n_rect_L = 9 # 0-9
    n_rect_R = 9 # 0-9
    n_rect_F = 3 # 0-3
    n_rect_C = 3 # 0-3
    prefix = f"D{difficulty:.1f}_S{seed}_L{n_rect_L:01d}_R{n_rect_R:01d}_F{n_rect_F:01d}_C{n_rect_C:01d}/"
    os.makedirs(prefix, exist_ok=True)
    cfg = Cfg(difficulty=difficulty, seed=seed, n_rect_L=n_rect_L, n_rect_R=n_rect_R, n_rect_F=n_rect_F, n_rect_C=n_rect_C)
    generate_and_save(cfg, prefix=prefix)
