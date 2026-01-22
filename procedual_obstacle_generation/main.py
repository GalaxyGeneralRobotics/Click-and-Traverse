from random_obstacle import make_axes, generate_and_save, extract_surface_voxels, Cfg as ObsCfg, get_elevation
from obs import build_obstacles
import os
import shutil
from pf_modular import make_sdf, make_guidance_field_progressive, grad3, PFConfig, visualize_all
import numpy as np
from utills import marching_cubes_mesh, occupancy_to_points, preview_matplotlib, combine_meshes
def generate_random_obstacle(difficulty, seed, n_rect_L, n_rect_R, n_rect_F, n_rect_C):
    prefix = f"D{int(difficulty*10)}_{n_rect_R:01d}{n_rect_F:01d}{n_rect_C:01d}_S{seed}/"

    print(prefix)
    save = False
    if save:
        os.makedirs(prefix, exist_ok=True)
    obs_cfg = ObsCfg(difficulty=difficulty, seed=seed, n_rect_L=n_rect_L, n_rect_R=n_rect_R, n_rect_F=n_rect_F, n_rect_C=n_rect_C)
    obs_mask, xv, yv, zv = generate_and_save(obs_cfg, prefix=prefix, save=save)
    if obs_mask.any() == False:
        shutil.rmtree(prefix)
        return

    os.makedirs(f"RandObs/{prefix}", exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (obs_cfg.voxel, obs_cfg.voxel, obs_cfg.voxel)
    mesh = marching_cubes_mesh(obs_mask, spacing=spacing)
    mesh.export(f"RandObs/{prefix}obs.obj")

    cfg = PFConfig()
    cfg.voxel = obs_cfg.voxel
    cfg.start_w = obs_cfg.start_w
    cfg.goal_w = obs_cfg.goal_w
    cfg.origin_w = obs_cfg.origin_w
    cfg.Lx = obs_cfg.Lx
    cfg.Ly = obs_cfg.Ly
    cfg.Lz = obs_cfg.Lz
    
    # SDF 与导数
    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)

    # Eikonal 导航场
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    # 保存
    np.save(f"../data/assets/RandObs/{prefix}sdf.npy", sdf)
    np.save(f"../data/assets/RandObs/{prefix}bf.npy",  bf)
    np.save(f"../data/assets/RandObs/{prefix}gf.npy",  gf)
    np.save(f"../data/assets/RandObs/{prefix}obs.npy", obs_mask.astype(np.uint8))
    sur = extract_surface_voxels(obs_mask)
    np.save(f"../data/assets/RandObs/{prefix}sur.npy", sur.astype(np.uint8))
    # pts = occupancy_to_points(sur, voxel_size=obstacle_cfg.voxel)
    # preview_matplotlib(pts)

    # 可视化
    os.makedirs(f"fig/", exist_ok=True)
    visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w, title_prefix=f'fig/{prefix[:-1]}')

def generate_typical_obstacle(scene_type):
    prefix = f"{scene_type}/"
    obs_cfg = ObsCfg()
    cfg = PFConfig()
    assert cfg.voxel == obs_cfg.voxel
    assert (cfg.start_w == obs_cfg.start_w).all()
    assert (cfg.goal_w == obs_cfg.goal_w).all()
    assert (cfg.origin_w == obs_cfg.origin_w).all()
    assert cfg.Lx == obs_cfg.Lx
    assert cfg.Ly == obs_cfg.Ly
    assert cfg.Lz == obs_cfg.Lz
    
    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    obs_mask = build_obstacles(scene_type, (X, Y, Z))

    os.makedirs(f"../data/assets/TypiObs/{prefix}", exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (cfg.voxel, cfg.voxel, cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(f"../data/assets/TypiObs/{prefix}obs.obj")

    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    np.save(f"../data/assets/TypiObs/{prefix}sdf.npy", sdf)
    np.save(f"../data/assets/TypiObs/{prefix}bf.npy",  bf)
    np.save(f"../data/assets/TypiObs/{prefix}gf.npy",  gf)
    np.save(f"../data/assets/TypiObs/{prefix}obs.npy", obs_mask.astype(np.uint8))
    # sur = extract_surface_voxels(obs_mask)
    # np.save(f"../data/assets/TypiObs/{prefix}sur.npy", sur.astype(np.uint8))
    # ground_idx, ceil_idx = get_elevation(obs_mask)
    # np.save(f"../data/assets/TypiObs/{prefix}ground.npy", ground_idx)
    # np.save(f"../data/assets/TypiObs/{prefix}ceil.npy", ceil_idx)
    # pts = occupancy_to_points(sur, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)

    # visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w)


def better_mesh(spacing, obs_mask):
    obs_mask[:,0,:] = 0
    obs_mask[:,-1,:] = 0
    obs_mask[:,:,-1] = 0
    obs_mask_erosion = 1-obs_mask
    mesh = marching_cubes_mesh(obs_mask_erosion, spacing=spacing)
    return mesh

if __name__ == "__main__":
    # difficulties = [0.6, 0.7, 0.8]
    # nLs = [1,3]
    # nRs = [1,3]
    # nFs = [0, 1, 2]
    # nCs = [0, 1, 2]
    # seeds = [1,2]
    # combos = itertools.product(difficulties, nLs, nRs, nFs, nCs, seeds)
    # for difficulty, nL, nR, nF, nC, seed in combos:
    #     if nL != nR:
    #         continue
    #     # if nL == 0 and nR == 0 and nF == 0 and nC == 0:
    #     #     continue
    #     rng = (nL + 0.5 * seed) * (nF + seed) * (nC + 1.5 * seed) + 1
    #     # seed = int((nL + nR + nF + nC) * difficulty)
    #     generate_test(difficulty, int(rng), nL, nR, nF, nC)
    # generate_typical_obstacle('ceil0')
    generate_typical_obstacle('narrow1')