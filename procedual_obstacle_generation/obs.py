import numpy as np

def build_obstacles(scene, grids):
    X, Y, Z = grids
    if scene == "pillar":
        return obs_pillar(X, Y, Z)
    elif scene == "narrow0":
        return obs_narrow(X, Y, Z)
    elif scene == "narrow1":
        return obs_narrow(X, Y, Z, gap_width=0.25)
    elif scene == "bar0":
        return obs_easy_bar(X, Y, Z)
    elif scene == "bar1":
        return obs_shin_bar(X, Y, Z)
    elif scene == "bar2":
        return obs_hard_bar(X, Y, Z)
    elif scene == "bar3":
        return obs_hard_bar(X, Y, Z, r=0.15, z=0.1,)
    elif scene == "ceil0":
        return obs_ceiling(X, Y, Z)
    elif scene == "ceil1":
        return obs_ceiling(X, Y, Z, z_low=1.0)
    elif scene == "ceilbar0":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z)
    elif scene == "ceilbar1":
        return obs_ceiling(X, Y, Z, z_low=1.0) | obs_shin_bar(X, Y, Z)
    elif scene == "Mceilbar0":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mceilbar1":
        return obs_ceiling(X, Y, Z, z_low=1.05) | obs_shin_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mhole0":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.5)
    elif scene == "empty":
        return obs_empty(X, Y, Z)
    elif scene == "Mceil0":
        return obs_ceiling(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar0":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mceil1":
        return obs_ceiling(X, Y, Z, z_low=1.0) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar1":
        return obs_shin_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar2":
        return obs_hard_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Nbar0":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.5)
    elif scene == "Nbar1":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.3)
    elif scene == "doubar":
        return obs_double_knee_bars(X, Y, Z)
    elif scene == "chest":
        return obs_chest_1(X, Y, Z)
    elif scene == "lowcorner":
        return obs_low_corner(X, Y, Z)
    elif scene == "highcorner":
        return obs_high_corner(X, Y, Z)
    else:
        raise ValueError(f"Unknown scene: {scene}")

def _box(X, Y, Z, *, x0, x1, y0, y1, z0, z1):
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)

def _cylinder_along_y(X, Y, Z, *, x, z, r, y0, y1):
    # 轴向为 y 的圆柱（横杆/水管）
    return ((X - x)**2 + (Z - z)**2 <= r**2) & (Y >= y0) & (Y <= y1)

def obs_threshold(X, Y, Z, *, x_center=1.5, thickness=0.12, z_low=0.00, z_high=0.12):
    """低门槛：沿 y 贯通，留出上方通过空间"""
    return ((X >= x_center - thickness/2) & (X <= x_center + thickness/2) &
            (Z >= z_low) & (Z <= z_high))

def obs_pillar(X, Y, Z, *, x=1.0, y=0.0, r=0.20, z_low=0.0, z_high=1.5):
    """圆柱：绕行"""
    return (((X - x)**2 + (Y - y)**2 <= r**2) &
            (Z >= z_low) & (Z <= z_high))

def obs_door(X, Y, Z, *, x=1.5, width=1.2, gap=0.5, thickness=0.12, z_low=0.0, z_high=1.5):
    """墙+中间门洞：x 方向一堵薄墙，中间留 y 向门洞"""
    left = ((X >= x - thickness/2) & (X <= x + thickness/2) &
            (Y <= (width - gap)/2) & (Z >= z_low) & (Z <= z_high))
    right = ((X >= x - thickness/2) & (X <= x + thickness/2) &
             (Y >= (width + gap)/2) & (Z >= z_low) & (Z <= z_high))
    return left | right

def obs_two_pillars(X, Y, Z, *, x=1.3, sep=0.8, r=0.18, z_low=0.0, z_high=1.5):
    """两根柱子留中缝：可穿行，增加转向需求"""
    p1 = (((X - x)**2 + (Y - (-sep/2))**2 <= r**2) &
          (Z >= z_low) & (Z <= z_high))
    p2 = (((X - x)**2 + (Y - ( +sep/2))**2 <= r**2) &
          (Z >= z_low) & (Z <= z_high))
    return p1 | p2
# 1) 腿部抬跨：小腿高度横杆（贯穿 y），适合“抬脚跨越”
def obs_hard_bar(X, Y, Z, *, x_center=1.0, r=0.1, z=0.08, y_width=2.0):
    """横向圆柱杆：z≈小腿高度，需要抬脚跨越"""
    y0, y1 = -y_width/2, +y_width/2
    # x_center -= ORIGIN_W[0]
    # y0 -= ORIGIN_W[1]
    # y1 -= ORIGIN_W[1]
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

def obs_shin_bar(X, Y, Z, *, x_center=1.0, r=0.08, z=0.05, y_width=2.0):
    """横向圆柱杆：z≈小腿高度，需要抬脚跨越"""
    y0, y1 = -y_width/2, +y_width/2
    # x_center -= ORIGIN_W[0]
    # y0 -= ORIGIN_W[1]
    # y1 -= ORIGIN_W[1]
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

def obs_easy_bar(X, Y, Z, *, x_center=1.0, r=0.08, z=0.00, y_width=2.0):
    """横向圆柱杆：z≈小腿高度，需要抬脚跨越"""
    y0, y1 = -y_width/2, +y_width/2
    # x_center -= ORIGIN_W[0]
    # y0 -= ORIGIN_W[1]
    # y1 -= ORIGIN_W[1]
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

def obs_air_cylinder(X, Y, Z, *, x_center=1.0, r=0.08, z=0.6, y_width=2.0):
    """横向圆柱杆：z≈小腿高度，需要抬脚跨越"""
    y0, y1 = -y_width/4, +y_width/4
    # x_center -= ORIGIN_W[0]
    # y0 -= ORIGIN_W[1]
    # y1 -= ORIGIN_W[1]
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

# 2) 膝盖高度双横杆：连续两根，练“连续高跨步”
def obs_double_knee_bars(X, Y, Z, *, x1=0.75, x2=1.25, r=0.08, z=0.05, y_width=2.0):
    """两根间隔横杆：需要两步连续高抬腿跨越"""
    y0, y1 = -y_width/2, +y_width/2
    bar1 = _cylinder_along_y(X, Y, Z, x=x1, z=z, r=r, y0=y0, y1=y1)
    bar2 = _cylinder_along_y(X, Y, Z, x=x2, z=z, r=r, y0=y0, y1=y1)
    return bar1 | bar2

# 3) 低头通过：头顶横梁（贯穿 y），适合“弯腰/低头”
def obs_overhead_beam(X, Y, Z, *, x_center=1.3, r=0.04, z=1.05, y_width=2.0):
    """头顶横梁：z≈胸/颈部上方，需要微蹲或低头通过"""
    y0, y1 = -y_width/2, +y_width/2
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

# 4) 低矮天花板段：一小段“压顶”，训练“下蹲前进”
def obs_ceiling(X, Y, Z, *, x_center=1, length=0.2, thickness=1.0, z_low=1.15, y_width=2.0):
    """短段低天花：在 [x_center - L/2, x_center + L/2] 内压低顶棚"""
    x0, x1 = x_center - length/2, x_center + length/2
    y0, y1 = -y_width/2, +y_width/2
    z0, z1 = z_low, z_low + thickness
    # x0 -= ORIGIN_W[0]
    # x1 -= ORIGIN_W[0]
    # y0 -= ORIGIN_W[1]
    # y1 -= ORIGIN_W[1]
    return _box(X, Y, Z, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

# 5) 窄缝门：薄墙上只有“一人宽”竖缝，训练“侧身/扭身”直行过缝
def obs_narrow(X, Y, Z, x=1., wall_thickness=0.12, y = 0, gap_width=0.4, z_low=0.0, z_high=1.5):
    """薄墙 + 窄门缝（沿 y 只在 |Y|<=gap/2 为空）"""
    # x -= ORIGIN_W[0]
    # y -= ORIGIN_W[1]
    wall = (X >= x - wall_thickness/2) & (X <= x + wall_thickness/2) & (Z >= z_low) & (Z <= z_high)
    slit = (np.abs(Y-y) <= gap_width/2)
    return wall & (~slit)

def obs_passage(X, Y, Z, gap_width=0.9):
    """薄墙 + 窄门缝（沿 y 只在 |Y|<=gap/2 为空）"""
    return ((X > -0.0) & (X < 2.0)) & ((Y > gap_width/2) | (Y < -gap_width/2))

def obs_empty(X, Y, Z):
    return (X < -0.4) & (Z > 1.44) & ((Y > 0.9) | (Y < -0.9)) # nearly empty

def obs_chair(X, Y, Z, x=1., x_thickness=0.15, y_area=0.3, y_bend = 0.1, h=0.33, z_thickness=0.2):
    """墙在 |Y|<=width/2 范围内存在；于 z∈[slot_center-Δ/2, slot_center+Δ/2] 留出槽"""
    d0 = y_area/2
    d1 = -y_area/2
    d2 = -y_bend
    h0 = h
    h1 = 2*h
    x0 = x - x_thickness/2
    x1 = x + x_thickness/2
    x0 -= ORIGIN_W[0]
    x1 -= ORIGIN_W[0]
    y0 = d0 - ORIGIN_W[1]
    y1 = - 2 * ORIGIN_W[1]
    z0 = h0 - z_thickness/2
    z1 = h0 + z_thickness/2
    box0 = _box(X, Y, Z, x0=x, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

    y0 = 0
    y1 = d1 - ORIGIN_W[1]
    z0 = 0
    z1 = h1
    box1 = _box(X, Y, Z, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

    y0 = 0
    y1 = d2 - ORIGIN_W[1]
    z0 = h1 - z_thickness/2
    z1 = h1 + z_thickness/2
    box2 = _box(X, Y, Z, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

    y0 = 2 * d0 - ORIGIN_W[1]
    y1 = - 2 * ORIGIN_W[1]
    z0 = 0
    z1 = h1
    box3 = _box(X, Y, Z, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
    
    return box0 | box1 | box2 | box3

# 7) 低矮门槛 + 头顶压梁 组合（同 x 位置上下夹击），练“抬腿+低头”同场景
def obs_bar_and_beam_combo(X, Y, Z, *, x=1.0, shin_z=0.30, beam_z=0.95, r=0.03, y_width=2.0):
    """小腿横杆 + 头顶横梁"""
    y0, y1 = -y_width/2, +y_width/2
    bar  = _cylinder_along_y(X, Y, Z, x=x, z=shin_z, r=r, y0=y0, y1=y1)
    beam = _cylinder_along_y(X, Y, Z, x=x, z=beam_z, r=r, y0=y0, y1=y1)
    return bar | beam

# 8) 腿部碎石阵：沿中线布置一串低立方体，需要“连续抬脚”但仍直线前进
def obs_ankle_block_field(X, Y, Z, *, xs=(0.6, 0.9, 1.2, 1.5), y_span=0.35, h=0.18, w=0.16):
    """若干方块：x 在 xs，y∈[-y_span,y_span]，z∈[0,h]"""
    mask = np.zeros_like(X, dtype=bool)
    for xc in xs:
        mask |= _box(X, Y, Z,
                     x0=xc - w/2, x1=xc + w/2,
                     y0=-y_span,  y1=+y_span,
                     z0=0.0,      z1=h)
    return mask

# 9) 侧摆肩训练：胸腔高度的“侧向片”，只遮挡 y 的 +侧或 -侧一部分，逼迫身体侧摆但不完全改路径
def obs_chest_1(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.24, y_cover=0.05, z0=0.5, z1=1.1):
    """两片错位侧挡板：一个覆盖 y>0 半边，另一个覆盖 y<0 半边"""
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=0.1, z0=z1, z1=2.0)
    fin2 = _box(X, Y, Z, x0=x0 - thickness, x1=x0 + thickness, y0=0.18, y1=1.0, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=x0 - thickness, x1=x0 + thickness, y0=-1.0, y1=-0.18, z0=0, z1=z0)
    return fin1 | fin2 | fin3

def obs_chest_2(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.05, z0=0.4, z1=1.15):
    """两片错位侧挡板：一个覆盖 y>0 半边，另一个覆盖 y<0 半边"""
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=1.0, z0=z1, z1=2.0)
    fin2 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=0.05, y1=1.0, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=-0.25, z0=0, z1=2.0)
    return fin1 | fin2 | fin3

def obs_low_corner(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.15, z0=0.4, z1=1.0):
    """通过box实现一个角落障碍物，一个L型通道需要拐弯通过"""
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=-0.5, x1=1.1, y0=-1.0, y1=-0.5, z0=0, z1=z0)
    fin2 = _box(X, Y, Z, x0=1.1, x1=1.3, y0=-1.0, y1=0.1, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=0.5, x1=0.7, y0=-0.1, y1=1.0, z0=0, z1=z0)
    return fin1 | fin2 | fin3

def obs_high_corner(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.15, z0=1.0, z1=1.0):
    """通过box实现一个角落障碍物，一个L型通道需要拐弯通过"""
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=-0.5, x1=1.3, y0=-1.0, y1=-0.5, z0=0, z1=z0)
    fin2 = _box(X, Y, Z, x0=1.3, x1=1.5, y0=-1.0, y1=0.1, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=0.5, x1=0.7, y0=-0.0, y1=1.0, z0=0, z1=z0)
    return fin1 | fin2 | fin3

# 10) 倾斜横梁：横杆随 y 改变高度，逼迫“侧身 + 点头”找相对低/高的一侧直穿
def obs_tilted_beam(X, Y, Z, *, x=1.4, r=0.035, z_mid=0.9, slope=0.25, y_width=2.0):
    """
    斜梁：z_beam(y)=z_mid + slope * y，贯穿 y∈[-y_width/2, y_width/2]
    slope>0 表示右侧更高
    """
    y0, y1 = -y_width/2, +y_width/2
    z_beam = z_mid + slope * Y
    # 近似为沿 y 的“变截面”圆柱：在每个体素，用其局部 z_beam 判断截面圆条件
    radial = (X - x)**2 + (Z - z_beam)**2 <= r**2
    band   = (Y >= y0) & (Y <= y1)
    return radial & band

# 11) 胸口高度“摆门”集合：三连杆（不同 x），训练“节奏化低头/挺胸”
def obs_three_beams(X, Y, Z, *, xs=(0.7, 1.1, 1.5), r=0.035, z=0.85, y_width=2.0):
    mask = np.zeros_like(X, dtype=bool)
    y0, y1 = -y_width/2, +y_width/2
    for xc in xs:
        mask |= _cylinder_along_y(X, Y, Z, x=xc, z=z, r=r, y0=y0, y1=y1)
    return mask

# 12) “猫洞墙”：薄墙 + 低位小洞，需要“深蹲或爬行”穿过（极限场景）
def obs_crawl_hole_wall(X, Y, Z, *, x=1.6, thickness=0.12, hole_w=0.45, hole_h=0.35, hole_z_mid=0.35, width=1.6):
    wall = (X >= x - thickness/2) & (X <= x + thickness/2) & (np.abs(Y) <= width/2) & (Z >= 0.0) & (Z <= 1.5)
    hole = (np.abs(Y) <= hole_w/2) & (Z >= hole_z_mid - hole_h/2) & (Z <= hole_z_mid + hole_h/2)
    return wall & (~hole)

# 13) 交错踏步条：一左一右错位的低条，诱导左右交替抬脚但仍可沿直线
def obs_staggered_strips(X, Y, Z, *, x_left=0.9, x_right=1.25, strip_len=0.50, strip_w=0.18, h=0.16, y_off=0.22):
    left  = _box(X, Y, Z, x0=x_left  - strip_w/2, x1=x_left  + strip_w/2, y0= -y_off - strip_len/2, y1= -y_off + strip_len/2, z0=0.0, z1=h)
    right = _box(X, Y, Z, x0=x_right - strip_w/2, x1=x_right + strip_w/2, y0= +y_off - strip_len/2, y1= +y_off + strip_len/2, z0=0.0, z1=h)
    return left | right

# 14) 胸口“之”字片：三片薄板在 y 方向交替占位，要求上身左右扭动（不改路径）
def obs_zigzag_chest_fins(X, Y, Z, *, xs=(0.8, 1.1, 1.4), thickness=0.10, y_cover=0.6, z0=0.65, z1=1.10):
    mask = np.zeros_like(X, dtype=bool)
    for i, xc in enumerate(xs):
        if i % 2 == 0:
            mask |= _box(X, Y, Z, x0=xc - thickness/2, x1=xc + thickness/2, y0=0.0, y1=+y_cover, z0=z0, z1=z1)
        else:
            mask |= _box(X, Y, Z, x0=xc - thickness/2, x1=xc + thickness/2, y0=-y_cover, y1=0.0, z0=z0, z1=z1)
    return mask

# 15) 头顶格栅：多根近距离横梁形成“栅格顶”，要求持续低头一段距离
def obs_overhead_grating(X, Y, Z, *, x0=1.0, x1=1.6, pitch=0.15, r=0.025, z=1.0, y_width=2.0):
    xs = np.arange(x0, x1 + 1e-6, pitch)
    mask = np.zeros_like(X, dtype=bool)
    y0, y1 = -y_width/2, +y_width/2
    for xc in xs:
        mask |= _cylinder_along_y(X, Y, Z, x=xc, z=z, r=r, y0=y0, y1=y1)
    return mask
