import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import animation
from noise import pnoise2

# ============================================
# CONFIG
# ============================================
WIDTH, HEIGHT = 300, 300
SEED = 42

SCALE_MAIN = 80.0
SCALE_RIDGES = 150.0
SCALE_DETAIL = 20.0

OCTAVES_MAIN = 4
OCTAVES_RIDGES = 2
OCTAVES_DETAIL = 6

DEMO_SECONDS = 3  # per static view
FPS = 20

# ============================================
# NOISE HELPERS
# ============================================
def perlin(x, y, scale, octaves):
    return pnoise2(
        x / scale,
        y / scale,
        octaves=octaves,
        persistence=0.5,
        lacunarity=2.0,
        repeatx=999999,
        repeaty=999999,
        base=SEED
    )


def generate_base_terrain():
    terrain = np.zeros((HEIGHT, WIDTH))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            main = perlin(x, y, SCALE_MAIN, OCTAVES_MAIN)
            ridges = abs(perlin(x, y, SCALE_RIDGES, OCTAVES_RIDGES))
            detail = perlin(x, y, SCALE_DETAIL, OCTAVES_DETAIL) * 0.15
            height_value = main * 0.6 + ridges * 0.5 + detail
            terrain[y, x] = height_value

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    yy, xx = np.mgrid[0:1:HEIGHT*1j, 0:1:WIDTH*1j]
    dist_from_center = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    island_mask = 1 - np.clip(dist_from_center * 1.4, 0, 1)
    terrain *= island_mask
    return terrain


# ============================================
# RIVERS AND EROSION
# ============================================
def add_rivers(terrain, strength=0.08):
    river_mask = np.zeros_like(terrain)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r = perlin(x, y, 60.0, 3)
            river_mask[y, x] = r

    river_mask = (river_mask - river_mask.min()) / (river_mask.max() - river_mask.min())
    channels = river_mask > 0.75
    carved = terrain.copy()
    carved[channels] -= strength
    carved = np.clip(carved, 0, 1)
    return carved, channels, river_mask


def apply_erosion(terrain, iterations=20, talus=0.02):
    t = terrain.copy()
    for _ in range(iterations):
        grad_y, grad_x = np.gradient(t)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mask = slope > talus
        smoothed = (t +
                    np.roll(t, 1, axis=0) +
                    np.roll(t, -1, axis=0) +
                    np.roll(t, 1, axis=1) +
                    np.roll(t, -1, axis=1)) / 5.0
        t[mask] = smoothed[mask]
    t = (t - t.min()) / (t.max() - t.min())
    return t


# ============================================
# BIOME COLORING
# ============================================
def biome_color(h, y_norm):
    if h < 0.32:
        return (0, 0.2, 0.5)
    if h < 0.38:
        return (0, 0.4, 0.7)
    if h < 0.43:
        return (0.92, 0.88, 0.65)
    if h < 0.6:
        if y_norm > 0.6:
            return (0.16, 0.45, 0.22)
        else:
            return (0.35, 0.55, 0.28)
    if h < 0.78:
        return (0.45, 0.35, 0.25)
    return (0.9, 0.9, 0.9)


def build_color_map(terrain):
    color_map = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        y_norm = y / (HEIGHT - 1)
        for x in range(WIDTH):
            color_map[y, x] = biome_color(terrain[y, x], y_norm)
    return color_map


# ============================================
# LIGHTING AND FOG
# ============================================
def compute_normals(terrain):
    gy, gx = np.gradient(terrain)
    nx = -gx
    ny = -gy
    nz = np.ones_like(terrain)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= norm
    ny /= norm
    nz /= norm
    return nx, ny, nz


def apply_lighting_and_fog(terrain, color_map, light_dir=np.array([1, 1, 2]), fog_strength=1.5):
    light_dir = light_dir / np.linalg.norm(light_dir)
    nx, ny, nz = compute_normals(terrain)
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot = np.clip(dot, 0.1, 1.0)
    shaded = color_map * dot[..., None]

    yy, xx = np.mgrid[0:1:HEIGHT*1j, 0:1:WIDTH*1j]
    dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    fog = np.clip(dist * fog_strength, 0, 1)
    fog_color = np.array([0.8, 0.85, 0.9])
    shaded = shaded * (1 - fog[..., None]) + fog_color * fog[..., None]
    shaded = np.clip(shaded, 0, 1)
    return shaded


# ============================================
# OBJ EXPORT
# ============================================
def export_obj(terrain, filename="california_coast.obj", scale_z=50.0):
    h, w = terrain.shape
    with open(filename, "w") as f:
        f.write("# California-style coast terrain\n")

        for y in range(h):
            for x in range(w):
                z = terrain[y, x] * scale_z
                f.write(f"v {x} {y} {z}\n")

        def vid(x, y):
            return y * w + x + 1

        for y in range(h - 1):
            for x in range(w - 1):
                v1 = vid(x, y)
                v2 = vid(x + 1, y)
                v3 = vid(x + 1, y + 1)
                v4 = vid(x, y + 1)
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")
    print(f"OBJ exported to {filename}")


# ============================================
# GENERIC VIEW HELPERS
# ============================================
def show_image(img, title, cmap=None):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


def show_3d_surface(terrain, facecolors=None, title="3D View", wireframe=False, points=False):
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if points:
        ax.scatter(X.flatten(), Y.flatten(), terrain.flatten(), c=terrain.flatten(), cmap="terrain", s=1)
    elif wireframe:
        ax.plot_wireframe(X, Y, terrain, color="black", linewidth=0.3)
    else:
        if facecolors is not None:
            ax.plot_surface(X, Y, terrain, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        else:
            ax.plot_surface(X, Y, terrain, cmap="terrain", linewidth=0, antialiased=True, shade=True)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


# ============================================
# CAMERA ANIMATION BASE
# ============================================
def camera_animation(terrain, facecolors, path_func, title="Flythrough"):
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, terrain,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.set_title(title)

    def init():
        ax.view_init(elev=35, azim=45)
        return surf,

    def animate(i):
        elev, azim = path_func(i)
        ax.view_init(elev=elev, azim=azim)
        return surf,

    frames = DEMO_SECONDS * FPS
    animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=1000 // FPS, blit=False
    )

    plt.show()
    plt.close()


# ============================================
# MODE IMPLEMENTATIONS (32 MODES)
# ============================================

# 1 Heightmap grayscale
def mode_heightmap_grayscale(terrain, *_):
    show_image(terrain, "Heightmap Grayscale", cmap="gray")


# 2 Slope map
def mode_slope_map(terrain, *_):
    gy, gx = np.gradient(terrain)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    slope = (slope - slope.min()) / (slope.max() - slope.min())
    show_image(slope, "Slope Map", cmap="inferno")


# 3 Aspect map
def mode_aspect_map(terrain, *_):
    gy, gx = np.gradient(terrain)
    aspect = np.arctan2(gy, gx)
    aspect = (aspect + np.pi) / (2 * np.pi)
    show_image(aspect, "Aspect Map", cmap="hsv")


# 4 Contour map
def mode_contour_map(terrain, *_):
    plt.figure(figsize=(7, 7))
    plt.title("Contour Map")
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)
    cs = plt.contour(X, Y, terrain, levels=20, cmap="terrain")
    plt.clabel(cs, inline=True, fontsize=8)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


# 5 Normal map
def mode_normal_map(terrain, *_):
    nx, ny, nz = compute_normals(terrain)
    normal_map = (np.stack([nx, ny, nz], axis=-1) + 1) / 2
    show_image(normal_map, "Normal Map")


# 6 Biome map raw
def mode_biome_raw(terrain, *_):
    color_map = build_color_map(terrain)
    show_image(color_map, "Biome Map Raw")


# 7 Biome map with fog only
def mode_biome_fog_only(terrain, *_):
    color_map = build_color_map(terrain)
    yy, xx = np.mgrid[0:1:HEIGHT*1j, 0:1:WIDTH*1j]
    dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    fog = np.clip(dist * 1.5, 0, 1)
    fog_color = np.array([0.8, 0.85, 0.9])
    shaded = color_map * (1 - fog[..., None]) + fog_color * fog[..., None]
    show_image(shaded, "Biome Map with Fog")


# 8 Biome map with lighting only
def mode_biome_lighting_only(terrain, *_):
    color_map = build_color_map(terrain)
    nx, ny, nz = compute_normals(terrain)
    light_dir = np.array([1, 1, 2])
    light_dir = light_dir / np.linalg.norm(light_dir)
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot = np.clip(dot, 0.1, 1.0)
    shaded = color_map * dot[..., None]
    show_image(shaded, "Biome Map with Lighting")


# 9 Season variations
def mode_season_variations(terrain, *_):
    base = build_color_map(terrain)

    def tint(img, color, strength):
        return np.clip(img * (1 - strength) + np.array(color) * strength, 0, 1)

    spring = tint(base, [0.6, 0.9, 0.6], 0.3)
    summer = tint(base, [1.0, 1.0, 0.7], 0.2)
    autumn = tint(base, [0.9, 0.6, 0.3], 0.4)
    winter = tint(base, [0.9, 0.9, 0.95], 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, img, title in zip(
        axes.flatten(),
        [spring, summer, autumn, winter],
        ["Spring", "Summer", "Autumn", "Winter"]
    ):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


# 10 Night mode
def mode_night_mode(terrain, *_):
    color_map = build_color_map(terrain)
    night = color_map * np.array([0.1, 0.2, 0.4])
    stars = np.random.rand(HEIGHT, WIDTH) > 0.995
    night[stars] = [1.0, 1.0, 1.0]
    show_image(night, "Night Mode Terrain")


# 11 River mask visualization
def mode_river_mask(terrain, with_rivers, channels, river_mask):
    show_image(river_mask, "River Mask", cmap="Blues")


# 12 Erosion heatmap
def mode_erosion_heatmap(base, eroded, *_):
    diff = base - eroded
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    show_image(diff, "Erosion Heatmap", cmap="magma")


# 13 Water flow simulation (simple downhill arrows)
def mode_water_flow(terrain, *_):
    gy, gx = np.gradient(terrain)
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 7))
    plt.title("Water Flow Directions")
    plt.imshow(terrain, cmap="terrain")
    step = 10
    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        -gx[::step, ::step],
        -gy[::step, ::step],
        color="cyan",
        scale=50
    )
    plt.axis("off")
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


# 14 Rainfall accumulation map (very simple)
def mode_rainfall_accumulation(terrain, *_):
    gy, gx = np.gradient(terrain)
    flow_mag = np.sqrt(gx ** 2 + gy ** 2)
    accumulation = 1 / (flow_mag + 0.01)
    accumulation = (accumulation - accumulation.min()) / (accumulation.max() - accumulation.min())
    show_image(accumulation, "Rainfall Accumulation Map", cmap="Blues")


# 15 Wireframe terrain
def mode_wireframe_terrain(terrain, *_):
    show_3d_surface(terrain, None, "Wireframe Terrain", wireframe=True)


# 16 Point cloud terrain
def mode_point_cloud_terrain(terrain, *_):
    show_3d_surface(terrain, None, "Point Cloud Terrain", points=True)


# 17 Shaded terrain with shadows (lighting + darker backfaces)
def mode_shadows_mode(terrain, *_):
    color_map = build_color_map(terrain)
    nx, ny, nz = compute_normals(terrain)
    light_dir = np.array([1, 1, 2])
    light_dir = light_dir / np.linalg.norm(light_dir)
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot = np.clip(dot, 0, 1)
    shaded = color_map * (0.2 + 0.8 * dot[..., None])
    show_3d_surface(terrain, shaded, "Shaded Terrain with Shadows")


# 18 Low poly terrain (downsampled)
def mode_low_poly_terrain(terrain, *_):
    factor = 4
    low = terrain[::factor, ::factor]
    show_3d_surface(low, None, "Low Poly Terrain")


# 19 Camera orbit slow
def mode_camera_orbit_slow(terrain, color_map_lit, *_):
    def path(i):
        azim = 45 + i * 1
        elev = 30
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Orbit Slow")


# 20 Camera orbit fast
def mode_camera_orbit_fast(terrain, color_map_lit, *_):
    def path(i):
        azim = 45 + i * 4
        elev = 30
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Orbit Fast")


# 21 Camera flyover north to south
def mode_camera_flyover_NS(terrain, color_map_lit, *_):
    def path(i):
        t = i / (DEMO_SECONDS * FPS)
        elev = 40
        azim = 180 * t
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Flyover North to South")


# 22 Camera flyover west to east
def mode_camera_flyover_WE(terrain, color_map_lit, *_):
    def path(i):
        t = i / (DEMO_SECONDS * FPS)
        elev = 40
        azim = 90 + 180 * t
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Flyover West to East")


# 23 Camera dive from sky
def mode_camera_dive(terrain, color_map_lit, *_):
    def path(i):
        t = i / (DEMO_SECONDS * FPS)
        elev = 80 - 50 * t
        azim = 45
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Dive from Sky")


# 24 Camera terrain crawl
def mode_camera_crawl(terrain, color_map_lit, *_):
    def path(i):
        t = i / (DEMO_SECONDS * FPS)
        elev = 20 + 5 * np.sin(t * 4 * np.pi)
        azim = 45 + 30 * t
        return elev, azim
    camera_animation(terrain, color_map_lit, path, "Camera Terrain Crawl")


# 25 Thermal infrared palette
def mode_infrared_palette(terrain, *_):
    show_image(terrain, "Thermal Infrared Palette", cmap="inferno")


# 26 Topographic poster style
def mode_poster_style(terrain, *_):
    levels = np.linspace(0, 1, 8)
    poster = np.digitize(terrain, levels) / len(levels)
    show_image(poster, "Topographic Poster Style", cmap="terrain")


# 27 Voxelized terrain (blocky)
def mode_voxelized(terrain, *_):
    levels = np.linspace(0, 1, 16)
    vox = np.digitize(terrain, levels) / len(levels)
    show_3d_surface(vox, None, "Voxelized Terrain")


# 28 ASCII terrain (printed to console)
def mode_ascii_terrain(terrain, *_):
    chars = " .:-=+*#%@"
    h, w = terrain.shape
    scaled = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    idx = (scaled * (len(chars) - 1)).astype(int)
    print("ASCII Terrain")
    for y in range(0, h, 4):
        line = "".join(chars[idx[y, x]] for x in range(0, w, 2))
        print(line)


# 29 Watercolor terrain (blurred soft colors)
def mode_watercolor(terrain, *_):
    from scipy.ndimage import gaussian_filter
    color_map = build_color_map(terrain)
    blurred = gaussian_filter(color_map, sigma=(2, 2, 0))
    show_image(blurred, "Watercolor Terrain")


# 30 Neon cyberpunk terrain
def mode_cyberpunk(terrain, *_):
    base = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    neon = np.zeros((HEIGHT, WIDTH, 3))
    neon[..., 0] = base
    neon[..., 1] = base ** 0.5
    neon[..., 2] = 1 - base
    show_image(neon, "Neon Cyberpunk Terrain")


# 31 OBJ preview mode
def mode_obj_preview(terrain, *_):
    show_3d_surface(terrain, None, "OBJ Preview")


# 32 Mesh quality visualization (gradient magnitude)
def mode_mesh_quality(terrain, *_):
    gy, gx = np.gradient(terrain)
    quality = np.sqrt(gx ** 2 + gy ** 2)
    quality = (quality - quality.min()) / (quality.max() - quality.min())
    show_image(quality, "Mesh Quality Visualization", cmap="viridis")


# ============================================
# DEMO MODE
# ============================================
def run_demo():
    base = generate_base_terrain()
    with_rivers, channels, river_mask = add_rivers(base)
    eroded = apply_erosion(with_rivers, iterations=25, talus=0.02)
    color_map_raw = build_color_map(eroded)
    color_map_lit = apply_lighting_and_fog(eroded, color_map_raw)

    export_obj(eroded, "california_coast.obj")

    modes = [
        lambda: mode_heightmap_grayscale(eroded),
        lambda: mode_slope_map(eroded),
        lambda: mode_aspect_map(eroded),
        lambda: mode_contour_map(eroded),
        lambda: mode_normal_map(eroded),
        lambda: mode_biome_raw(eroded),
        lambda: mode_biome_fog_only(eroded),
        lambda: mode_biome_lighting_only(eroded),
        lambda: mode_season_variations(eroded),
        lambda: mode_night_mode(eroded),
        lambda: mode_river_mask(eroded, with_rivers, channels, river_mask),
        lambda: mode_erosion_heatmap(base, eroded),
        lambda: mode_water_flow(eroded),
        lambda: mode_rainfall_accumulation(eroded),
        lambda: mode_wireframe_terrain(eroded),
        lambda: mode_point_cloud_terrain(eroded),
        lambda: mode_shadows_mode(eroded),
        lambda: mode_low_poly_terrain(eroded),
        lambda: mode_camera_orbit_slow(eroded, color_map_lit),
        lambda: mode_camera_orbit_fast(eroded, color_map_lit),
        lambda: mode_camera_flyover_NS(eroded, color_map_lit),
        lambda: mode_camera_flyover_WE(eroded, color_map_lit),
        lambda: mode_camera_dive(eroded, color_map_lit),
        lambda: mode_camera_crawl(eroded, color_map_lit),
        lambda: mode_infrared_palette(eroded),
        lambda: mode_poster_style(eroded),
        lambda: mode_voxelized(eroded),
        lambda: mode_ascii_terrain(eroded),
        lambda: mode_watercolor(eroded),
        lambda: mode_cyberpunk(eroded),
        lambda: mode_obj_preview(eroded),
        lambda: mode_mesh_quality(eroded),
    ]

    for i, m in enumerate(modes, start=1):
        print(f"Running mode {i} of {len(modes)}")
        m()


if __name__ == "__main__":
    run_demo()

