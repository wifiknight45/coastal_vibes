import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import animation
from noise import pnoise2

# =============================
# CONFIG
# =============================
WIDTH, HEIGHT = 300, 300
SEED = 42

SCALE_MAIN = 80.0
SCALE_RIDGES = 150.0
SCALE_DETAIL = 20.0

OCTAVES_MAIN = 4
OCTAVES_RIDGES = 2
OCTAVES_DETAIL = 6

DEMO_SECONDS = 5  # 5 or 10


# =============================
# NOISE HELPERS
# =============================
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


# =============================
# RIVERS & SIMPLE EROSION
# =============================
def add_rivers(terrain, strength=0.08):
    """Use another noise field to carve river channels."""
    river_mask = np.zeros_like(terrain)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r = perlin(x, y, 60.0, 3)
            river_mask[y, x] = r

    # Normalize and threshold to get thin channels
    river_mask = (river_mask - river_mask.min()) / (river_mask.max() - river_mask.min())
    channels = river_mask > 0.75
    carved = terrain.copy()
    carved[channels] -= strength
    carved = np.clip(carved, 0, 1)
    return carved, channels


def apply_erosion(terrain, iterations=20, talus=0.02):
    """Very simple thermal erosion: steep slopes get smoothed."""
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


# =============================
# BIOME COLORING (COAST + REDWOODS + CHAPARRAL + DUNES)
# =============================
def biome_color(h, y_norm):
    """
    h: height [0,1]
    y_norm: 0 bottom -> 1 top (use to bias redwoods to 'north')
    """
    if h < 0.32:
        return (0, 0.2, 0.5)      # deep ocean
    if h < 0.38:
        return (0, 0.4, 0.7)      # shallow water
    if h < 0.43:
        return (0.92, 0.88, 0.65) # beach sand / dunes
    if h < 0.6:
        # split between chaparral vs grass by latitude
        if y_norm > 0.6:
            return (0.16, 0.45, 0.22)  # redwood-ish darker green
        else:
            return (0.35, 0.55, 0.28)  # Santa Cruz hills / chaparral
    if h < 0.78:
        return (0.45, 0.35, 0.25) # Big Sur cliffs
    return (0.9, 0.9, 0.9)        # peaks


def build_color_map(terrain):
    color_map = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        y_norm = y / (HEIGHT - 1)
        for x in range(WIDTH):
            color_map[y, x] = biome_color(terrain[y, x], y_norm)
    return color_map


# =============================
# FOG, LIGHTING, SHADOWS (SIMPLE)
# =============================
def apply_lighting_and_fog(terrain, color_map, light_dir=np.array([1, 1, 2])):
    light_dir = light_dir / np.linalg.norm(light_dir)

    gy, gx = np.gradient(terrain)
    # approximate normals
    nx = -gx
    ny = -gy
    nz = np.ones_like(terrain)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= norm
    ny /= norm
    nz /= norm

    # dot with light
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot = np.clip(dot, 0.1, 1.0)  # keep some ambient

    shaded = color_map * dot[..., None]

    # fog based on distance from center
    yy, xx = np.mgrid[0:1:HEIGHT*1j, 0:1:WIDTH*1j]
    dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    fog = np.clip(dist * 1.5, 0, 1)
    fog_color = np.array([0.8, 0.85, 0.9])
    shaded = shaded * (1 - fog[..., None]) + fog_color * fog[..., None]

    shaded = np.clip(shaded, 0, 1)
    return shaded


# =============================
# OBJ EXPORT
# =============================
def export_obj(terrain, filename="california_coast.obj", scale_z=50.0):
    """
    Export terrain as a simple grid mesh OBJ for 3D printing / modeling.
    """
    h, w = terrain.shape
    with open(filename, "w") as f:
        f.write("# California-style coast terrain\n")
        # vertices
        for y in range(h):
            for x in range(w):
                z = terrain[y, x] * scale_z
                f.write(f"v {x} {y} {z}\n")
        # faces (two triangles per quad)
        def vid(x, y):
            return y * w + x + 1  # OBJ is 1-based

        for y in range(h - 1):
            for x in range(w - 1):
                v1 = vid(x, y)
                v2 = vid(x + 1, y)
                v3 = vid(x + 1, y + 1)
                v4 = vid(x, y + 1)
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")
    print(f"OBJ exported to {filename}")


# =============================
# VISUALIZATION HELPERS
# =============================
def show_2d_map(color_map, title="California Coast Perlin Landscape"):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.imshow(color_map)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


def show_3d_static(terrain, color_map, title="3D California-Style Coastline"):
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, terrain,
        facecolors=color_map,
        linewidth=0,
        antialiased=True,
        shade=False
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


def show_erosion_comparison(original, eroded):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Original Heightmap")
    axes[0].imshow(original, cmap="terrain")
    axes[0].axis("off")

    axes[1].set_title("After Erosion")
    axes[1].imshow(eroded, cmap="terrain")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(DEMO_SECONDS)
    plt.close()


# =============================
# FLY-THROUGH ANIMATION
# =============================
def flythrough_animation(terrain, color_map):
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, terrain,
        facecolors=color_map,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.set_title("Fly-through California Coast")

    def init():
        ax.view_init(elev=35, azim=45)
        return surf,

    def animate(i):
        # orbit around + slight pitch change
        azim = 45 + i * 2
        elev = 25 + 10 * np.sin(i * 0.05)
        ax.view_init(elev=elev, azim=azim)
        return surf,

    frames = int(DEMO_SECONDS * 20)  # ~20 fps
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=50, blit=False
    )

    plt.show()
    plt.close()


# =============================
# DEMO MODE
# =============================
def run_demo():
    # base terrain
    base = generate_base_terrain()

    # rivers + erosion
    with_rivers, river_mask = add_rivers(base)
    eroded = apply_erosion(with_rivers, iterations=25, talus=0.02)

    # color + lighting/fog
    color_map_raw = build_color_map(eroded)
    color_map_lit = apply_lighting_and_fog(eroded, color_map_raw)

    # export OBJ
    export_obj(eroded, "california_coast.obj")

    # 2D map (lit)
    show_2d_map(color_map_lit, title="California Coast (Biomes + Fog + Lighting)")

    # 3D static
    show_3d_static(eroded, color_map_lit, title="3D Coast (Rivers + Erosion + Fog)")

    # erosion comparison (heightmap only)
    show_erosion_comparison(base, eroded)

    # fly-through
    flythrough_animation(eroded, color_map_lit)


if __name__ == "__main__":
    run_demo()
