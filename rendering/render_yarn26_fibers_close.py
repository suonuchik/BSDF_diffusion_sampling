"""
Render fleece (idx=26) using create_single_yarn helical fiber geometry.
Uses neuralyarn's khungurn BSDF (physics-based) on actual helical fiber curves.
Camera is closer than render_yarn26_fibers.py (origin=[0,-3,1.5]).

Run from any directory with:
    conda run -n neuralyarn python <this_file> [--spp 256] [--name fleece]
"""

import os
import sys
import argparse
import importlib.util

NEURALYARN = '/home/suno/research/neuralyarn'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_bsdf_myresult')
ENVMAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matpreview', 'envmap.exr')

# Change to neuralyarn so config/__init__.py's relative Mitsuba path resolves
os.chdir(NEURALYARN)
sys.path.insert(0, NEURALYARN)

# Remove the script's own directory from sys.path to avoid utils/ conflict
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

# Load config to add neuralyarn Mitsuba to sys.path and set env vars
import config  # noqa: E402

import mitsuba as mi  # noqa: E402
import drjit as dr    # noqa: E402
mi.set_variant('llvm_ad_rgb')

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from mitsuba import ScalarTransform4f as sT

# Register khungurn BSDF plugin (side effect: mi.register_bsdf('khungurn', ...))
import bsdf.khungurn  # noqa: E402

# -----------------------------------------------------------------------
# Geometry helpers (from neuralyarn/utils/geometry.py, inlined to avoid
# package conflict with BSDF_diffusion_sampling/rendering/utils/)

def cross_section_circle(macro_radius, count, density):
    micro_radius = np.sqrt((density * macro_radius**2) / count)
    radius = macro_radius - micro_radius
    sample_x = np.random.rand(count * 10) * 2 * radius - radius
    sample_y = np.random.rand(count * 10) * 2 * radius - radius
    x_points = np.full(count, np.inf)
    y_points = np.full(count, np.inf)
    idx = 0
    for x, y in zip(sample_x, sample_y):
        micro_dist  = np.min(np.sqrt((x_points - x)**2 + (y_points - y)**2))
        macro_dist  = np.sqrt(x**2 + y**2)
        if macro_dist < radius and micro_dist >= 2 * micro_radius:
            x_points[idx] = x
            y_points[idx] = y
            idx += 1
        if idx == count:
            break
    x_points = x_points[~np.isinf(x_points)]
    y_points = y_points[~np.isinf(y_points)]
    if x_points.shape[0] != count:
        raise RuntimeError(f"Unable to generate all fibers ({x_points.shape[0]}/{count}).")
    return np.vstack([x_points, y_points]), micro_radius


def helix(points, radius, offset, twist_factor, up_vector=np.array([0, 0, 1])):
    xs, ys, zs = [], [], []
    twist_angle = 0
    for p1, p2 in zip(points[:-1], points[1:]):
        displacement = p2 - p1
        distance = np.linalg.norm(displacement)
        if np.isclose(distance, 0):
            xs.append(xs[-1]); ys.append(ys[-1]); zs.append(zs[-1])
            continue
        tangent  = displacement / distance
        binormal = np.cross(up_vector, tangent)
        binormal /= np.linalg.norm(binormal)
        normal   = np.cross(tangent, binormal)
        normal   /= np.linalg.norm(normal)
        twist_angle += np.pi * twist_factor * distance / radius
        rot      = Rot.from_rotvec(tangent * twist_angle)
        binormal = rot.apply(binormal)
        normal   = rot.apply(normal)
        cs = p1 + offset[0][np.newaxis].T * binormal + offset[1][np.newaxis].T * normal
        cs = cs.T
        xs.append(cs[0]); ys.append(cs[1]); zs.append(cs[2])
    xs = np.asarray(xs).T
    ys = np.asarray(ys).T
    zs = np.asarray(zs).T
    return np.dstack([xs, ys, zs])   # [N_fibers, n_pts, 3]

# -----------------------------------------------------------------------
# Material table
MATERIALS = {
    'fleece':    dict(N=300, Rho=0.300, Alpha=0.240,
                      R=[0.040,0.087,0.087], TT=[0.452,0.725,0.948],
                      R_long=7.238, TT_long=10.000, TT_azim=25.989),
    'gabardine': dict(N=450, Rho=0.250, Alpha=0.120,
                      R=[0.185,0.047,0.069], TT=[0.999,0.330,0.354],
                      R_long=2.141, TT_long=10.000, TT_azim=23.548),
    'silk':      dict(N=300, Rho=0.200, Alpha=0.000,
                      R=[0.745,0.008,0.070], TT=[0.620,0.553,0.562],
                      R_long=1.000, TT_long=10.000, TT_azim=19.823),
    'cotton':    dict(N=600, Rho=0.350, Alpha=0.060,
                      R=[0.989,0.959,0.874], TT=[0.999,0.999,0.999],
                      R_long=1.000, TT_long=27.197, TT_azim=38.269),
    'polyester': dict(N=200, Rho=0.400, Alpha=0.200,
                      R=[0.700,0.700,0.700], TT=[0.600,0.000,0.800],
                      R_long=5.238, TT_long=10.000, TT_azim=25.000),
}

# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--spp',  type=int, default=256)
parser.add_argument('--name', type=str, default='fleece', choices=list(MATERIALS.keys()))
parser.add_argument('--res',  type=int, default=201, help='Points per fiber curve')
parser.add_argument('--seed', type=int, default=0,
                    help='Fiber placement seed (use the same value for NN and physics-based renders)')
args = parser.parse_args()

mat = MATERIALS[args.name]
print(f"Material : {args.name}  N={mat['N']}  Rho={mat['Rho']}  Alpha={mat['Alpha']}")

# -----------------------------------------------------------------------
# Generate helical fiber geometry
ply_radius = 0.5
resolution = args.res
start, end = -2.5, 2.5

ply = np.stack([
    np.linspace(start, end, resolution),
    np.zeros(resolution),
    np.zeros(resolution),
], axis=1)

np.random.seed(args.seed)
fiber_offset, fiber_radius = cross_section_circle(ply_radius, mat['N'], mat['Rho'])
fiber_curves = helix(ply, ply_radius, fiber_offset, mat['Alpha'])

print(f"Fibers   : {fiber_curves.shape[0]} × {fiber_curves.shape[1]} pts, r={fiber_radius:.5f}")

# -----------------------------------------------------------------------
# Write curves to temp file
curves_path = f'/tmp/{args.name}_fibers_close.txt'
with open(curves_path, 'w') as f:
    for i in range(fiber_curves.shape[0]):
        for j in range(fiber_curves.shape[1]):
            p = fiber_curves[i, j]
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {fiber_radius:.6f}\n')
        f.write('\n')
print(f"Curves   : written to {curves_path}")

# -----------------------------------------------------------------------
# Scene
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path', 'max_depth': -1},
    'sensor': {
        'type': 'perspective',
        'fov': 40,
        'fov_axis': 'smaller',
        'to_world': sT.look_at(origin=[0, -3, 1.5], target=[0, 0, 0], up=[0, 0, 1]),
        'film': {
            'type': 'hdrfilm',
            'width': 512, 'height': 512,
            'pixel_format': 'rgb',
        },
        'sampler': {'type': 'independent', 'sample_count': 64},
    },
    'light_a': {
        'type': 'sphere', 'center': [2, -4, 3], 'radius': 0.3,
        'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [80, 80, 75]}},
    },
    'light_b': {
        'type': 'sphere', 'center': [-2, -3, 2], 'radius': 0.3,
        'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [50, 50, 48]}},
    },
    'envmap': {
        'type': 'envmap',
        'filename': ENVMAP,
        'scale': 1.5,
    },
    'yarn': {
        'type': 'linearcurve',
        'filename': curves_path,
        'bsdf': {
            'type': 'khungurn',
            'reflectance':   {'type': 'rgb', 'value': mat['R']},
            'transmittance': {'type': 'rgb', 'value': mat['TT']},
            'r_longwidth':   mat['R_long'],
            'tt_longwidth':  mat['TT_long'],
            'tt_aziwidth':   mat['TT_azim'],
        },
    },
    'floor': {
        'type': 'rectangle',
        'to_world': sT.translate([0, 0, -0.52]) @ sT.scale([6, 6, 1]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': 0.3},
        },
    },
})

# -----------------------------------------------------------------------
# Render
spp_per_pass = 32
num_passes   = max(1, args.spp // spp_per_pass)
image = None
for i in range(num_passes):
    img = mi.render(scene, spp=spp_per_pass, seed=i)
    image = img if image is None else image + img
    print(f"  pass {i + 1}/{num_passes}")
image = image / num_passes

# -----------------------------------------------------------------------
# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
stem = os.path.join(OUTPUT_DIR, f'scene_bsdf_{args.name}_fibers_close')
mi.util.write_bitmap(stem + '.exr', image)
mi.util.write_bitmap(stem + '.png', image)
print(f"Saved    : {stem}.png")
