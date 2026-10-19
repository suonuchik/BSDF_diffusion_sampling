"""
Render fleece (idx=26) with helical fiber geometry using BSDF_diffusion_sampling's myyarnbsdf (NN).
Camera is closer than render_yarn26_fibers_nn.py (origin=[0,-3,1.5]).

Run from: /home/suno/research/BSDF_diffusion_sampling/rendering/
    conda run -n bsdf35 python render_yarn26_fibers_nn_close.py [--spp 256] [--name fleece]
"""

import os, sys, argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# -----------------------------------------------------------------------
# Parse this script's own args FIRST, before bsdf_myresult hijacks sys.argv
_my_parser = argparse.ArgumentParser()
_my_parser.add_argument('--spp',  type=int, default=256)
_my_parser.add_argument('--name', type=str, default='fleece',
                        choices=['fleece','gabardine','silk','cotton','polyester'])
_my_parser.add_argument('--res',  type=int, default=201)
_my_parser.add_argument('--seed', type=int, default=0,
                        help='Fiber placement seed (use the same value for NN and physics-based renders)')
args = _my_parser.parse_args()

# -----------------------------------------------------------------------
# Add rendering/ to sys.path so bsdf_myresult's utils imports work
_rendering_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _rendering_dir)

# bsdf_myresult.py calls argparse.parse_args() at module level.
# Feed it empty args so it uses defaults and doesn't error on our --name/--spp flags.
sys.argv = [sys.argv[0]]

# Import bsdf_myresult — this sets up: mi variant, device, BSDFs, MyYarnBSDF, etc.
import bsdf_myresult  # noqa: E402

# Register the custom BSDF plugins (only done in __main__ normally)
import mitsuba as mi
mi.register_bsdf("mybsdf",     lambda props: bsdf_myresult.MyBSDF(props))
mi.register_bsdf("myyarnbsdf", lambda props: bsdf_myresult.MyYarnBSDF(props))

from mitsuba import ScalarTransform4f as sT

OUTPUT_DIR = os.path.join(_rendering_dir, 'diffusion_bsdf_myresult')
ENVMAP     = os.path.join(_rendering_dir, 'matpreview', 'envmap.exr')

# -----------------------------------------------------------------------
# Material table (same as neuralyarn/config/parameters.py)
MATERIALS = {
    'fleece':    dict(N=300, Rho=0.300, Alpha=0.240, idx=26),
    'gabardine': dict(N=450, Rho=0.250, Alpha=0.120, idx=27),
    'silk':      dict(N=300, Rho=0.200, Alpha=0.000, idx=28),
    'cotton':    dict(N=600, Rho=0.350, Alpha=0.060, idx=29),
    'polyester': dict(N=200, Rho=0.400, Alpha=0.200, idx=30),
}

mat = MATERIALS[args.name]
print(f"Material : {args.name}  idx={mat['idx']}  N={mat['N']}  Rho={mat['Rho']}  Alpha={mat['Alpha']}")

# -----------------------------------------------------------------------
# Helical fiber geometry (from neuralyarn/utils/geometry.py, inlined)

def cross_section_circle(macro_radius, count, density):
    micro_radius = np.sqrt((density * macro_radius**2) / count)
    radius = macro_radius - micro_radius
    sample_x = np.random.rand(count * 10) * 2 * radius - radius
    sample_y = np.random.rand(count * 10) * 2 * radius - radius
    x_points = np.full(count, np.inf)
    y_points = np.full(count, np.inf)
    idx = 0
    for x, y in zip(sample_x, sample_y):
        micro_dist = np.min(np.sqrt((x_points - x)**2 + (y_points - y)**2))
        macro_dist = np.sqrt(x**2 + y**2)
        if macro_dist < radius and micro_dist >= 2 * micro_radius:
            x_points[idx] = x; y_points[idx] = y; idx += 1
        if idx == count: break
    x_points = x_points[~np.isinf(x_points)]
    y_points = y_points[~np.isinf(y_points)]
    if x_points.shape[0] != count:
        raise RuntimeError(f"Unable to place all fibers ({x_points.shape[0]}/{count}).")
    return np.vstack([x_points, y_points]), micro_radius


def helix(points, radius, offset, twist_factor, up_vector=np.array([0, 0, 1])):
    xs, ys, zs = [], [], []
    twist_angle = 0
    for p1, p2 in zip(points[:-1], points[1:]):
        d = p2 - p1; dist = np.linalg.norm(d)
        if np.isclose(dist, 0):
            xs.append(xs[-1]); ys.append(ys[-1]); zs.append(zs[-1]); continue
        tangent  = d / dist
        binormal = np.cross(up_vector, tangent); binormal /= np.linalg.norm(binormal)
        normal   = np.cross(tangent, binormal);  normal   /= np.linalg.norm(normal)
        twist_angle += np.pi * twist_factor * dist / radius
        rot      = Rot.from_rotvec(tangent * twist_angle)
        binormal = rot.apply(binormal); normal = rot.apply(normal)
        cs = p1 + offset[0][np.newaxis].T * binormal + offset[1][np.newaxis].T * normal
        cs = cs.T
        xs.append(cs[0]); ys.append(cs[1]); zs.append(cs[2])
    return np.dstack([np.asarray(xs).T, np.asarray(ys).T, np.asarray(zs).T])  # [N, pts, 3]

# -----------------------------------------------------------------------
ply_radius = 0.5
resolution = args.res
start, end = -2.5, 2.5

ply = np.stack([np.linspace(start, end, resolution),
                np.zeros(resolution), np.zeros(resolution)], axis=1)

np.random.seed(args.seed)
fiber_offset, fiber_radius = cross_section_circle(ply_radius, mat['N'], mat['Rho'])
fiber_curves = helix(ply, ply_radius, fiber_offset, mat['Alpha'])

print(f"Fibers   : {fiber_curves.shape[0]} × {fiber_curves.shape[1]} pts, r={fiber_radius:.5f}")

curves_path = f'/tmp/{args.name}_fibers_nn_close.txt'
with open(curves_path, 'w') as f:
    for i in range(fiber_curves.shape[0]):
        for j in range(fiber_curves.shape[1]):
            p = fiber_curves[i, j]
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {fiber_radius:.6f}\n')
        f.write('\n')
print(f"Curves   : {curves_path}")

# -----------------------------------------------------------------------
# Scene: linearcurve + myyarnbsdf
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path', 'max_depth': -1},
    'sensor': {
        'type': 'perspective',
        'fov': 40, 'fov_axis': 'smaller',
        'to_world': sT.look_at(origin=[0,-3,1.5], target=[0,0,0], up=[0,0,1]),
        'film': {'type':'hdrfilm','width':512,'height':512,'pixel_format':'rgb'},
        'sampler': {'type':'independent','sample_count':64},
    },
    'light_a': {
        'type': 'sphere', 'center': [2,-4,3], 'radius': 0.3,
        'emitter': {'type':'area','radiance':{'type':'rgb','value':[80,80,75]}},
    },
    'light_b': {
        'type': 'sphere', 'center': [-2,-3,2], 'radius': 0.3,
        'emitter': {'type':'area','radiance':{'type':'rgb','value':[50,50,48]}},
    },
    'envmap': {'type':'envmap','filename':ENVMAP,'scale':1.5},
    'yarn': {
        'type': 'linearcurve',
        'filename': curves_path,
        'bsdf': {
            'type': 'myyarnbsdf',
            'idx': mat['idx'],
            'albedo': [1.0, 1.0, 1.0],
            'ckpt_tag': 'yarn_curve',
        },
    },
    'floor': {
        'type': 'rectangle',
        'to_world': sT.translate([0,0,-0.52]) @ sT.scale([6,6,1]),
        'bsdf': {'type':'diffuse','reflectance':{'type':'rgb','value':0.3}},
    },
})

# -----------------------------------------------------------------------
# Render
spp_per_pass = 4
num_passes   = max(1, args.spp // spp_per_pass)
image = None
for i in range(num_passes):
    img = mi.render(scene, spp=spp_per_pass, seed=i).numpy()
    image = img if image is None else image + img
    if (i + 1) % 8 == 0 or i == num_passes - 1:
        print(f"  pass {i+1}/{num_passes}")
image = image / num_passes

# -----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
stem = os.path.join(OUTPUT_DIR, f'scene_bsdf_{args.name}_fibers_nn_close')
mi.util.write_bitmap(stem + '.png', image, args.spp)
mi.util.write_bitmap(stem + '.exr', image, args.spp)
print(f"Saved    : {stem}.png")
