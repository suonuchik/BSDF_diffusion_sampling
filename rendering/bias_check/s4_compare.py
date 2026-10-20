"""Stage 4: yarn-region brightness ratio of the NN renders (with / without fixes) against the corrected reference.

Same mask and method as the earlier 1.028 measurement: the yarn mask is taken from the reference
(blue clearly dominant in display space), trimmed by 4 rows at the top and bottom.
Run (conda env bsdf-diff or bsdf35):  python s4_compare.py [spp]
"""
import os
import sys

import numpy as np
import mitsuba as mi

mi.set_variant('scalar_rgb')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
OLD = os.path.join(os.path.dirname(HERE), 'diffusion_bsdf_myresult')
W = np.array([0.2126, 0.7152, 0.0722])
SPP = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def load(path):
    a = np.array(mi.Bitmap(path), dtype=np.float64)[..., :3]
    return a


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


ref_name = 'scene_bsdf_fleece_fibers_khungurn_fixed_seed0_spp4096.exr'
ref = load(os.path.join(OUT, ref_name))
s = srgb(ref)
yarn = (s[..., 2] - s[..., 0] > 0.15) & (s[..., 2] > s[..., 1])
rows = np.where(yarn.mean(axis=1) > 0.5)[0]
core = np.zeros_like(yarn)
core[rows.min() + 4: rows.max() - 3, :] = True
core &= yarn
bg = np.zeros_like(core); bg[: rows.min() - 50, :] = True
fl = np.zeros_like(core); fl[rows.max() + 20:, :] = True

cands = [
    ('old NN render (512 spp, diffusion_bsdf_myresult)', os.path.join(OLD, 'scene_bsdf_fleece_fibers_nn.exr')),
    ('no fix', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_seed0_spp{SPP}.exr')),
    ('fix_base', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_fixbase_seed0_spp{SPP}.exr')),
    ('fix_chart', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_fixchart_seed0_spp{SPP}.exr')),
    ('fix_base + fix_chart', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_fixbase_fixchart_seed0_spp{SPP}.exr')),
    ('fix_base + fix_chart + fix_mis', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_fixbase_fixchart_fixmis_seed0_spp{SPP}.exr')),
    ('NN, all fixes (base + chart + MIS + eval)', os.path.join(OUT, f'scene_bsdf_fleece_fibers_nn_fixbase_fixchart_fixmis_fixeval_seed0_spp{SPP}.exr')),
    ('analytic sampler, same pipeline as NN', os.path.join(OUT, f'scene_bsdf_fleece_fibers_analytic_seed0_spp{SPP}.exr')),
    ('analytic sampler + original eval (N_TT bug)', os.path.join(OUT, f'scene_bsdf_fleece_fibers_analytic_origeval_seed0_spp{SPP}.exr')),
    ('original fiber BSDF (4096 spp)', os.path.join(OUT, 'scene_bsdf_fleece_fibers_khungurn_seed0_spp4096.exr')),
]
print(f'reference: {ref_name}   yarn pixels {int(core.sum())}')
print(f"{'render':48s} {'yarn R':>7s} {'G':>7s} {'B':>7s} {'Y':>7s} {'SE':>6s}   {'bg Y':>6s} {'floor Y':>7s}  neg px  max")
Yref = ref @ W
for name, path in cands:
    if not os.path.exists(path):
        print(f'{name:48s} (missing)')
        continue
    a = load(path)
    out = []
    for m in (core, bg, fl):
        x, y = a[m].mean(0), ref[m].mean(0)
        out.append((x / y, (x @ W) / (y @ W)))
    (r, yy), (_, bgy), (_, fly) = out
    d = (a @ W - Yref)[core]                       # pixel noise is independent -> standard error of the ratio
    se = d.std() / np.sqrt(d.size) / Yref[core].mean()
    print(f'{name:48s} {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} {yy:7.4f} {se:6.4f}   {bgy:6.4f} {fly:7.4f}  '
          f'{int((a < 0).any(-1).sum()):6d}  {a.max():.3g}')
