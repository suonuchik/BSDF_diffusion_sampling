"""Stage 6: per-bounce energy ratio E[weight] / int eval d omega of the (patched) NN sampler over a dense grid of
incident directions, to find where the remaining image-level bias (+1.4 % with fix_base + fix_chart) comes from.

theta_i in [-85, 85] deg (step 10), phi_i in [-180, 180) deg (step 30) in yarn coordinates.
Run (conda env bsdf35):  python s6_sweep_wi.py [--n 131072] [--no_fix]
Outputs results/s6_sweep_wi[_nofix].csv and figures/s6_sweep_wi[_nofix].png
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERING = os.path.dirname(HERE)
_a = argparse.ArgumentParser()
_a.add_argument('--n', type=int, default=1 << 17)
_a.add_argument('--no_fix', action='store_true', help='original plugin behaviour (no fix_base, no fix_chart)')
ARGS = _a.parse_args()

os.chdir(RENDERING)
sys.path.insert(0, RENDERING)
sys.argv = [sys.argv[0]]
import bsdf_myresult as bm  # noqa: E402
import mitsuba as mi  # noqa: E402
import drjit as dr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, HERE)
import yarn_patches  # noqa: E402

IDX = 26
LUM = np.array([0.2126, 0.7152, 0.0722])
N_T, N_P = 1800, 720
if not ARGS.no_fix:
    yarn_patches.apply_base_fix(bm)
_cls = yarn_patches.make_patched_bsdf(bm, fix_chart=not ARGS.no_fix)
mi.register_bsdf('myyarnbsdf_patched', lambda props: _cls(props))
plugin = mi.load_dict({'type': 'myyarnbsdf_patched', 'idx': IDX, 'albedo': [1.0, 1.0, 1.0], 'ckpt_tag': 'yarn_curve'})
dev = bm.device


def eval_integral(ti, pi):
    """int eval d omega with the plugin's own eval (eval_rgb of khungurn_bsdf), on a fine grid"""
    kb = bm.bsdf_materials[IDX]
    te = torch.linspace(-math.pi / 2, math.pi / 2, N_T + 1, device=dev, dtype=torch.float64)
    tc = 0.5 * (te[1:] + te[:-1])
    pc = -math.pi + (torch.arange(N_P, device=dev, dtype=torch.float64) + 0.5) * (2 * math.pi / N_P)
    dA_t = (torch.sin(te[1:]) - torch.sin(te[:-1])) * (2 * math.pi / N_P)
    tot = torch.zeros(3, dtype=torch.float64, device=dev)
    for ch in torch.arange(N_T, device=dev).split(300):
        T, P = torch.meshgrid(tc[ch], pc, indexing='ij')
        wo = torch.stack([T.reshape(-1), P.reshape(-1)], 1).float()
        wi = torch.tensor([[ti, pi]], device=dev, dtype=torch.float32).expand_as(wo)
        val = kb.eval_rgb(wi, wo).double()
        tot += (val * dA_t[ch][:, None].expand(-1, N_P).reshape(-1, 1)).sum(0)
    return tot.cpu().numpy()


def mc(ti, pi, n):
    wi = (math.sin(ti), math.cos(ti) * math.cos(pi), math.cos(ti) * math.sin(pi))
    s1 = np.zeros(3); s2 = np.zeros(3); m = 0; neg = 0
    for b in range(0, n, 1 << 16):
        k = min(1 << 16, n - b)
        si = dr.zeros(mi.SurfaceInteraction3f, k)
        si.wi = mi.Vector3f(*[np.full(k, c, dtype=np.float32) for c in wi])
        u = np.random.default_rng(b).random((k, 3), dtype=np.float32)
        _, w = plugin.sample(mi.BSDFContext(), si, mi.Float(u[:, 0]), mi.Point2f(u[:, 1], u[:, 2]), True)
        w = np.stack([np.array(w[0]), np.array(w[1]), np.array(w[2])], -1).astype(np.float64)
        w[~np.isfinite(w)] = 0
        neg += int((w < 0).any(1).sum())
        s1 += w.sum(0); s2 += (w ** 2).sum(0); m += k
        dr.flush_malloc_cache()
    mean = s1 / m
    se = np.sqrt(np.maximum(s2 / m - mean ** 2, 0) / m)
    return mean, se, neg / m


def main():
    tag = '_nofix' if ARGS.no_fix else ''
    rows = []
    thetas = np.arange(-85, 86, 10)
    phis = np.arange(-180, 180, 30)
    for t in thetas:
        for p in phis:
            ti, pi = math.radians(t), math.radians(p)
            ref = eval_integral(ti, pi)
            mean, se, neg = mc(ti, pi, ARGS.n)
            rY = (mean @ LUM) / (ref @ LUM)
            seY = math.sqrt((se ** 2) @ LUM ** 2) / (ref @ LUM)
            rows.append((t, p, ref @ LUM, rY, seY, mean[0] / ref[0], mean[2] / ref[2], neg))
            print(f'theta_i {t:+4d} phi_i {p:+5d}: ratio Y {rY:.4f} +- {seY:.4f}  (R {mean[0]/ref[0]:.4f}, B {mean[2]/ref[2]:.4f})  '
                  f'int eval Y {ref @ LUM:.3f}  neg {neg:.4f}', flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    arr = np.array(rows)
    hdr = 'theta_i_deg,phi_i_deg,int_eval_Y,ratio_Y,se_Y,ratio_R,ratio_B,neg_frac'
    np.savetxt(os.path.join(HERE, 'results', f's6_sweep_wi{tag}.csv'), arr, delimiter=',', header=hdr, comments='', fmt='%.6g')
    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid = arr[:, 3].reshape(len(thetas), len(phis))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, origin='lower', cmap='RdBu_r', vmin=0.97, vmax=1.03, aspect='auto',
                   extent=[-195, 165, -90, 90])
    for (i, t) in enumerate(thetas):
        for (j, p) in enumerate(phis):
            ax.text(p, t, f'{grid[i, j]:.3f}', ha='center', va='center', fontsize=6)
    ax.set_xlabel('phi_i [deg] (yarn frame: 0 = +y grazing, 90 = normal)')
    ax.set_ylabel('theta_i [deg] (from the plane normal to the fiber)')
    ax.set_title('per-bounce E[weight] / int eval  (' + ('original plugin' if ARGS.no_fix else 'fix_base + fix_chart') + ')')
    fig.colorbar(im, ax=ax, label='ratio (1 = unbiased)')
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'figures', f's6_sweep_wi{tag}.png'), dpi=150, bbox_inches='tight')
    print('mean ratio over grid', grid.mean(), 'min', grid.min(), 'max', grid.max())


if __name__ == '__main__':
    main()
