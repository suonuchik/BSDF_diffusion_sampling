"""Stage 2: one fiber bounce of MyYarnBSDF (diffusion sampler, fleece idx=26), split into R and TT.

For one bounce the path tracer multiplies the throughput by  weight = eval(wo) / pdf(wo)  of the sampled wo.
An unbiased sampler gives  E[weight] = int eval(wo) d omega  for each lobe separately, so we compare
  E[eval_R / pdf]  with  int eval_R d omega   and   E[eval_TT / pdf]  with  int eval_TT d omega
(eval already contains cos theta_o, as in khungurn.py).  The plugin's own weight (bs value) is also averaged.

Extra checks on the same samples:
  * share of raw samples with |theta_o| > pi/2 (outside the yarn chart) and their weight
  * forward PDF (from sampling) vs inverse-ODE PDF (MyYarnBSDF.pdf, used for MIS) of the same direction
  * effect of the base-distribution log_prob normalisation (uses exp(log_scale) instead of exp(log_scale)+eps)

Run (conda env bsdf35):  python s2_single_bounce.py [--n 1048576]
Outputs results/s2_single_bounce.csv
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERING = os.path.dirname(HERE)
_args = argparse.ArgumentParser()
_args.add_argument('--n', type=int, default=1 << 20)
_args.add_argument('--batch', type=int, default=1 << 18)
_args.add_argument('--fix_base', action='store_true', help='apply yarn_patches.apply_base_fix (affects all columns)')
_args.add_argument('--fix_chart', action='store_true', help='use the PatchedYarnBSDF for the plugin column')
_args.add_argument('--fix_mis', action='store_true', help='PatchedYarnBSDF with consistent MIS pdf (plugin column)')
_args.add_argument('--analytic', action='store_true', help='plugin column = analytic khungurn sampler (pipeline check)')
_args.add_argument('--theta_i', type=float, nargs='*', default=None)
ARGS = _args.parse_args()

# bsdf_myresult parses sys.argv at import time and imports utils/ relative to rendering/
os.chdir(RENDERING)
sys.path.insert(0, RENDERING)
sys.argv = [sys.argv[0]]
import bsdf_myresult as bm  # noqa: E402
import mitsuba as mi  # noqa: E402
import drjit as dr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from utils.mitsuba_brdf_yarn import _cos2_normalized_gaussian, _uct_gaussian_pdf, _regularize  # noqa: E402

IDX = 26
THETA_I = [-60.0, -30.0, 0.0, 30.0, 60.0, 80.0]
PHI_I = [0.7, 2.0, -1.5]           # radians; -1.5: wi below the surface (z < 0)
N_T, N_P = 3600, 720               # integration grid (0.05 deg in theta; R lobe width 7.2 deg)
LUM = np.array([0.2126, 0.7152, 0.0722])
dev = bm.device

sys.path.insert(0, HERE)
import yarn_patches  # noqa: E402
if ARGS.fix_base:
    yarn_patches.apply_base_fix(bm)
if ARGS.theta_i:
    THETA_I = ARGS.theta_i
TAG = ('_fixbase' if ARGS.fix_base else '') + ('_fixchart' if ARGS.fix_chart else '') \
    + ('_fixmis' if ARGS.fix_mis else '') + ('_analytic' if ARGS.analytic else '')
if ARGS.analytic:
    _cls = yarn_patches.make_analytic_bsdf(bm)
else:
    _cls = yarn_patches.make_patched_bsdf(bm, fix_chart=ARGS.fix_chart, fix_mis=ARGS.fix_mis)
mi.register_bsdf("myyarnbsdf_patched", lambda props: _cls(props))
plugin = mi.load_dict({'type': 'myyarnbsdf_patched', 'idx': IDX, 'albedo': [1.0, 1.0, 1.0], 'ckpt_tag': 'yarn_curve'})
kb = bm.bsdf_materials[IDX]
# the networks, loaded exactly as MyYarnBSDF does (same checkpoint folder)
ckpt_dir = os.path.join(RENDERING, '..', 'learning_repo_cleanup', 'checkpoints_new', f'bsdf_{IDX}_yarn_curve')
D_sample = bm.NN_cond_pos(input_dim=6, output_dim=2, N_NEURONS=32, POSITIONAL_ENCODING_BASIS_NUM=5).to(dev)
D_sample.load_state_dict(torch.load(os.path.join(ckpt_dir, f'brdf_rectify_network{IDX}.pth'), map_location=dev))
D_sample.eval()
D_base = bm.NN_cond_pretrain_spherical_one(input_dim=2, N_NEURONS=16).to(dev)
D_base.load_state_dict(torch.load(os.path.join(ckpt_dir, f'brdf_pretrain_network{IDX}.pth'), map_location=dev))


def eval_parts(ti, pi, to, po):
    """(N,) angles in yarn coords -> R and TT parts of eval_rgb, each (N, 3), including cos(theta_o)."""
    c_r = torch.tensor(kb.c_r_rgb, dtype=to.dtype, device=to.device)
    c_tt = torch.tensor(kb.c_tt_rgb, dtype=to.dtype, device=to.device)
    ti = torch.as_tensor(ti, dtype=to.dtype, device=to.device).expand_as(to)
    pi = torch.as_tensor(pi, dtype=to.dtype, device=to.device).expand_as(to)
    F_R = c_r + (1.0 - c_r) * ((1.0 - torch.abs(torch.cos(ti))) ** 5)[:, None]
    M_R = _cos2_normalized_gaussian(to, -ti, kb.r_longwidth)
    S_R = F_R * (M_R / (2.0 * math.pi))[:, None]
    M_TT = _cos2_normalized_gaussian(to, -ti, kb.tt_longwidth)
    phi_d = _regularize(po - _regularize(pi + math.pi))
    N_TT = _uct_gaussian_pdf(phi_d, 0.0, kb.tt_aziwidth, -math.pi, math.pi)
    S_TT = (1.0 - F_R) * c_tt * (M_TT * N_TT)[:, None]
    c = torch.cos(to)[:, None]
    return S_R * c, S_TT * c


def integrals(ti, pi):
    te = torch.linspace(-math.pi / 2, math.pi / 2, N_T + 1, dtype=torch.float64, device=dev)
    tc = 0.5 * (te[1:] + te[:-1])
    pc = -math.pi + (torch.arange(N_P, dtype=torch.float64, device=dev) + 0.5) * (2 * math.pi / N_P)
    dA_t = (torch.sin(te[1:]) - torch.sin(te[:-1])) * (2 * math.pi / N_P)
    tot_r = torch.zeros(3, dtype=torch.float64, device=dev)
    tot_t = torch.zeros(3, dtype=torch.float64, device=dev)
    for ch in torch.arange(N_T, device=dev).split(400):
        T, P = torch.meshgrid(tc[ch], pc, indexing='ij')
        r, t = eval_parts(ti, pi, T.reshape(-1), P.reshape(-1))
        w = dA_t[ch][:, None].expand(-1, N_P).reshape(-1, 1)
        tot_r += (r * w).sum(0)
        tot_t += (t * w).sum(0)
    return tot_r.cpu().numpy(), tot_t.cpu().numpy()


def yarn_angles(v):
    return torch.asin(v[:, 0].clamp(-1, 1)), torch.atan2(v[:, 2], v[:, 1])


def to_np(v):
    return np.stack([np.array(v[0]), np.array(v[1]), np.array(v[2])], -1).astype(np.float64)


def run(ti, pi, n):
    wi_vec = (math.sin(ti), math.cos(ti) * math.cos(pi), math.cos(ti) * math.sin(pi))
    acc = {k: [] for k in ('w_plugin', 'w_R', 'w_TT', 'w_R_inv', 'w_TT_inv', 'w_R_base', 'w_TT_base')}
    stats = {'out_of_chart': 0, 'neg_weight': 0, 'zero_pdf': 0}
    ratios = []
    for b in range(0, n, ARGS.batch):
        m = min(ARGS.batch, n - b)
        # (1) the plugin itself, exactly as the renderer calls it
        si = dr.zeros(mi.SurfaceInteraction3f, m)
        si.wi = mi.Vector3f(*[np.full(m, c, dtype=np.float32) for c in wi_vec])
        u = np.random.default_rng(b).random((m, 3), dtype=np.float32)
        bs, w = plugin.sample(mi.BSDFContext(), si, mi.Float(u[:, 0]), mi.Point2f(u[:, 1], u[:, 2]), True)
        wp = to_np(w)
        wp[~np.isfinite(wp)] = 0.0
        acc['w_plugin'].append(wp)
        stats['neg_weight'] += int((wp < 0).any(1).sum())

        # (2) same sampler called directly, to see raw angles and both PDFs
        wi_in = torch.tensor([[ti, pi]], dtype=torch.float32, device=dev).expand(m, 2).contiguous()
        wo_raw, pdf_fwd = bm.network_sampling_spherical(D_base, D_sample, wi_in, T=8)
        pdf_fwd = pdf_fwd.detach()
        stats['out_of_chart'] += int((wo_raw[:, 0].abs() > math.pi / 2).sum())
        v = bm.yarn_spher_to_cart(wo_raw)
        to, po = yarn_angles(v)                                   # canonical angles of the same direction
        cos_raw = torch.cos(wo_raw[:, 0]).abs().clamp(min=1e-6)
        pdf_sa_fwd = pdf_fwd / cos_raw                            # solid-angle pdf, as bs.pdf
        wo_can = torch.stack([to, po], 1)
        pdf_inv = bm.network_pdf_spherical(D_base, D_sample, wo_can, wi_in, T=8) / torch.cos(to).abs().clamp(min=1e-6)
        # base-distribution normalisation: correct Gaussian uses sigma' = exp(log_scale) + eps
        with torch.no_grad():
            _, log_scale, _, _ = D_base.get_param(wi_in[:1])
            s = torch.exp(log_scale).item()
        base_fix = s / (s + D_base.eps)                           # p_correct / p_used (constant per wi)
        r, t = eval_parts(ti, pi, to.double(), po.double())
        good = (pdf_sa_fwd > 0) & torch.isfinite(pdf_sa_fwd)
        stats['zero_pdf'] += int((~good).sum())
        inv = 1.0 / pdf_sa_fwd.double().clamp(min=1e-30)
        inv = torch.where(good, inv, torch.zeros_like(inv))[:, None]
        acc['w_R'].append((r * inv).cpu().numpy())
        acc['w_TT'].append((t * inv).cpu().numpy())
        acc['w_R_base'].append((r * inv / base_fix).cpu().numpy())
        acc['w_TT_base'].append((t * inv / base_fix).cpu().numpy())
        ginv = (pdf_inv > 0) & torch.isfinite(pdf_inv)
        inv2 = torch.where(ginv, 1.0 / pdf_inv.double().clamp(min=1e-30),
                           torch.zeros(m, dtype=torch.float64, device=dev))[:, None]
        acc['w_R_inv'].append((r * inv2).cpu().numpy())
        acc['w_TT_inv'].append((t * inv2).cpu().numpy())
        ok = good & ginv
        ratios.append((pdf_inv[ok] / pdf_sa_fwd[ok]).double().cpu().numpy())
        dr.flush_malloc_cache()
    out = {}
    for k, lst in acc.items():
        a = np.concatenate(lst)
        out[k] = (a.mean(0), a.std(0) / math.sqrt(n))
    rr = np.concatenate(ratios)
    return out, stats, rr, base_fix


def main():
    rows = []
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    for t in THETA_I:
        for p in PHI_I:
            ti = math.radians(t)
            ref_r, ref_t = integrals(ti, p)
            est, st, rr, base_fix = run(ti, p, ARGS.n)
            ref = ref_r + ref_t
            row = {'theta_i_deg': t, 'phi_i_rad': p,
                   'int_R_Y': ref_r @ LUM, 'int_TT_Y': ref_t @ LUM,
                   'ratio_R_Y': (est['w_R'][0] @ LUM) / (ref_r @ LUM),
                   'ratio_TT_Y': (est['w_TT'][0] @ LUM) / (ref_t @ LUM),
                   'ratio_total_Y': ((est['w_R'][0] + est['w_TT'][0]) @ LUM) / (ref @ LUM),
                   'ratio_plugin_Y': (est['w_plugin'][0] @ LUM) / (ref @ LUM),
                   'ratio_plugin_B': est['w_plugin'][0][2] / ref[2],
                   'ratio_plugin_R': est['w_plugin'][0][0] / ref[0],
                   'z_total_Y': (((est['w_R'][0] + est['w_TT'][0]) - ref) @ LUM)
                                / math.sqrt(((est['w_R'][1] ** 2 + est['w_TT'][1] ** 2) @ LUM ** 2)),
                   'ratio_total_Y_basefix': ((est['w_R_base'][0] + est['w_TT_base'][0]) @ LUM) / (ref @ LUM),
                   'ratio_total_Y_invpdf': ((est['w_R_inv'][0] + est['w_TT_inv'][0]) @ LUM) / (ref @ LUM),
                   'base_pdf_factor': base_fix,
                   'pdf_inv/fwd_median': float(np.median(rr)),
                   'pdf_inv/fwd_p05': float(np.quantile(rr, 0.05)),
                   'pdf_inv/fwd_p95': float(np.quantile(rr, 0.95)),
                   'out_of_chart_frac': st['out_of_chart'] / ARGS.n,
                   'neg_weight_frac': st['neg_weight'] / ARGS.n,
                   'zero_pdf_frac': st['zero_pdf'] / ARGS.n}
            rows.append(row)
            print(f"theta_i {t:+5.0f}  phi_i {p:+.1f}:  R {row['ratio_R_Y']:.4f}  TT {row['ratio_TT_Y']:.4f}  "
                  f"total {row['ratio_total_Y']:.4f} (z {row['z_total_Y']:+.1f})  plugin {row['ratio_plugin_Y']:.4f}  "
                  f"inv/fwd med {row['pdf_inv/fwd_median']:.3f}  out-of-chart {row['out_of_chart_frac']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'results', f's2_single_bounce{TAG}.csv'), index=False)
    pd.set_option('display.width', 250)
    print(df.to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
