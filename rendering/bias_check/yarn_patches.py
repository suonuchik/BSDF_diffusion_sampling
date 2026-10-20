"""Candidate fixes for the brightness bias of MyYarnBSDF, applied without editing the original files.

fix_base  : the base distribution NN_cond_pretrain_spherical_one samples theta with
            sigma' = exp(log_scale) + eps, but log_prob normalises the Gaussian with exp(log_scale).
            The PDF is then too large by sigma'/exp(log_scale) and every weight too small.
fix_chart : the ODE can move theta outside [-pi/2, pi/2].  The direction is valid, but
            - sample() evaluates the BSDF at the raw (out-of-range) angles -> wrong value, even negative,
            - the density of a direction is the sum over BOTH chart points that map to it,
              (theta, phi) and (sign(theta)*pi - theta, phi + pi); the plugin counts only one.
            The patched BSDF evaluates at the canonical angles and sums both chart densities
            (forward PDF for the sampled chart point, inverse-ODE PDF for the other one).
"""
import math
import sys

import numpy as np
import torch


def apply_base_fix(bm):
    cls = bm.NN_cond_pretrain_spherical_one

    def log_prob(self, x, x_co):
        loc, log_scale, loc_von, concentration = self.get_param(x_co)
        sigma = torch.exp(log_scale) + self.eps                     # the sigma that sample() uses
        eps = (x[:, 0:1] - loc) / sigma
        loggau = -0.5 * np.log(2 * torch.pi) - torch.sum(torch.log(sigma), dim=1) - 0.5 * torch.sum(eps ** 2, dim=1)
        logvon = torch.distributions.von_mises.VonMises(loc_von.flatten(), concentration.flatten()).log_prob(x[:, 1])
        logp = loggau + logvon
        self.previous = logp
        return logp

    cls.log_prob = log_prob
    print('yarn_patches: base-distribution log_prob normalisation fixed')


def apply_eval_fix():
    """The PyTorch khungurn_bsdf (utils/mitsuba_brdf_yarn.py) reproduces the original gaussian_cdf bug in N_TT:
    the uniform tail weight becomes (1 - (Phi(pi) - Phi(-pi))) / 2pi = 0.00168 / 2pi instead of ~0, so
    int N_TT dphi = 1.0017 and every TT scattering carries 0.17 % too much energy (the corrected reference does not).
    Replace the module function with the corrected one (uses mu and sigma)."""
    mod = sys.modules['utils.mitsuba_brdf_yarn']

    def _uct_gaussian_pdf(x, mu, sigma, a, b):
        mu_t = torch.as_tensor(mu, dtype=torch.float64)
        mass = 0.5 * (torch.erf((b - mu_t) / (sigma * math.sqrt(2.0))) - torch.erf((a - mu_t) / (sigma * math.sqrt(2.0))))
        return mod._gaussian_pdf(x, mu, sigma) + float((1.0 - mass) / (b - a))

    mod._uct_gaussian_pdf = _uct_gaussian_pdf
    print('yarn_patches: N_TT normalisation in khungurn_bsdf.eval fixed')


def _canonical(wo_raw):
    """raw chart point (theta may exceed pi/2) -> canonical (theta, phi) of the same direction, and the other chart point."""
    th, ph = wo_raw[:, 0], wo_raw[:, 1]
    x = torch.sin(th)
    y = torch.cos(th) * torch.cos(ph)
    z = torch.cos(th) * torch.sin(ph)
    tc = torch.asin(x.clamp(-1, 1))
    pc = torch.atan2(z, y)
    return torch.stack([tc, pc], 1)


def _other_chart(wo_can):
    tc, pc = wo_can[:, 0], wo_can[:, 1]
    to = torch.where(tc >= 0, math.pi - tc, -math.pi - tc)
    return torch.stack([to, pc + math.pi], 1)


def make_patched_bsdf(bm, fix_chart, fix_mis=False):
    """fix_mis (needs fix_chart): report the inverse-ODE PDF in bs.pdf, i.e. the same function that pdf() returns
    for emitter samples, so the two MIS weights of a direction sum to one.  The sample weight (value) keeps the
    exact forward PDF, so it stays unbiased; bs.pdf is only used by the integrator for MIS."""
    assert fix_chart or not fix_mis, 'fix_mis is implemented on top of fix_chart'
    mi, dr = bm.mi, bm.dr
    device = bm.device

    class PatchedYarnBSDF(bm.MyYarnBSDF):
        def _pdf_theta_phi(self, wo_can, wi_input):
            p = bm.network_pdf_spherical(self.D_base, self.D_sample, wo_can, wi_input, T=8)
            if fix_chart:
                p = p + bm.network_pdf_spherical(self.D_base, self.D_sample, _other_chart(wo_can), wi_input, T=8)
            return p.detach()

        def sample(self, ctx, si, sample1, sample2, active=True):
            if not fix_chart:
                return super().sample(ctx, si, sample1, sample2, active)
            rot = self._yarn_rotation(si)
            wi = self._local_to_yarn(si.wi.torch().to(device), rot)
            wi_input = bm.cart_to_yarn_spher(wi)
            wo_raw, pdf_raw = bm.network_sampling_spherical(self.D_base, self.D_sample, wi_input, T=8)
            pdf_raw = pdf_raw.detach()
            wo_can = _canonical(wo_raw)
            in_chart = wo_raw[:, 0].abs() <= math.pi / 2
            # density of the direction = sampled chart point (exact forward PDF) + the other chart point (inverse ODE)
            other = torch.where(in_chart[:, None], _other_chart(wo_can), wo_can)
            pdf = pdf_raw + bm.network_pdf_spherical(self.D_base, self.D_sample, other, wi_input, T=8).detach()
            cos_c = torch.cos(wo_can[:, 0]).clamp(min=0.0)
            valid = (cos_c > 5e-5) & torch.isfinite(pdf) & (pdf > 0)
            pdf = torch.where(valid, pdf, torch.zeros_like(pdf))

            bs = mi.BSDFSample3f()
            bs.wo = bm.to_mi_vector(self._yarn_to_local(bm.yarn_spher_to_cart(wo_can), rot))
            floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
            pdf_mis = pdf
            if fix_mis:
                pdf_mis = self._pdf_theta_phi(wo_can, wi_input)
                pdf_mis = torch.where(valid & torch.isfinite(pdf_mis) & (pdf_mis > 0), pdf_mis, torch.zeros_like(pdf_mis))
                valid = valid & (pdf_mis > 0)
                pdf = torch.where(valid, pdf, torch.zeros_like(pdf))
            bs.pdf = bm.to_mi_float(pdf_mis) * dr.clamp(bm.to_mi_float(1.0 / cos_c.clamp(min=1e-6)), 1.0, floatmax)
            bs.sampled_component = mi.UInt32(0)
            bs.eta = mi.Float(1.0)
            is_refl = bm.to_mi_float((torch.sin(wo_can[:, 1]) > 0).float()) > 0.5
            bs.sampled_type = dr.select(is_refl, mi.UInt32(+mi.BSDFFlags.GlossyReflection),
                                        mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
            brdf = self.bsdf.eval_rgb(wi_input, wo_can)                  # canonical angles: cos(theta) >= 0
            value = brdf * (cos_c / pdf.clamp(min=1e-8)).unsqueeze(1)
            value = torch.where(valid[:, None], value, torch.zeros_like(value))
            value = bm.to_mi_color(value) * self.albedo
            return bs, dr.select(bs.pdf > 0.0, value, mi.Color3f(0))

        def pdf(self, ctx, si, wo, active=True):
            if not fix_chart:
                return super().pdf(ctx, si, wo, active)
            rot = self._yarn_rotation(si)
            wi_t = bm.cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
            wo_t = bm.cart_to_yarn_spher(self._local_to_yarn(wo.torch().to(device), rot))
            p = self._pdf_theta_phi(wo_t, wi_t)
            cos_o = torch.cos(wo_t[:, 0]).abs().clamp(min=1e-6)
            floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
            return bm.to_mi_float(p) * dr.clamp(bm.to_mi_float(1.0 / cos_o), 1.0, floatmax)

        def to_string(self):
            return "PatchedYarnBSDF[\n    fix_chart=%s,\n    fix_mis=%s,\n]" % (fix_chart, fix_mis)

    return PatchedYarnBSDF


# ---------------------------------------------------------------------------------------------------------------
# Analytic sampler: the corrected khungurn sampler (neuralyarn/analysis/khungurn_bias/khungurn_fixed.py) ported to
# torch and run inside exactly the same pip-Mitsuba / reconstructed-frame pipeline as the NN sampler.  If this render
# matches the corrected reference, the pipeline itself is unbiased and any remaining difference comes from the NN.
# ---------------------------------------------------------------------------------------------------------------

def _gauss_pdf(x, mu, s):
    return torch.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def _std_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _uct_pdf(x, mu, s, a, b):
    """density of: Gaussian(mu, s) sample, replaced by Uniform(a, b) when it falls outside (a, b)  (fixed CDF)"""
    mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    inside_mass = _std_cdf((b - mu) / s) - _std_cdf((a - mu) / s)
    return _gauss_pdf(x, mu, s) + (1.0 - inside_mass) / (b - a)


def _uct_sample(u_g, u_u, mu, s, a, b):
    g = mu + s * math.sqrt(2.0) * torch.erfinv((2.0 * u_g - 1.0).clamp(-1 + 1e-7, 1 - 1e-7))
    return torch.where((g > a) & (g < b), g, a + u_u * (b - a))


def _regularize(a):
    a = torch.remainder(a, 2 * math.pi)
    return torch.where(a > math.pi, a - 2 * math.pi, a)


def make_analytic_bsdf(bm, original_eval=False):
    """original_eval=True: evaluate the BSDF with khungurn_bsdf.eval_rgb (the function the NN plugin uses, incl. the
    N_TT normalisation bug) instead of the corrected formula; the sampler and PDF stay exact."""
    mi, dr = bm.mi, bm.dr
    device = bm.device

    class AnalyticYarnBSDF(bm.MyYarnBSDF):
        def _params(self, theta_i):
            kb = self.bsdf
            c_r = torch.tensor(kb.c_r_rgb, dtype=torch.float32, device=device)
            c_tt = torch.tensor(kb.c_tt_rgb, dtype=torch.float32, device=device)
            F_R = c_r + (1 - c_r) * ((1 - torch.abs(torch.cos(theta_i))) ** 5)[:, None]   # (N,3)
            att_R, att_TT = F_R, (1 - F_R) * c_tt
            att = att_R + att_TT
            ratio = att / att.sum(1, keepdim=True)
            p_R = (ratio * att_R / att).sum(1)
            return F_R, c_tt, p_R

        def _pdf_tp(self, ti, pi, to, po):
            kb = self.bsdf
            _, _, p_R = self._params(ti)
            phi_d = _regularize(po - _regularize(pi + math.pi))
            h = math.pi / 2
            pr = _uct_pdf(to, -ti, kb.r_longwidth, -h, h) / (2 * math.pi)
            pt = _uct_pdf(to, -ti, kb.tt_longwidth, -h, h) * _uct_pdf(phi_d, 0.0, kb.tt_aziwidth, -math.pi, math.pi)
            return p_R * pr + (1 - p_R) * pt

        def _eval_tp(self, ti, pi, to, po):
            kb = self.bsdf
            if original_eval:
                return kb.eval_rgb(torch.stack([ti, pi], 1), torch.stack([to, po], 1))
            F_R, c_tt, _ = self._params(ti)
            M_R = bm_cos2(to, -ti, kb.r_longwidth)
            M_TT = bm_cos2(to, -ti, kb.tt_longwidth)
            phi_d = _regularize(po - _regularize(pi + math.pi))
            N_TT = _uct_pdf(phi_d, 0.0, kb.tt_aziwidth, -math.pi, math.pi)
            S = F_R * (M_R / (2 * math.pi))[:, None] + (1 - F_R) * c_tt * (M_TT * N_TT)[:, None]
            return S * torch.cos(to)[:, None]

        def sample(self, ctx, si, sample1, sample2, active=True):
            kb = self.bsdf
            rot = self._yarn_rotation(si)
            wi_t = bm.cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
            ti, pi = wi_t[:, 0], wi_t[:, 1]
            n = ti.shape[0]
            u = torch.rand(n, 5, device=device)
            _, _, p_R = self._params(ti)
            sel_R = u[:, 0] < p_R
            h = math.pi / 2
            to = torch.where(sel_R, _uct_sample(u[:, 1], u[:, 2], -ti, kb.r_longwidth, -h, h),
                             _uct_sample(u[:, 1], u[:, 2], -ti, kb.tt_longwidth, -h, h))
            phi_d = torch.where(sel_R, -math.pi + 2 * math.pi * u[:, 3],
                                _uct_sample(u[:, 3], u[:, 4], torch.zeros_like(ti), kb.tt_aziwidth, -math.pi, math.pi))
            po = _regularize(pi + math.pi + phi_d)
            pdf = self._pdf_tp(ti, pi, to, po)
            cos_o = torch.cos(to)
            valid = (cos_o > 5e-5) & (pdf > 0)
            ev = self._eval_tp(ti, pi, to, po)
            value = torch.where(valid[:, None], ev * (cos_o / pdf.clamp(min=1e-12))[:, None], torch.zeros_like(ev))
            bs = mi.BSDFSample3f()
            wo_cart = bm.yarn_spher_to_cart(torch.stack([to, po], 1))
            bs.wo = bm.to_mi_vector(self._yarn_to_local(wo_cart, rot))
            bs.pdf = bm.to_mi_float(torch.where(valid, pdf / cos_o.clamp(min=1e-6), torch.zeros_like(pdf)))
            bs.sampled_component = mi.UInt32(0)
            bs.eta = mi.Float(1.0)
            is_refl = bm.to_mi_float((torch.sin(po) > 0).float()) > 0.5
            bs.sampled_type = dr.select(is_refl, mi.UInt32(+mi.BSDFFlags.GlossyReflection),
                                        mi.UInt32(+mi.BSDFFlags.GlossyTransmission))
            return bs, dr.select(bs.pdf > 0.0, bm.to_mi_color(value) * self.albedo, mi.Color3f(0))

        def eval(self, ctx, si, wo, active=True):
            rot = self._yarn_rotation(si)
            wi_t = bm.cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
            wo_t = bm.cart_to_yarn_spher(self._local_to_yarn(wo.torch().to(device), rot))
            return bm.to_mi_color(self._eval_tp(wi_t[:, 0], wi_t[:, 1], wo_t[:, 0], wo_t[:, 1])) * self.albedo

        def pdf(self, ctx, si, wo, active=True):
            rot = self._yarn_rotation(si)
            wi_t = bm.cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
            wo_t = bm.cart_to_yarn_spher(self._local_to_yarn(wo.torch().to(device), rot))
            p = self._pdf_tp(wi_t[:, 0], wi_t[:, 1], wo_t[:, 0], wo_t[:, 1])
            return bm.to_mi_float(p / torch.cos(wo_t[:, 0]).abs().clamp(min=1e-6))

        def to_string(self):
            return "AnalyticYarnBSDF[]"

    from utils.mitsuba_brdf_yarn import _cos2_normalized_gaussian as bm_cos2  # noqa: E402
    return AnalyticYarnBSDF
