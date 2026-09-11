# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from tqdm import tqdm
import torch
from utils.model import *

import os
import sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from utils.mitsuba_brdf_draw import *
from utils.analytical_brdf_torch import *
from utils.mlp_brdf_sampling import *

torch.set_default_dtype(torch.float32)
_mi_variant = None
for _variant in ("cuda_ad_rgb", "llvm_ad_rgb"):
    try:
        mi.set_variant(_variant)
        # Load a minimal scene with a path integrator to trigger OptiX init check
        mi.load_dict({
            "type": "scene",
            "integrator": {"type": "path"},
            "sensor": {
                "type": "perspective",
                "film": {"type": "hdrfilm", "width": 1, "height": 1},
                "sampler": {"type": "independent"},
            },
        })
        _mi_variant = _variant
        print(f"Mitsuba variant: {_variant}")
        break
    except Exception as _exc:
        print(f"Mitsuba variant {_variant} failed: {_exc}", file=sys.stderr)

# With llvm_ad_rgb, DrJIT's multi-threaded LLVM pool calling PyTorch CUDA from
# worker threads causes nanothread assertion failures. set_thread_count(1) makes
# callbacks happen on a single thread, making CUDA PyTorch safe to use.
if _mi_variant != "cuda_ad_rgb":
    dr.set_thread_count(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Propagate device choice to mlp_brdf_sampling so its internal tensors match
import utils.mlp_brdf_sampling as _mlp_mod
_mlp_mod._torch_device = device

dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)


parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, default="scene_bsdf.xml")
parser.add_argument("--passes", type=int, default=128)

parser = parser.parse_args()

from utils.bsdf_dict import *
from utils.mitsuba_brdf_yarn import khungurn_bsdf


def to_mi_float(tensor):
    # mi.Float(cuda_tensor) only works with cuda_ad_rgb.
    # Using .cpu().numpy() works for both cuda_ad_rgb and llvm_ad_rgb.
    return mi.Float(tensor.detach().cpu().numpy())


# ---- 表面BSDF用の座標変換（法線 = z 軸） ----

def sph_to_dir(theta, phi):
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)


def cart_to_spher(xyz):
    r = torch.norm(xyz, dim=1)
    theta = torch.acos(xyz[:,2]/(r+1e-8))
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([theta, phi], dim=1)


# ---- ヤーン繊維BSDF用の座標変換（繊維軸 = x 軸） ----
# 【表面BSDFとの違い】
# 表面BSDF: theta = acos(z),  phi = atan2(y, x)  → theta ∈ [0, π]
# ヤーンBSDF: theta = asin(x), phi = atan2(z, y)  → theta ∈ [-π/2, π/2]

def cart_to_yarn_spher(xyz):
    """3D直交座標 → ヤーン繊維球面座標 (theta, phi)"""
    theta = torch.asin(xyz[:,0].clamp(-1 + 1e-6, 1 - 1e-6))  # 縦方向角
    phi   = torch.atan2(xyz[:,2], xyz[:,1])                    # 方位角
    return torch.stack([theta, phi], dim=1)


def yarn_to_dir_mi(wo_tp):
    """ヤーン繊維球面座標 (theta, phi) → Mitsuba Vector3f
    (x, y, z) = (sin(θ), cos(θ)cos(φ), cos(θ)sin(φ))
    """
    theta = wo_tp[:, 0]
    phi   = wo_tp[:, 1]
    x = torch.sin(theta)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.cos(theta) * torch.sin(phi)
    return mi.Vector3f(to_mi_float(x), to_mi_float(y), to_mi_float(z))
class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.idx = props["idx"]
        self.albedo = mi.Color3f(props["albedo"])
        self.bsdf = bsdf_materials[self.idx]
        
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self.D_sample = NN_cond_pos(input_dim=6,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to(device)
        self.D_sample.load_state_dict(torch.load(
            os.path.join(_script_dir, f"checkpoints_new/bsdf_{self.idx}_spherical/brdf_rectify_network{self.idx}.pth"),
            map_location=device))
        self.D_sample.eval()

        self.D_base = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16).to(device)
        self.D_base.load_state_dict(torch.load(
            os.path.join(_script_dir, f"checkpoints_new/bsdf_{self.idx}_spherical/brdf_pretrain_network{self.idx}.pth"),
            map_location=device))
        
        reflection_flags = mi.BSDFFlags.Diffuse | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        # .torch() returns CPU tensor with llvm_ad_rgb; move to device for network
        wi = si.wi.torch().to(device)
        wi_input = cart_to_spher(wi)

        wo,pdf = network_sampling_spherical(self.D_base,self.D_sample,wi_input,T=8)
        pdf = torch.where(torch.sin(wo[:,0]) > 0.00005, pdf, torch.zeros_like(pdf))

        wo = mi.Vector2f(to_mi_float(wo[...,0]), to_mi_float(wo[...,1]))
        wo = sph_to_dir(wo.x, wo.y)

        bs = mi.BSDFSample3f()

        bs.wo = wo

        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))

        invsin_theta_o =dr.clamp(1/ (dr.abs(mi.Frame3f.sin_theta(bs.wo))) ,1,floatmax)
        if dr.any_nested(invsin_theta_o<0):
            print("invsin_theta_o<0")
        bs.pdf = to_mi_float(pdf) * invsin_theta_o
        bs.sampled_component = 2
        bs.eta = dr.select((mi.Frame3f.cos_theta(bs.wo) > 0.0), 1.0, 1.788)
        bs.sampled_type = dr.select((mi.Frame3f.cos_theta(bs.wo) > 0.0), 8, 16)

        brdf = self.bsdf.eval(ctx, si, bs.wo)
        value = brdf * self.albedo / to_mi_float(pdf) * mi.Frame3f.sin_theta(bs.wo)
        value = dr.select((bs.pdf > 0.0), value, mi.Vector3f(0))
        pdf = bs.pdf.torch().to(device)
        value_torch = value.torch()[:,0].to(device)
        pdf = torch.where(value_torch<3.5, pdf, torch.zeros_like(pdf))
        bs.pdf = to_mi_float(pdf)
        return (bs, dr.select((bs.pdf > 0.0) , value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        
        brdf = self.bsdf.eval(ctx, si, wo)
        value =  brdf * self.albedo 
        return value

    def pdf(self, ctx, si, wo, active=True):

        wi = si.wi.torch().to(device)
        wi_input = cart_to_spher(wi)
        # Use separate variable to avoid shadowing the Mitsuba Vector3f `wo`
        wo_torch = wo.torch().to(device)
        wo_input = cart_to_spher(wo_torch)
        pdf = network_pdf_spherical(self.D_base,self.D_sample,wo_input,wi_input,T=8)
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        invsin_theta_o =dr.clamp(1/ (dr.abs(mi.Frame3f.sin_theta(wo))) ,1,floatmax)
        pdf = to_mi_float(pdf) * invsin_theta_o
        # # pdf = self.bsdf.pdf(ctx, si, wo)
        # value =  self.bsdf.eval(ctx, si, wo) / mi.Float(pdf) 
        # # print("value1: ", value1)
        # pdf = pdf.torch()
        # value_torch = value.torch()[:,0]
        # pdf = torch.where(value_torch<30, pdf, torch.zeros_like(pdf))
        # pdf = mi.Float(pdf) 
        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


class MyYarnBSDF(mi.BSDF):
    """拡散サンプリングで学習したヤーン繊維BSDFのMitsuba BSDFプラグイン。

    【MyBSDFとの主な違い】

    1. 座標系: 繊維軸 = x 軸（theta ∈ [-π/2, π/2]）
       MyBSDF   は法線 = z 軸（theta ∈ [0, π]）

    2. 入射方向の制限なし:
       MyBSDF   は `active &= cos_theta_i > 0` で上半球のみ許可
       MyYarnBSDF はどの方向からの入射も処理する（R + TT 両ローブ）

    3. サンプリング重みのヤコビアン:
       表面BSDF は dω = sin(θ)dθdφ → 重みに sin(θo) を掛ける
       ヤーンBSDF は dω = cos(θ)dθdφ → khungurn_bsdf.eval が既に cos(θo) を含むため
                    追加の Jacobian は不要（重みは brdf_yarn / pdf のみ）

    4. チェックポイントフォルダ: checkpoints_new/bsdf_{idx}_yarn/
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.idx = props["idx"]
        self.albedo = mi.Color3f(props["albedo"])
        self.bsdf = bsdf_materials[self.idx]  # khungurn_bsdf インスタンス

        _script_dir = os.path.dirname(os.path.abspath(__file__))
        # ヤーン素材のチェックポイントは learning_repo_cleanup 側に保存されている
        ckpt_dir = os.path.join(_script_dir, "..", "learning_repo_cleanup",
                                "checkpoints_new", f"bsdf_{self.idx}_yarn")

        self.D_sample = NN_cond_pos(
            input_dim=6, output_dim=2, N_NEURONS=32,
            POSITIONAL_ENCODING_BASIS_NUM=5
        ).to(device)
        self.D_sample.load_state_dict(torch.load(
            os.path.join(ckpt_dir, f"brdf_rectify_network{self.idx}.pth"),
            map_location=device,
        ))
        self.D_sample.eval()

        self.D_base = NN_cond_pretrain_spherical_one(
            input_dim=2, N_NEURONS=16
        ).to(device)
        self.D_base.load_state_dict(torch.load(
            os.path.join(ckpt_dir, f"brdf_pretrain_network{self.idx}.pth"),
            map_location=device,
        ))

        # ヤーン繊維は Glossy かつ両面（反射 + 透過）
        flags = (mi.BSDFFlags.Glossy | mi.BSDFFlags.FrontSide
                 | mi.BSDFFlags.BackSide | mi.BSDFFlags.Anisotropic)
        self.m_components = [flags]
        self.m_flags = flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        # ---- 入射方向をヤーン球面座標に変換 ----
        wi = si.wi.torch().to(device)
        wi_input = cart_to_yarn_spher(wi)  # (N, 2): (theta_i, phi_i)

        # ---- 学習済みネットワークで出射方向をサンプリング ----
        wo, pdf = network_sampling_spherical(
            self.D_base, self.D_sample, wi_input, T=8
        )
        # wo: (N, 2): (theta_o, phi_o) in yarn coords

        # cos(θ_fiber) = ヤーン座標の Jacobian（dω = cos(θ)dθdφ）
        cos_theta_fiber = torch.cos(wo[:, 0]).abs()

        # cos(θ_surface) = サーフェス法線との内積 = z成分 = cos(θ_fiber) × sin(φ_fiber)
        # yarn_to_dir_mi が返す (x,y,z) = (sin θ, cos θ cos φ, cos θ sin φ) なので z = cos_fiber × sin_phi
        cos_theta_surface = torch.cos(wo[:, 0]) * torch.sin(wo[:, 1])

        # 有効条件: 極点でなく（Jacobian が 0 でない）かつサーフェス上半球（z > 0）
        valid = (cos_theta_fiber > 0.00005) & (cos_theta_surface > 0)
        pdf = torch.where(valid, pdf, torch.zeros_like(pdf))

        # ---- ヤーン球面座標 → Mitsuba 3D 方向 ----
        bs = mi.BSDFSample3f()
        bs.wo = yarn_to_dir_mi(wo)

        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        # 立体角 pdf = pdf_θφ / cos_fiber（ヤーン座標 Jacobian）
        invcos_theta_fiber = dr.clamp(
            to_mi_float(1.0 / cos_theta_fiber.clamp(min=1e-6)), 1.0, floatmax
        )
        bs.pdf = to_mi_float(pdf) * invcos_theta_fiber
        bs.sampled_component = mi.UInt32(0)
        bs.eta = mi.Float(1.0)  # ヤーンは屈折なし
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        # ---- サンプリング重みの計算 ----
        # Mitsuba の path integrator は value = f(wi,wo) × cos_surface / pdf_solid を期待する。
        # khungurn_bsdf.eval() = (S_R + S_TT) × cos_fiber（= brdf_yarn）なので:
        #
        #   value = f × cos_surface / pdf_solid
        #         = f × cos_surface / (pdf_θφ / cos_fiber)
        #         = f × cos_fiber × cos_surface / pdf_θφ
        #         = brdf_yarn × cos_surface / pdf_θφ
        #
        # 従来は cos_surface を掛け忘れており、phi ≈ 0,π の方向で value → ∞ に
        # なって firefly の原因になっていた。
        cos_surface_pos = cos_theta_surface.clamp(min=0)
        brdf = self.bsdf.eval(wi_input, wo)  # (N,) PyTorch tensor
        safe_pdf = pdf.clamp(min=1e-8)
        value = mi.Color3f(to_mi_float(brdf * cos_surface_pos / safe_pdf)) * self.albedo
        value = dr.select(bs.pdf > 0.0, value, mi.Color3f(0))

        # 過大な値をクリップ（数値的な外れ値を除去）
        pdf_torch    = bs.pdf.torch().to(device)
        value_scalar = value.torch()[:, 0].to(device)
        pdf_torch = torch.where(value_scalar < 3.5, pdf_torch,
                                torch.zeros_like(pdf_torch))
        bs.pdf = to_mi_float(pdf_torch)

        return (bs, dr.select(bs.pdf > 0.0, value, mi.Color3f(0)))

    def eval(self, ctx, si, wo, active=True):
        # ---- 入出射方向をヤーン球面座標に変換して BSDF 値を評価 ----
        # Mitsuba の規約: eval() は f(wi,wo) × |cos_surface| を返す。
        # khungurn.eval() = f × cos_fiber なので、
        #   f × cos_surface = khungurn.eval() × sin(φ_fiber)
        # （cos_surface = cos_fiber × sin_phi_fiber = khungurn.eval()/f × sin_phi = ... = brdf_yarn × sin_phi / cos_fiber × cos_fiber = brdf_yarn × sin_phi）
        # ただし cos_fiber で割ることは数値不安定なため、直接 brdf_yarn × sin_phi で計算。
        wi_t = cart_to_yarn_spher(si.wi.torch().to(device))
        wo_t = cart_to_yarn_spher(wo.torch().to(device))
        brdf_yarn = self.bsdf.eval(wi_t, wo_t)  # f × cos_fiber
        sin_phi = torch.sin(wo_t[:, 1])          # sin(φ_fiber); cos_surface = cos_fiber × sin_phi
        brdf_surface = (brdf_yarn * sin_phi).clamp(min=0)  # f × cos_surface
        return mi.Color3f(to_mi_float(brdf_surface)) * self.albedo

    def pdf(self, ctx, si, wo, active=True):
        wi_t = cart_to_yarn_spher(si.wi.torch().to(device))
        wo_t = cart_to_yarn_spher(wo.torch().to(device))
        pdf = network_pdf_spherical(
            self.D_base, self.D_sample, wo_t, wi_t, T=8
        )
        # (θ,φ) 空間の pdf → 立体角 pdf へ変換
        cos_theta_o = torch.cos(wo_t[:, 0]).abs().clamp(min=1e-6)
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        invcos_theta_o = dr.clamp(
            to_mi_float(1.0 / cos_theta_o), 1.0, floatmax
        )
        return to_mi_float(pdf) * invcos_theta_o

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return "MyYarnBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    import time

    start_time = time.time()
    mi.register_bsdf("mybsdf",     lambda props: MyBSDF(props))
    mi.register_bsdf("myyarnbsdf", lambda props: MyYarnBSDF(props))

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_file = parser.scene_file
    if not scene_file.lower().endswith(".xml"):
        scene_file += ".xml"
    scene_path = os.path.join(_script_dir, "matpreview", scene_file)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file does not exist: {scene_path}")
    scene = mi.load_file(scene_path)

    SPP = 4
    spp = SPP * parser.passes

    seed = 0
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1

    output_dir = os.path.join(_script_dir, "diffusion_bsdf_myresult")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{parser.scene_file}.png")
    mi.util.write_bitmap(filepath, image, spp)
    filepath = os.path.join(output_dir, f"{parser.scene_file}.exr")
    mi.util.write_bitmap(filepath, image, spp)