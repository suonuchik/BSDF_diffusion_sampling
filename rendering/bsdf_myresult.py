# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from tqdm import tqdm
import torch
from utils.model import *

import math
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


def to_mi_color(tensor):
    """(N, 3) PyTorch tensor → mi.Color3f"""
    t = tensor.t().contiguous()  # (3, N): mi.Float には連続メモリを渡す
    return mi.Color3f(to_mi_float(t[0]), to_mi_float(t[1]), to_mi_float(t[2]))


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


def yarn_spher_to_cart(wo_tp):
    """ヤーン繊維球面座標 (theta, phi) → 3D直交座標 (N, 3)
    (x, y, z) = (sin(θ), cos(θ)cos(φ), cos(θ)sin(φ))
    """
    theta = wo_tp[:, 0]
    phi   = wo_tp[:, 1]
    return torch.stack([torch.sin(theta),
                        torch.cos(theta) * torch.cos(phi),
                        torch.cos(theta) * torch.sin(phi)], dim=1)


def yarn_canonical(wo_tp):
    """ODE が出力した (theta, phi) は theta が [-π/2, π/2] の外に出ることがある。
    同じ方向を表す正規の角度（theta ∈ [-π/2, π/2]）に直す。"""
    return cart_to_yarn_spher(yarn_spher_to_cart(wo_tp))


def yarn_other_chart(wo_tp):
    """正規の角度 (theta, phi) と同じ方向を表す、もう一方のチャート上の点 (±π - theta, phi + π)。
    方向の密度は、この2点の (theta, phi) 空間の密度の和になる。"""
    theta, phi = wo_tp[:, 0], wo_tp[:, 1]
    other = torch.where(theta >= 0, math.pi - theta, -math.pi - theta)
    return torch.stack([other, phi + math.pi], dim=1)


def to_mi_vector(tensor):
    """(N, 3) PyTorch tensor → mi.Vector3f"""
    t = tensor.t().contiguous()
    return mi.Vector3f(to_mi_float(t[0]), to_mi_float(t[1]), to_mi_float(t[2]))


def attach_curve_frames(scene):
    """linearcurve 上の MyYarnBSDF に、線分ごとの繊維接線（ワールド座標）を設定する。

    【なぜ必要か】
    Khungurn BSDF はローカル座標の x 軸 = 繊維接線を前提とする。
    neuralyarn は Mitsuba の linearcurve.cpp をパッチして sh_frame.s = 接線にしているが、
    pip 版 Mitsuba は曲線で dp_du を設定しないため、sh_frame.s が法線 n から
    適当に作られた（繊維とは無関係な）軸になる。
    そこで線分の接線を BSDF に渡し、BSDF 内でパッチ版と同じヤーン座標系を再構成する。

    シーン読み込み後、レンダリング前に呼ぶこと。曲線以外の形状には何もしない。
    """
    seen = {}
    for shape in scene.shapes():
        bsdf = shape.bsdf()
        if not isinstance(bsdf, MyYarnBSDF):
            continue
        params = mi.traverse(shape)
        if "segment_indices" not in params:
            continue  # 曲線以外（球など）は sh_frame をそのまま使う
        if id(bsdf) in seen:
            raise RuntimeError("1つの MyYarnBSDF が複数の曲線形状で共有されています。"
                               "prim_index が曖昧になるため、形状ごとに BSDF を分けてください。")
        seen[id(bsdf)] = True
        cp  = torch.from_numpy(np.array(params["control_points"])).reshape(-1, 4)[:, :3]
        idx = torch.from_numpy(np.array(params["segment_indices"]).astype(np.int64))
        tangents = torch.nn.functional.normalize(cp[idx + 1] - cp[idx], dim=1)
        bsdf.set_segment_tangents(tangents)
        print(f"attach_curve_frames: {tangents.shape[0]} segments -> {bsdf.to_string().splitlines()[0]}")
class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.idx = props["idx"]
        self.albedo = mi.Color3f(props["albedo"])
        self.bsdf = bsdf_materials[self.idx]
        # 透過側の相対屈折率。素材側は principled が既定の 1.5、roughdielectric(idx=23〜25) が bk7 ≒ 1.50。
        # bs.eta としてパス積分器のロシアンルーレットにだけ使われ、BSDF の値は変えない
        self.ior = props.get("ior", 1.5)

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
        
        # 反射と透過（半透明素材）の両方を出力し、どちらの面からの入射も扱う
        refl_flags  = mi.BSDFFlags.GlossyReflection   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        trans_flags = mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [refl_flags, trans_flags]
        self.m_flags = refl_flags | trans_flags

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
        # 入射側と出射側で z の符号が逆なら透過。裏側（物体の内側）から入射した場合も正しく判定できる
        transmit = (cos_theta_i * mi.Frame3f.cos_theta(bs.wo)) < 0.0
        bs.sampled_component = dr.select(transmit, mi.UInt32(1), mi.UInt32(0))
        bs.eta = dr.select(transmit, dr.select(cos_theta_i > 0.0, self.ior, 1.0 / self.ior), 1.0)
        bs.sampled_type = dr.select(
            transmit,
            mi.UInt32(+mi.BSDFFlags.GlossyTransmission),
            mi.UInt32(+mi.BSDFFlags.GlossyReflection),
        )

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
        return "MyBSDF[\n" "    albedo=%s,\n" "    ior=%.4f,\n" "]" % (self.albedo, self.ior)


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

        # ckpt_tag: チェックポイントフォルダの末尾タグ（デフォルト "yarn"）
        # 例: ckpt_tag="yarn"       → checkpoints_new/bsdf_26_yarn/
        #     ckpt_tag="yarn_curve" → checkpoints_new/bsdf_26_yarn_curve/
        ckpt_tag = props.get("ckpt_tag", "yarn")

        _script_dir = os.path.dirname(os.path.abspath(__file__))
        # ヤーン素材のチェックポイントは learning_repo_cleanup 側に保存されている
        ckpt_dir = os.path.join(_script_dir, "..", "learning_repo_cleanup",
                                "checkpoints_new", f"bsdf_{self.idx}_{ckpt_tag}")

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

        # ヤーン繊維は Glossy かつ両面（反射 R + 透過 TT 両ローブ）
        refl_flags  = (mi.BSDFFlags.GlossyReflection  | mi.BSDFFlags.FrontSide
                       | mi.BSDFFlags.BackSide | mi.BSDFFlags.Anisotropic)
        trans_flags = (mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide
                       | mi.BSDFFlags.BackSide | mi.BSDFFlags.Anisotropic)
        self.m_components = [refl_flags, trans_flags]
        self.m_flags = refl_flags | trans_flags

        # 線分ごとの繊維接線（ワールド座標）。attach_curve_frames() で設定される。
        # None の場合は sh_frame.s を繊維軸とみなす（球などの非曲線形状用）
        self.seg_tangents = None

    def set_segment_tangents(self, tangents):
        self.seg_tangents = tangents.to(device, dtype=torch.float32)

    def _yarn_rotation(self, si):
        """シェーディングフレーム → ヤーン座標の z 軸まわり回転 (cos ψ, sin ψ) を返す。

        ヤーン座標: x = 接線を法線に垂直な面へ射影したもの, y = n × x, z = n
        （パッチ版 Mitsuba の initialize_sh_frame() と同じ定義）
        seg_tangents が無い場合は None（回転なし）。
        """
        if self.seg_tangents is None:
            return None
        prim = torch.from_numpy(np.array(si.prim_index).astype(np.int64)).to(device)
        # 無効レーンの prim_index は不定値になりうるので範囲内に収める
        prim = prim.clamp(0, self.seg_tangents.shape[0] - 1)
        t_world = self.seg_tangents[prim]
        frame = [si.sh_frame.s.torch().to(device), si.sh_frame.t.torch().to(device)]
        tx = (t_world * frame[0]).sum(1)
        ty = (t_world * frame[1]).sum(1)
        norm = torch.sqrt(tx * tx + ty * ty)
        ok = norm > 1e-6
        c = torch.where(ok, tx / norm.clamp(min=1e-6), torch.ones_like(tx))
        s = torch.where(ok, ty / norm.clamp(min=1e-6), torch.zeros_like(ty))
        return c, s

    @staticmethod
    def _local_to_yarn(v, rot):
        if rot is None:
            return v
        c, s = rot
        return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=1)

    @staticmethod
    def _yarn_to_local(v, rot):
        if rot is None:
            return v
        c, s = rot
        return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=1)

    def sample(self, ctx, si, sample1, sample2, active=True):
        # ---- 入射方向をヤーン球面座標に変換 ----
        rot = self._yarn_rotation(si)
        wi = self._local_to_yarn(si.wi.torch().to(device), rot)
        wi_input = cart_to_yarn_spher(wi)  # (N, 2): (theta_i, phi_i)

        # ---- 学習済みネットワークで出射方向をサンプリング ----
        wo_raw, pdf_raw = network_sampling_spherical(
            self.D_base, self.D_sample, wi_input, T=8
        )
        pdf_raw = pdf_raw.detach()
        # wo_raw: (N, 2) の (theta_o, phi_o)。ODE で theta が [-π/2, π/2] の外に出ることがあるので、
        # BSDF の評価と方向の計算には、同じ方向を表す正規の角度 wo を使う
        wo = yarn_canonical(wo_raw)
        in_chart = wo_raw[:, 0].abs() <= math.pi / 2

        # 方向の (θ,φ) 空間の密度 = 2つのチャート上の点の密度の和。
        # サンプルした点は順方向の厳密な pdf_raw、もう一方の点は逆 ODE で求める
        other = torch.where(in_chart[:, None], yarn_other_chart(wo), wo)
        pdf = pdf_raw + network_pdf_spherical(self.D_base, self.D_sample, other, wi_input, T=8).detach()

        # cos(θ_fiber) = ヤーン座標の Jacobian（dω = cos(θ)dθdφ）。正規の角度なので cos >= 0
        cos_theta_fiber = torch.cos(wo[:, 0]).clamp(min=0.0)

        # 有効条件: 極点でなく（Jacobian が 0 でない）、pdf が正
        # TT 方向（表面の法線と逆向き）も許可して透過ローブを有効化
        valid = (cos_theta_fiber > 0.00005) & torch.isfinite(pdf) & (pdf > 0)

        # MIS 用の pdf: 光源サンプリング側の pdf() と同じ関数（逆 ODE、2つのチャートの和）を使い、
        # 同じ方向の2つの MIS 重みの和が 1 になるようにする。重み value には厳密な順方向の pdf を使う
        pdf_mis = self._pdf_theta_phi(wo, wi_input)
        valid &= torch.isfinite(pdf_mis) & (pdf_mis > 0)
        pdf = torch.where(valid, pdf, torch.zeros_like(pdf))
        pdf_mis = torch.where(valid, pdf_mis, torch.zeros_like(pdf_mis))

        # ---- ヤーン球面座標 → Mitsuba 3D 方向（シェーディングフレームのローカル座標） ----
        bs = mi.BSDFSample3f()
        bs.wo = to_mi_vector(self._yarn_to_local(yarn_spher_to_cart(wo), rot))

        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        # 立体角 pdf = pdf_θφ / cos_fiber（ヤーン座標 Jacobian）
        invcos_theta_fiber = dr.clamp(
            to_mi_float(1.0 / cos_theta_fiber.clamp(min=1e-6)), 1.0, floatmax
        )
        bs.pdf = to_mi_float(pdf_mis) * invcos_theta_fiber
        bs.sampled_component = mi.UInt32(0)
        bs.eta = mi.Float(1.0)  # ヤーンは屈折なし（透過も eta=1）
        # R か TT かを表面の法線成分 z = cos θ × sin φ の符号で判定
        is_refl = to_mi_float((torch.sin(wo[:, 1]) > 0).float()) > 0.5
        bs.sampled_type = dr.select(
            is_refl,
            mi.UInt32(+mi.BSDFFlags.GlossyReflection),
            mi.UInt32(+mi.BSDFFlags.GlossyTransmission),
        )

        # ---- サンプリング重みの計算 ----
        # weight = eval() / pdf_solid_angle = brdf_yarn × cos_fiber / pdf_θφ
        # eval() = (S_R+S_TT)*cos_fiber なので weight = eval() / pdf_solid_angle
        # eval_rgb: 反射色・透過色を RGB のまま評価（eval は輝度に潰れており色が失われる）
        # サンプリング（NN）は輝度で学習したままでよい。pdf が正しければ不偏
        brdf = self.bsdf.eval_rgb(wi_input, wo)  # (N, 3) = (S_R+S_TT)*cos_fiber
        value = brdf * (cos_theta_fiber / pdf.clamp(min=1e-8)).unsqueeze(1)
        value = torch.where(valid[:, None], value, torch.zeros_like(value))
        value = to_mi_color(value) * self.albedo

        return (bs, dr.select(bs.pdf > 0.0, value, mi.Color3f(0)))

    def _pdf_theta_phi(self, wo_t, wi_t):
        """正規の角度 wo_t の方向の (θ,φ) 空間の密度（逆 ODE、2つのチャート上の点の和）"""
        pdf = network_pdf_spherical(self.D_base, self.D_sample, wo_t, wi_t, T=8)
        pdf = pdf + network_pdf_spherical(self.D_base, self.D_sample, yarn_other_chart(wo_t), wi_t, T=8)
        return pdf.detach()

    def eval(self, ctx, si, wo, active=True):
        # khungurn.eval() は (S_R + S_TT) × cos(θ_fiber) を返す。
        # linearcurve では eval() をそのまま返す（neuralyarn の khungurn と同じ）。
        rot = self._yarn_rotation(si)
        wi_t = cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
        wo_t = cart_to_yarn_spher(self._local_to_yarn(wo.torch().to(device), rot))
        brdf_yarn = self.bsdf.eval_rgb(wi_t, wo_t)  # (N, 3) = (S_R + S_TT) × cos(θ_fiber)
        return to_mi_color(brdf_yarn) * self.albedo

    def pdf(self, ctx, si, wo, active=True):
        rot = self._yarn_rotation(si)
        wi_t = cart_to_yarn_spher(self._local_to_yarn(si.wi.torch().to(device), rot))
        wo_t = cart_to_yarn_spher(self._local_to_yarn(wo.torch().to(device), rot))
        pdf = self._pdf_theta_phi(wo_t, wi_t)
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
    attach_curve_frames(scene)

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