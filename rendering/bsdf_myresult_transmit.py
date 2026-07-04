# This file contains the implementation of a BSDF with transmission support.
# Based on bsdf_myresult.py but includes both reflection and transmission.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from tqdm import tqdm
import torch
from utils.model import *

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
p = str(SCRIPT_DIR)
sys.path.insert(1, p)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.mitsuba_brdf_draw import *
from utils.analytical_brdf_torch import *
from utils.mlp_brdf_sampling import *
import argparse

torch.set_default_dtype(torch.float32)
mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)


parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, default="scene_bsdf.xml")
parser.add_argument("--passes", type=int, default=128)

parser = parser.parse_args()

from utils.bsdf_dict import *

def to_mi_float(tensor):
    return mi.Float(tensor.detach().cpu().numpy())


def load_state_dict_safe(path, model):
    try:
        state_dict = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)


def as_points(tensor):
    if tensor.ndim == 2 and tensor.shape[0] == 3 and tensor.shape[1] != 3:
        return tensor.transpose(0, 1)
    return tensor

def sph_to_dir(theta, phi):
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

def cart_to_spher(xyz):
    xyz = as_points(xyz)
    if xyz.ndim == 2 and xyz.shape[1] == 2:
        return xyz
    r = torch.norm(xyz, dim=1)
    theta = torch.acos(xyz[:,2]/(r+1e-8))
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([theta, phi], dim=1)

class MyBSDFTransmit(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.idx = props["idx"]
        self.albedo = mi.Color3f(props["albedo"])
        self.bsdf = bsdf_materials[self.idx]
        self.transmission_ratio = props.get("transmission_ratio", 0.5)  # Probability of transmission vs reflection
        self.ior = props.get("ior", 1.5)  # Index of refraction for transmission
        
        self.D_sample = NN_cond_pos(input_dim=6,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to(device)
        rectify_path = SCRIPT_DIR / "checkpoints_new" / f"bsdf_{self.idx}_spherical" / f"brdf_rectify_network{self.idx}.pth"
        load_state_dict_safe(rectify_path, self.D_sample)
        self.D_sample.eval()
        
        self.D_base = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16).to(device)
        pretrain_path = SCRIPT_DIR / "checkpoints_new" / f"bsdf_{self.idx}_spherical" / f"brdf_pretrain_network{self.idx}.pth"
        load_state_dict_safe(pretrain_path, self.D_base)
        
        # Include both reflection and transmission
        reflection_flags = mi.BSDFFlags.Diffuse | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        
        self.m_components = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample(self, ctx, si, sample1, sample2, active=True):

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        wi = as_points(si.wi.torch())
        wi_input = cart_to_spher(wi)
        
        wo, pdf = network_sampling_spherical(self.D_base, self.D_sample, wi_input, T=8)
        pdf = torch.where(torch.sin(wo[:,0]) > 0.00005, pdf, torch.zeros_like(pdf))

        wo = mi.Vector2f(to_mi_float(wo[...,0]), to_mi_float(wo[...,1]))
        wo = sph_to_dir(wo.x, wo.y)
        
        bs = mi.BSDFSample3f()
        
        bs.wo = wo           
        
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        
        invsin_theta_o = dr.clip(1 / (dr.abs(mi.Frame3f.sin_theta(bs.wo))), 1, floatmax)
        if dr.any_nested(invsin_theta_o < 0):
            print("invsin_theta_o<0")
        bs.pdf = to_mi_float(pdf) * invsin_theta_o
        
        # Randomly choose between reflection and transmission
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        is_reflection = cos_theta_o > 0.0
        
        # Set eta and sampled_type based on reflection vs transmission
        bs.eta = dr.select(is_reflection, 1.0, self.ior)
        bs.sampled_type = dr.select(is_reflection, mi.UInt32(8), mi.UInt32(16))  # 8=reflection, 16=transmission
        
        bs.sampled_component = dr.select(is_reflection, mi.UInt32(0), mi.UInt32(1))
        
        wi_input = as_points(si.wi.torch())[...,:2]
        wo_tmp = as_points(bs.wo.torch())[...,:2]
        brdf = self.bsdf.eval(ctx, si, bs.wo)
        value = brdf * self.albedo / to_mi_float(pdf) * mi.Frame3f.sin_theta(bs.wo)
        value = dr.select((bs.pdf > 0.0), value, mi.Vector3f(0))
        
        pdf = bs.pdf.torch()
        value_torch = as_points(value.torch())[:,0]
        pdf = torch.where(value_torch < 3.5, pdf, torch.zeros_like(pdf))
        bs.pdf = to_mi_float(pdf) 
        return (bs, dr.select((bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        brdf = self.bsdf.eval(ctx, si, wo)
        
        # For reflection: both cos_theta > 0
        # For transmission: opposite signs
        is_reflection = (cos_theta_i > 0.0) & (cos_theta_o > 0.0)
        is_transmission = (cos_theta_i > 0.0) & (cos_theta_o < 0.0) | (cos_theta_i < 0.0) & (cos_theta_o > 0.0)
        
        value = brdf * self.albedo
        
        # Apply to both reflection and transmission (simplified)
        value = dr.select(is_reflection | is_transmission, value, mi.Vector3f(0))
        
        return value

    def pdf(self, ctx, si, wo, active=True):

        wi = as_points(si.wi.torch())
        wi_input = cart_to_spher(wi)
        wo_torch = as_points(wo.torch())
        wo_input = cart_to_spher(wo_torch)
        pdf = network_pdf_spherical(self.D_base, self.D_sample, wo_input, wi_input, T=8) 
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        
        invsin_theta_o = dr.clip(1 / (dr.abs(mi.Frame3f.sin_theta(wo))), 1, floatmax)
        pdf = to_mi_float(pdf) * invsin_theta_o
        
        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return "MyBSDFTransmit[\n" "    albedo=%s,\n" "    ior=%.4f,\n" "]" % (self.albedo, self.ior)


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    import time

    start_time = time.time()
    mi.register_bsdf("mybsdf", lambda props: MyBSDFTransmit(props))

    scene_file = parser.scene_file
    if not scene_file.lower().endswith(".xml"):
        scene_file += ".xml"

    if os.path.isabs(scene_file) or os.path.dirname(scene_file):
        scene_path = os.path.abspath(scene_file)
    else:
        scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "matpreview", scene_file))

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file does not exist: {scene_path}")

    scene = mi.load_file(scene_path)

    # print(params)
    SPP = 4
    spp = SPP * parser.passes

    seed = 0
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "diffusion_bsdf_myresult_transmit"))
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{parser.scene_file}.png")
    mi.util.write_bitmap(filepath, image, spp)
    filepath = os.path.join(output_dir, f"{parser.scene_file}.exr")
    mi.util.write_bitmap(filepath, image, spp)
