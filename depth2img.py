import os, sys
import argparse
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import repeat, rearrange
from itertools import islice
import time
import math
from pytorch_lightning import seed_everything
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.util import AddMiDaS
from ldm.modules.losses.perspective import perspective_loss

def pos_enc(pos, L=10):
    lin = 2**torch.arange(0, L)

    x, y = pos
    x /= 512.
    y /= 512.
    enc = []
    for l in range(L):
        enc.extend([torch.sin(math.pi * (2**l) * x), torch.cos(math.pi * (2**l) * y)])
    return torch.tensor(enc, device="cuda")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def modify_proj_weights(w):
    new_w = w[:,:,None,None]
    return new_w

def load_model_from_config(config, ckpt, verbose=False, fix_size=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    layers_to_change = [
        "model.diffusion_model.input_blocks.1.1.proj_in.weight",
        "model.diffusion_model.input_blocks.1.1.proj_out.weight",
        "model.diffusion_model.input_blocks.2.1.proj_in.weight",
        "model.diffusion_model.input_blocks.2.1.proj_out.weight",
        "model.diffusion_model.input_blocks.4.1.proj_in.weight",
        "model.diffusion_model.input_blocks.4.1.proj_out.weight",
        "model.diffusion_model.input_blocks.5.1.proj_in.weight",
        "model.diffusion_model.input_blocks.5.1.proj_out.weight",
        "model.diffusion_model.input_blocks.7.1.proj_in.weight",
        "model.diffusion_model.input_blocks.7.1.proj_out.weight",
        "model.diffusion_model.input_blocks.8.1.proj_in.weight",
        "model.diffusion_model.input_blocks.8.1.proj_out.weight",
        "model.diffusion_model.middle_block.1.proj_in.weight",
        "model.diffusion_model.middle_block.1.proj_out.weight",
        "model.diffusion_model.output_blocks.3.1.proj_in.weight",
        "model.diffusion_model.output_blocks.3.1.proj_out.weight",
        "model.diffusion_model.output_blocks.4.1.proj_in.weight",
        "model.diffusion_model.output_blocks.4.1.proj_out.weight",
        "model.diffusion_model.output_blocks.5.1.proj_in.weight",
        "model.diffusion_model.output_blocks.5.1.proj_out.weight",
        "model.diffusion_model.output_blocks.6.1.proj_in.weight",
        "model.diffusion_model.output_blocks.6.1.proj_out.weight",
        "model.diffusion_model.output_blocks.7.1.proj_in.weight",
        "model.diffusion_model.output_blocks.7.1.proj_out.weight",
        "model.diffusion_model.output_blocks.8.1.proj_in.weight",
        "model.diffusion_model.output_blocks.8.1.proj_out.weight",
        "model.diffusion_model.output_blocks.9.1.proj_in.weight",
        "model.diffusion_model.output_blocks.9.1.proj_out.weight",
        "model.diffusion_model.output_blocks.10.1.proj_in.weight",
        "model.diffusion_model.output_blocks.10.1.proj_out.weight",
        "model.diffusion_model.output_blocks.11.1.proj_in.weight",
        "model.diffusion_model.output_blocks.11.1.proj_out.weight"
    ]
    if fix_size:
        for l in layers_to_change:
            sd[l] = modify_proj_weights(sd[l])
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def get_samples(batch, model, sampler, seed=50, n_samples=1, scale=5.0):
    seed_everything(seed)
    t_enc = 49

    with torch.no_grad(), torch.autocast("cuda"):
        # z = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))
        z = batch["image"]
        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = []
        depth_data = batch["midas_in"]
        
        depth_data = torch.nn.functional.interpolate(
                depth_data,
                size=z.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        depth_min, depth_max = torch.amin(depth_data, dim=[1, 2, 3], keepdim=True), torch.amax(depth_data, dim=[1, 2, 3], keepdim=True)
        depth_data = 2. * (depth_data - depth_min) / (depth_max - depth_min) - 1.
        
        cond = {"c_concat": [depth_data], "c_crossattn": [c]}

        uc_cross = model.get_unconditional_conditioning(n_samples, "")
        uncond = {"c_concat": [depth_data], "c_crossattn": [uc_cross]}

        z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc] * n_samples).to(model.device))

        samples = sampler.decode(z_enc, cond, t_enc, 
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uncond, callback=None)
        x_samples = model.decode_first_stage(samples)
        result = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        output_img = torch.permute(result[0], (1,2,0)).cpu().numpy()
        
        return output_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--dyn",
        type=float,
        help="dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="",
        help="depth map to condition on"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(base_config, f"{opt.ckpt}", fix_size=opt.fix_size)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=True)

    t_enc = 45

    midas_trafo = AddMiDaS(model_type="dpt_hybrid")

    depth_fname = datapoint[0]
    img_fname = datapoint[1]
    prompt = opt.prompt
    
    # image = np.array(Image.open(opt.data_dir+img_fname).convert("RGB"))
    # image = cv2.resize(image, (512,512))
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    # image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.

    depth_arr = np.load(opt.data_dir+depth_fname)

    depth_arr = cv2.resize(depth_arr, (512,512))

    batch = {
        "image": start_code,
        "txt": opt.n_samples * [prompt],
    }

    batch["midas_in"] = torch.from_numpy(depth_arr[None,None,:,:]).to(device=device, dtype=torch.float32)

    output_imgs = get_samples(batch, model, sampler, seed=opt.seed, n_samples=opt.n_samples, scale=opt.scale)

    for idx, img in enumerate(output_imgs):
        img = (img * 255).astype(np.uint8)
        I = Image.fromarray(img)
        I.save(f"{opt.outdir}/{idx}.png")

if __name__ == "__main__":
    main()