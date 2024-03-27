import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import math
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import matplotlib.pyplot as plt

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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

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
    for l in layers_to_change:
        sd[l] = modify_proj_weights(sd[l])


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
    	"--config2",
    	type=str,
    	default="",
    	help="",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
    	"--ckpt2",
    	type=str,
    	default="",
    	help="",
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

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    
    #config2 = OmegaConf.load(f"{opt.config2}")
    #model2 = load_model_from_config(config2, f"{opt.ckpt2}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    #model2 = model2.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
        #sampler2 = PLMSSampler(model2)
    else:
        sampler = DDIMSampler(model)
        #sampler2 = DDIMSampler(model2)
        
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    
    vps = torch.tensor([[0.,0.],[768.,256.],[768.,256.]])
    
    vcs = []
    for vp in vps:
        vcs.append(torch.tile(pos_enc(vp).unsqueeze(1).unsqueeze(0), (1,1,768)))
    
    vc = torch.cat(vcs, axis=0)
    
    #print(vc.shape)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    sc = None
    sc2 = None
    
    saved_start_code = start_code.clone().detach()

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        print(prompts)
                        c = model.get_learned_conditioning(prompts)
                        #c = torch.cat([c,vc], axis=1)
                        print(c.shape)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sc = start_code[0].clone()
                        plt.imshow(torch.permute(start_code[0], (1,2,0)).cpu().numpy().astype(np.float64))
                        plt.savefig("start_code.png")
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         dynamic_threshold=opt.dyn,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()
            """
            with model2.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model2.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        print(prompts)
                        c = model2.get_learned_conditioning(prompts)
                        #c = torch.cat([c,vc], axis=1)
                        print(c.shape)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sc2 = start_code[0].clone()
                        print("DIFF")
                        print(torch.sum(torch.abs(sc2 - sc)))
                        plt.imshow(torch.permute(start_code[0], (1,2,0)).cpu().numpy().astype(np.float64))
                        plt.savefig("start_code2.png")
                        plt.imshow(torch.permute(saved_start_code[0], (1,2,0)).cpu().numpy().astype(np.float64))
                        plt.savefig("start_code3.png")   
                                             
                        samples_ddim, _ = sampler2.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         dynamic_threshold=opt.dyn,
                                                         x_T=saved_start_code)

                        x_samples_ddim = model2.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"2-{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid2-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()
            """
    #diff = sc - sc2
    #plt.imshow(torch.permute(diff, (1,2,0)).cpu().numpy())
    #plt.savefig("start_code_diff.png")
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f"Sampling took {toc - tic}s, i.e. produced {opt.n_iter * opt.n_samples / (toc - tic):.2f} samples/sec."
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
