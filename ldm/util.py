import importlib

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import math

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

## Perspective Loss
class Sobel(nn.Module):
	def __init__(self):
		super().__init__()
		self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding="same", padding_mode="replicate", bias=False)

		Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
		Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
		G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
		G = G.unsqueeze(1)
		self.filter.weight = nn.Parameter(G, requires_grad=False)

	def forward(self, img):
		x = self.filter(img)
		return x
		
def get_perp_line(d):
	perp_d = torch.stack([d[1], -d[0]])

	return perp_d
	
def pos_enc(pos, L=10):
    lin = 2**torch.arange(0, L)

    x, y = pos
    x /= 512.
    y /= 512.
    enc = []
    for l in range(L):
        enc.extend([torch.sin(math.pi * (2**l) * x), torch.cos(math.pi * (2**l) * y)])
    return torch.tensor(enc, device="cuda")
	
def _local_maxima_1d(x):
	midpoints = torch.empty(x.shape[0] // 2, dtype=torch.long, device="cuda")
	left_edges = torch.empty(x.shape[0] // 2, dtype=torch.long, device="cuda")
	right_edges = torch.empty(x.shape[0] // 2, dtype=torch.long, device="cuda")
	m = 0

	i = 1
	i_max = x.shape[0] - 1
	
	while i < i_max:
		if x[i-1] < x[i]:
			i_ahead = i + 1

			while i_ahead < i_max and x[i_ahead] == x[i]:
				i_ahead += 1

			if x[i_ahead] < x[i]:
				left_edges[m] = i
				right_edges[m] = i_ahead - 1
				midpoints[m] = (left_edges[m] + right_edges[m]) // 2
				m += 1
				i = i_ahead
		i += 1

	midpoints.resize_(m)
	left_edges.resize_(m)
	right_edges.resize_(m)

	return midpoints, left_edges, right_edges
	
def _local_maxima_1d_vec(x):
    x_t1 = x[1:-1]-x[:-2]
    x_t2 = x[1:-1]-x[2:]
    
    out = torch.where(x_t1>0,1,0)
    out2 = torch.where(x_t2>0,out,0)
    
    midpoints = (torch.nonzero(out2)+1).reshape(-1)
    
    return midpoints

def _peak_prominences(x, peaks, wlen):
	prominences = torch.empty(peaks.shape[0], dtype=torch.float, device="cuda")
	left_bases = torch.empty(peaks.shape[0], dtype=torch.float, device="cuda")
	right_bases = torch.empty(peaks.shape[0], dtype=torch.float, device="cuda")

	for peak_nr in range(peaks.shape[0]):
		peak = peaks[peak_nr]
		i_min = 0
		i_max = x.shape[0] - 1
		if not i_min <= peak_nr <= i_max:
			print("error!")

		if wlen >= 2:
			i_min = max(peak - wlen // 2, i_min)
			i_max = min(peak + wlen // 2, i_max)

		i = left_bases[peak_nr] = peak.clone()
		left_min = x[peak]

		while i_min <= i and x[i] <= x[peak]:
			if x[i] < left_min:
				left_min = x[i]
				left_bases[peak_nr] = i
			i -= 1

		i = right_bases[peak_nr] = peak.clone()
		right_min = x[peak]

		while i <= i_max and x[i] <= x[peak]:
			if x[i] < right_min:
				right_min = x[i]
				right_bases[peak_nr] = i
			i += 1

		prominences[peak_nr] = x[peak] - max(left_min, right_min)

	return prominences, left_bases, right_bases
	
def _peak_prominences_vec(x, peaks):
    prominences = torch.empty(peaks.shape[0], dtype=torch.float, device="cuda")
    for peak_nr in range(peaks.shape[0]):
        peak = peaks[peak_nr]
        x_val = x[peak]
        larger_idx = torch.nonzero(torch.where(x > x_val, 1, 0)).reshape(-1)

        if larger_idx.shape[0] == 0:
            min_l = torch.min(x[:peak])
            min_r = torch.min(x[peak:])
        else:
            left_large = torch.max(torch.where(larger_idx < peak, larger_idx, 0))
            right_large = torch.min(torch.where(larger_idx > peak, larger_idx, x.shape[0]-1))

            min_l = torch.min(x[left_large:peak])
            min_r = torch.min(x[peak+1:right_large+1])
		
        prominences[peak_nr] = x_val - max(min_l, min_r)
    return prominences
    

def find_peaks(x, prominence=1, wlen=-1):
    #import time
    #s1 = time.time()
    #peaks2, left_edges, right_edges = _local_maxima_1d(x)
    #e1 = time.time()
    #s2 = time.time()
    peaks = _local_maxima_1d_vec(x)
    #e2 = time.time()
	
    #print(e1-s1,e2-s2)
    #print("***")
	#print(peaks.shape, peaks2.shape)

    pmin, pmax = prominence, None

    #s3 = time.time()
    #prominences, left_bases, right_bases = _peak_prominences(x, peaks, wlen)
    #e3 = time.time()
    
    #s4 = time.time()
    prominences = _peak_prominences_vec(x, peaks)
    #e4 = time.time()
    #print(prominences, prominences2)
    #print(e3-s3)
    #print(e4-s4)
    #print("^^^^")

    keep = (pmin <= prominences)

    peaks = peaks[keep]
    prominences = prominences[keep]

    return peaks, prominences



def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=True):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss
