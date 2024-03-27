import sys
import os
import cv2
import argparse
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import random
from natsort import natsorted
from glob import glob
from pathlib import Path
import imageio
import h5py
import mat73

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import functools
from torch.nn import init
import torchvision
from torchvision.transforms import Compose


from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
# %load_ext tensorboard
# %matplotlib inline
import matplotlib.pyplot as plt

"""# Helper"""

# Plot image
def show_img(img, title=None, figsize=(15, 15)):
  #img = (img+1)/2
  plt.figure(figsize=figsize)
  plt.imshow(img)
  if title:
    plt.title(title)
  plt.show()


params = {
  'batch_size': 1, # batch size
  'num_epochs': 100, # number of epochs to train
  'warmup_epochs': 4, # number of epochs for warmup
  'initial_lr': 1e-4, # initial learning rate used by scheduler
  'min_lr': 1e-6, # minimum learning rate used by scheduler
  'val_epoch': 1, # validation done every k epochs
  'save_every': 5, # save every k epoch
  'save_dir': './checkpoints/',
  'root_dir_list': [
      ''
    ], # Dir for the training data
  'val_dir_list': [
      ''
    ], # Dir for the validation data
  'scale':  1.0, # scale for Dense Dataset
  'trans' : 0.0, # translation for Dense Dataset
  'masked_ssi_loss_weight': False, # weight for masked l1 loss (only where lidar points are)
  'smooth_loss_weight': False, # weight for smooth loss
  'ssi_trim_loss_weight': 1.0,
  'scale_inv_grad_loss_weight': 0.5,
  'resume_train': False, # begin training using loaded checkpoint
  'model_path': './checkpoints/dpt_hybrid-midas-501f0c75.pt', # Dir to load model weights
  'tensorboard_log_step_train': 100, # Number of steps to log into tensorboard when training
  'tensorboard_log_step_val': 1, # This number will be updated automatically based after creating the dataloadersrue
  'test': True,
  'test_root_dir': [''],
  'test_root_dir2': ['']
}


# DataLoaders for Training and Validation set
class DPTData(Dataset):
  """
    The dataset class for testing.

    Parameters:
        root_dir_list (list) -- list of dirs for the dataset.
        is_train (bool) -- True for training set.
  """
  def __init__(self, root_dir_list, is_train=True):
    super(DPTData, self).__init__()

    self.is_train = is_train
    self.img_paths = []
    self.depth_paths = []
    for root_dir in root_dir_list:
      print(root_dir)
      self.img_paths += natsorted(glob(f"{root_dir}/img/*.jpg"))
      self.depth_paths += natsorted(glob(f"{root_dir}/depth/*.npy"))
       
    # number of images
    self.data_len = len(self.img_paths)
  
  def __len__(self):
    # return 16
    return self.data_len

  def get_scene_indices(self):
    return self.scene_indices
  
  def __getitem__(self, index):
    inp_path = self.img_paths[index]
    inp_img = Image.open(inp_path)
    filename = inp_path.split('/')[-1][:-4]

    depth_path = self.depth_paths[index]
    
    depth_img = np.load(depth_path)

    # To numpy
    inp_img = np.array(inp_img, dtype=np.float32)
    depth_img = np.array(depth_img, dtype=np.float32)
    depth_img = depth_img[:,:,None]

    if not np.isfinite(inp_img).all() or not np.isfinite(depth_img).all():
      print("Non finite!")

    inp_img *= 1/255.0
    
    # Crop Code
    h = inp_img.shape[0]
    w = inp_img.shape[1]

    # Random Crop for square
    cc_x = random.randint(0, 512-384)
    cc_y = random.randint(0, 512-384)
    inp_img = inp_img[cc_y:cc_y+384, cc_x:cc_x+384,:]
    depth_img = depth_img[cc_y:cc_y+384, cc_x:cc_x+384,:]

    inp_img = (inp_img - .5) / .5
    
    inp_img = torch.from_numpy(inp_img).permute((2,0,1))
    depth_img = torch.from_numpy(depth_img).permute((2,0,1))

    # Data augmentations: flip x, flip y, rotate by (90, 180, 270), combinations of flipping and rotating
    if self.is_train:
      aug = random.randint(0, 1)
    else:
      aug = 0
    
    if aug==1:
      inp_img = inp_img.flip(2)
      depth_img = depth_img.flip(2)

    # Dict for return
    # If using tanh as the last layer, the range should be [-1, 1]
    sample_dict = {
        'input_img': inp_img,
        'depth_img': depth_img,
        'file_name': filename
    }

    return sample_dict

# Create the DataLoaders for training and validation
train_dataset = DPTData(
    root_dir_list=params['root_dir_list'],
    is_train=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1, 
)


"""
LOSS FUNCTIONS
"""
class ScaleInvariantLoss(nn.Module):
  #Scale Invariant Loss according to https://arxiv.org/pdf/1406.2283.pdf

  def __init__(self, eps=.5):
    super(ScaleInvariantLoss, self).__init__()
    self.eps = eps
  
  def forward(self, tar, est):
    diff = (torch.log(tar) - torch.log(est))
    l2 = torch.mean(torch.square(diff))
    scale_inv = torch.sum(diff)**2
    num_pixels = tar.size(dim=0)
    total = l2 - (self.eps/(num_pixels**2))*scale_inv
    return total

class SSITRIM(nn.Module):
  def __init__(self):
    super(SSITRIM, self).__init__()

  def forward(self, d, d_star):
    M = d.shape[2]*d.shape[3]
    M_80 = int(0.8 * M)
    d = d.flatten(start_dim=1)
    d_star = d_star.flatten(start_dim=1)
    t_d = torch.median(d, dim=1)[0]
    s_d = (torch.sum(torch.abs(d - t_d.unsqueeze(1))) + 1e-8) / M
    d = (d - t_d.unsqueeze(1)) / s_d
    t_dstar = torch.median(d_star, dim=1)[0]
    s_dstar = (torch.sum(torch.abs(d_star - t_dstar.unsqueeze(1))) + 1e-8) / M
    d_star = (d_star - t_dstar.unsqueeze(1)) / s_dstar
    diff = torch.abs(d - d_star)
    filter_diff = torch.topk(diff, k=M_80, dim=1, largest=False)[0]
    loss = torch.sum(filter_diff) / (2 * M)
    return loss

class ScaleInvariantGradientLoss(nn.Module):
  def __init__(self, scale):
    super(ScaleInvariantGradientLoss, self).__init__()
    self.scales = []
    for i in range(scale):
      self.scales.append(1 / (2**i))

  def interpolate(self, img, scale):
    return torch.nn.functional.interpolate(img, scale_factor=scale, mode="bilinear")

  def standardize(self, arr):
    M = arr.shape[2] * arr.shape[3]
    t = torch.median(arr.flatten(start_dim=1), dim=1)[0]
    t = t.reshape(-1,1,1,1)
    s = torch.sum(torch.abs(arr - t)) / M
    arr = (arr - t) / s
    return arr

  def forward(self, d, d_star):
    loss = torch.zeros(d.shape[0], device="cuda")
    d = self.standardize(d)
    d_star = self.standardize(d_star)

    for s in self.scales:
      d_resize = self.interpolate(d, s)
      d_star_resized = self.interpolate(d_star, s)

      R = d_resize - d_star_resized

      R_x = torch.mean(torch.abs(R[:,:,:,:-1] - R[:,:,:,1:]), axis=(1,2,3))
      R_y = torch.mean(torch.abs(R[:,:,:-1,:] - R[:,:,1:,:]), axis=(1,2,3))

      R_grad = R_x + R_y

      loss += R_grad
    mloss = torch.mean(loss)
    return mloss

def SmoothLoss(disp, img):
  """Computes the smoothness loss for a disparity image
  The color image is used for edge-aware smoothness
  """
  grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
  grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

  grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
  grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

  grad_disp_x *= torch.exp(-grad_img_x)
  grad_disp_y *= torch.exp(-grad_img_y)

  return grad_disp_x.mean() + grad_disp_y.mean()

torch.manual_seed(100)
np.random.seed(100)

# Create dir to save the weights
Path(params['save_dir']).mkdir(parents=True, exist_ok=True)

test_dataset = DPTData(params['test_root_dir2'])

test_dataloader = DataLoader(
  dataset = test_dataset,
  batch_size=params['batch_size'],
  num_workers=2,
  pin_memory=True
)

print('Test set length:', len(test_dataset))

# Adjust the log freq based on the number of training and val samples
#params['tensorboard_log_step_val'] = int(params['tensorboard_log_step_train'] * len(val_dataset) / len(train_dataset))

model = DPTDepthModel(
            invert=False,
            path=params['model_path'],
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
model.cuda()

key_name_list = ['blocks']

criterion_masked_ssi = ScaleInvariantLoss().cuda()
ssi_trim = SSITRIM().cuda()
scale_inv_grad_loss = ScaleInvariantGradientLoss(scale=4).cuda()
L1_loss = nn.L1Loss().cuda()

start_epoch = 0

if False and (params['resume_train'] or params['test']):
  print(f"Loading checkpoint {params['model_path']}")
  checkpoint = torch.load(params['model_path'])

  # Load Model
  model.load_state_dict(checkpoint['state_dict'])
  start_epoch = checkpoint['epoch'] + 1
  print(f"Resuming epoch: {start_epoch}")

  # # Load the optimizer we were using for that run
  # optimizer_first.load_state_dict(checkpoint['optimizer_first'])
  # optimizer_second.load_state_dict(checkpoint['optimizer_second'])

def get_s_t(depth_pred, depth_gt):
  depth_pred  = depth_pred.detach().cpu().numpy()
  depth_gt  = depth_gt.detach().cpu().numpy()  
  flattened_pred = depth_pred.flatten()
  flattened_gt = depth_gt.flatten()
  pred_mat = np.vstack([flattened_pred, np.ones(len(flattened_pred))]).T
  s, t = np.linalg.lstsq(pred_mat, flattened_gt, rcond=None)[0]
  return s, t

def compute_scale_and_shift(prediction, target, mask):
  # system matrix: A = [[a_00, a_01], [a_10, a_11]]
  a_00 = torch.sum(mask * prediction * prediction, (1, 2))
  a_01 = torch.sum(mask * prediction, (1, 2))
  a_11 = torch.sum(mask, (1, 2))

  # right hand side: b = [b_0, b_1]
  b_0 = torch.sum(mask * prediction * target, (1, 2))
  b_1 = torch.sum(mask * target, (1, 2))

  # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
  x_0 = torch.zeros_like(b_0)
  x_1 = torch.zeros_like(b_1)

  det = a_00 * a_11 - a_01 * a_01
  # A needs to be a positive definite matrix.
  valid = det > 0

  x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
  x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

  return x_0, x_1

def get_sum_loss(net_out, gt, criterion):
  if isinstance(net_out, list):
    loss = 0
    for net_out_i in net_out:
      loss += criterion(net_out_i, gt)
  else:
    loss = criterion(net_out, gt)
  return loss

def standardize(arr):
  M = arr.shape[2] * arr.shape[3]
  a_max = torch.max(arr.flatten(start_dim=1), dim=1)[0]
  a_min = torch.min(arr.flatten(start_dim=1), dim=1)[0]
  a_max = a_max.reshape(-1,1,1,1)
  a_min = a_min.reshape(-1,1,1,1)
  arr = (arr - a_min) / (a_max - a_min + 0.001)
  return arr

def abs_rel(out, gt):
  diff = torch.abs(out.flatten(start_dim=1) - gt.flatten(start_dim=1)) / gt.flatten(start_dim=1)
  tempdiff = torch.nan_to_num(diff, nan=-1, posinf=-1, neginf=-1)
  diff = torch.nan_to_num(diff, nan=0, posinf=0, neginf=0)
  N = torch.count_nonzero(tempdiff > 0.0, dim=1)
  N[N == 0] = 1
  loss = torch.mean(torch.sum(diff, axis=1) / N)
  return loss

def sq_rel(out, gt):
  diff = torch.square(out.flatten(start_dim=1) - gt.flatten(start_dim=1)) / gt.flatten(start_dim=1)
  tempdiff = torch.nan_to_num(diff, nan=-1, posinf=-1, neginf=-1)
  diff = torch.nan_to_num(diff, nan=0, posinf=0, neginf=0)
  N = torch.count_nonzero(tempdiff > 0.0, dim=1)
  N[N == 0] = 1
  loss = torch.mean(torch.sum(diff, axis=1) / N)
  return loss

def rmse(out, gt):
  loss = torch.mean(torch.sqrt(torch.nn.functional.mse_loss(out, gt)))
  return loss

def log_rmse(out, gt):
  loss = torch.mean(torch.sqrt(torch.nn.functional.mse_loss(torch.log(out+1e-15), torch.log(gt+1e-15))))
  return loss

# TRAINING AND VALIDATION

for epoch in range(start_epoch, params['num_epochs']):
  print(epoch)
  epoch_loss = 0

  # TRAINING
  model.train()

  for batch_idx, batch_data in enumerate(tqdm(train_loader)):
    
    # Load the data
    input_img = batch_data['input_img'].cuda()
    depth_img = batch_data['depth_img'].cuda()

    # Check the current step
    current_step = epoch * len(train_loader) + batch_idx

    #########
    # Train #
    #########

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass through the entire model (both parts)
    out = model(input_img)
    out = mout / torch.max(mout)
    out = out * params['scale'] + params['trans']
    out[out>100] = 100
    out = torch.unsqueeze(out, 1)

    out_masked = out[depth_img>0]
    depth_img_masked = depth_img[depth_img>0]

    # Calculate losses between gt and output
    loss = 0

    if params['masked_ssi_loss_weight']:
      out_masked = out[depth_img>0]
      depth_img_masked = depth_img[depth_img>0]
      loss_ssi_masked = get_sum_loss(out_masked, depth_img_masked, criterion_masked_ssi)
      loss += params['masked_ssi_loss_weight'] * loss_ssi_masked
      loss_ssi_masked_log = loss_ssi_masked.item()

    if params['smooth_loss_weight']:
      loss_smooth = SmoothLoss(out, input_img)
      loss += params['smooth_loss_weight'] * loss_smooth
      loss_smooth_log = loss_smooth.item()

    if params['ssi_trim_loss_weight']:
      loss_ssi_trim = get_sum_loss(out, depth_img, ssi_trim)
      loss += params['ssi_trim_loss_weight'] * loss_ssi_trim
      loss_ssi_trim_log = loss_ssi_trim.item()

    if params['scale_inv_grad_loss_weight']:
      loss_scale_inv_grad = get_sum_loss(out, depth_img, scale_inv_grad_loss)
      loss += params['scale_inv_grad_loss_weight'] * loss_scale_inv_grad
      loss_scale_inv_grad_log = loss_scale_inv_grad.item()

    # Backwards pass and step
    loss.backward()
    optimizer.step()

    # Log

    epoch_loss += loss
    loss_log = loss.item()

    # Tensorboard
    if (current_step % params['tensorboard_log_step_train']) == 0:
      
      # Log loss
      writer.add_scalar('loss/train', loss_log, current_step)

      if params['masked_ssi_loss_weight']:
        writer.add_scalar('masked_ssi_loss/train', loss_ssi_masked_log, current_step)
      if params['smooth_loss_weight']:
        writer.add_scalar('smooth_loss/train', loss_smooth_log, current_step)
      if params['ssi_trim_loss_weight']:
        writer.add_scalar('ssi_trim_loss/train', loss_ssi_trim, current_step)
      if params['scale_inv_grad_loss_weight']:
        writer.add_scalar('scale_inv_grad_loss/train', loss_ssi_trim, current_step)
  
  # Print info
  print(
  f"Epoch: {epoch}\n"
  f"Train Loss: {epoch_loss / len(train_loader):.4f}\n"
  f"Learning Rate First {optimizer.param_groups[0]['lr']:.8f}\t"
  )
  out_plot = out[0, 0, :, :].detach().cpu()
  depth_img_plot = depth_img[0, 0, :, :].cpu()
  plt.figure()
  plt.imshow(input_img[0, :, :, :].cpu().permute(1, 2, 0) * .5 + .5)
  plt.figure()
  plt.imshow(out_plot, cmap = 'jet')
  plt.colorbar()
  plt.figure()
  plt.imshow(depth_img_plot, cmap = 'jet')
  plt.colorbar()
  plt.savefig("epoch_"+str(epoch)+".jpg")
  plt.close()

  ##############
  # Validation #
  ##############

  if epoch %  params['val_epoch'] == 0:
    model.eval()
    epoch_loss = 0

    val_loop = tqdm(val_loader, leave=False, position=0)
    val_loop.set_description('Val Epoch')
    for batch_idx, batch_data in enumerate(val_loop):
      
      # Load data
      input_img = batch_data['input_img'].cuda()
      depth_img = batch_data['depth_img'].cuda()

      # Check the current step
      current_step = epoch * len(val_loader) + batch_idx

      # Forward pass of model
      with torch.no_grad():
        out = model(input_img)
        #out = mout / torch.max(mout)
        #out = out * params['scale'] + params['trans']
        #out[out>100] = 100
        out = torch.unsqueeze(out, 1)

        # Calculate losses between pseudo-gt and output
        loss = 0

        if params['masked_ssi_loss_weight']:
          out_masked = out[depth_img>0]
          depth_img_masked = depth_img[depth_img>0]
          loss_ssi_masked = get_sum_loss(out_masked, depth_img_masked, criterion_masked_ssi)
          loss += params['masked_ssi_loss_weight'] * loss_ssi_masked
          loss_ssi_masked_log = loss_ssi_masked.item()

        if params['smooth_loss_weight']:
          loss_smooth = SmoothLoss(out, input_img)
          loss += params['smooth_loss_weight'] * loss_smooth
          loss_smooth_log = loss_smooth.item()

        if params['ssi_trim_loss_weight']:
          loss_ssi_trim = get_sum_loss(out, depth_img, ssi_trim)
          loss += params['ssi_trim_loss_weight'] * loss_ssi_trim
          loss_ssi_trim_log = loss_ssi_trim.item()

        if params['scale_inv_grad_loss_weight']:
          loss_scale_inv_grad = get_sum_loss(out, depth_img, scale_inv_grad_loss)
          loss += params['scale_inv_grad_loss_weight'] * loss_scale_inv_grad
          loss_scale_inv_grad_log = loss_scale_inv_grad.item()


        # if params['ssim_loss_weight']:
        #   loss_ssim = get_sum_loss(second_stage_out, target_img, criterion_neg_ssim)
        #   loss += params['ssim_loss_weight'] * loss_ssim
        #   loss_ssim_second_log = loss_ssim.item()
        
        #out_masked = out[depth_img>0]
        #depth_img_masked = depth_img[depth_img>0]
        epoch_loss += loss#get_sum_loss(out_masked, depth_img_masked, L1_loss).item()
        loss_log = loss.item()
      
        # Tensorboard
        if (current_step % params['tensorboard_log_step_val']) == 0:
          
          # Log loss
          writer.add_scalar('loss_second/val', loss_log, current_step)

          # Seperate loss
          if params['masked_ssi_loss_weight']:
            writer.add_scalar('masked_ssi_loss/val', loss_ssi_masked_log, current_step)
          if params['smooth_loss_weight']:
            writer.add_scalar('smooth_loss/val', loss_smooth_log, current_step)
          if params['ssi_trim_loss_weight']:
            writer.add_scalar('ssi_trim_loss/train', loss_ssi_trim, current_step)
          if params['scale_inv_grad_loss_weight']:
            writer.add_scalar('scale_inv_grad_loss/train', loss_ssi_trim, current_step)
    # Print info
    avg_val_loss = epoch_loss / len(val_loader)
    print(
        f"Val Epoch\t"
        f"Val Loss: {avg_val_loss:.4f}"
        )
    
    # Log images 
    out_plot = out[0, 0, :, :].cpu()
    depth_img_plot = depth_img[0, 0, :, :].cpu()
    plt.figure()
    plt.imshow(input_img[0, :, :, :].cpu().permute(1, 2, 0) * .5 + .5)
    plt.figure()
    plt.imshow(out_plot, cmap = 'jet')
    plt.colorbar()
    plt.figure()
    plt.imshow(depth_img_plot, cmap = 'jet')
    plt.colorbar()
    plt.savefig("val_epoch_"+str(epoch)+".jpg")
    plt.close()
  
  # Move the scheduler forward
  scheduler.step()

  # Save every few epochs
  if epoch % params['save_every'] == 0:
    print('Saving...')
    torch.save({
        'epoch': epoch, 
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}, 
        os.path.join(params['save_dir'], f'model_epoch_{epoch}.pth')
        )
  
  # Tensorboard
  writer.flush()
# Close tensorboard
writer.close()

