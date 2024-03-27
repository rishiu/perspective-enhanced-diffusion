import torch
from ldm.util import Sobel, find_peaks, get_perp_line
from torchvision.transforms.functional import rgb_to_grayscale
import time
import matplotlib.pyplot as plt
import numpy as np

def pearson_corr_coef(signal, signal_gt, T):
    numerator = T * torch.sum(torch.mul(signal, signal_gt)) - (torch.sum(signal) * torch.sum(signal_gt))
    denom_left = T * torch.sum(torch.square(signal_gt)) - torch.square(torch.sum(signal_gt))
    denom_right = T * torch.sum(torch.square(signal)) - torch.square(torch.sum(signal))
    corr = numerator / torch.sqrt(denom_left * denom_right)
    return corr

def perspective_loss_img(img, vpts, sobel):
    out = sobel(img.float())

    dx = out[0]
    dy = out[1]
    
    h, w = img.shape[1:]
    

    if len(vpts.shape) == 1:
        vpts = [vpts]
    vpt_count = 0

    vals_arr = []
    for vp in vpts:
        if not torch.all(torch.isfinite(vp)) or (vp[0] == 0.0 and vp[1] == 0.0):
            continue
        vpt_count += 1
        
        min_angle = float("Inf")
        max_angle = -float("Inf")
        for i in [0,h]:
            for j in [0,w]:
                v = torch.tensor([j,i], dtype=torch.float, device="cuda") - vp

                angle = torch.arccos((v / torch.linalg.norm(v))[0] * (-1 if vp[0] > w and 0 <= vp[1] < h else 1)) * torch.sign(v[1])
                min_angle = angle if angle < min_angle else min_angle
                max_angle = angle if angle > max_angle else max_angle
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            min_angle = 0
            max_angle = 6.2831

        temp_vals = []
        angles = torch.linspace(min_angle, max_angle, 1000, device="cuda")
        #print(angles)
        vec = torch.stack([-torch.cos(angles), torch.sin(angles)] if vp[0] > w and 0 <= vp[1] < h else [torch.cos(angles), torch.sin(angles)])
        perp_vec = get_perp_line(vec)

        y = torch.tile(torch.arange(h, device="cuda"), (1000,1))
        x = torch.tile(torch.arange(w, device="cuda"), (1000,1))

        y_big = abs(vec[0]) < abs(vec[1])
        x_big = ~y_big

        x_vals = (vp[0] + (vec[0].unsqueeze(1) * ((y - vp[1]) / vec[1].unsqueeze(1)))).long()
        y_vals = (vp[1] + (vec[1].unsqueeze(1) * ((x - vp[0]) / vec[0].unsqueeze(1)))).long()

        x_in = (x_vals < w) & (x_vals >= 0)
        y_in = (y_vals < h) & (y_vals >= 0)

        y_counts = torch.clip(torch.sum(x_in[y_big,:], axis=1), 1)
        x_counts = torch.clip(torch.sum(y_in[x_big,:], axis=1), 1)

        x_vals = torch.clip(x_vals, 0, w-1)
        y_vals = torch.clip(y_vals, 0, h-1)

        y_big_dot_p = torch.abs(torch.mul(perp_vec[0].unsqueeze(1) * dx[y,x_vals] + perp_vec[1].unsqueeze(1) * dy[y,x_vals], x_in))
        x_big_dot_p = torch.abs(torch.mul(perp_vec[0].unsqueeze(1) * dx[y_vals,x] + perp_vec[1].unsqueeze(1) * dy[y_vals,x], y_in))

        y_cumul_dotps = torch.sum(y_big_dot_p[y_big,:], axis=1) / y_counts
        x_cumul_dotps = torch.sum(x_big_dot_p[x_big,:], axis=1) / x_counts

        cat_array = [x for x in (y_cumul_dotps, x_cumul_dotps) if x.shape[0] != 0]
        
        vals = torch.cat(cat_array)
        
        if False:
            f1 = plt.figure(figsize=(6, 6))
            plt.imshow(torch.tile(torch.permute(img.cpu(), (1, 2, 0)), (1,1,3)).numpy(), cmap='gray')
            for i in range(1000):
                vcpu = vals.cpu().numpy()
                if vcpu[i] < 0.4:
                    continue
                color = np.clip(np.array([1,0,0])*vcpu[i], 0, 1)
                if y_big[i] == False:
                    plt.scatter(x[i].cpu().numpy(), y_vals[i].cpu().numpy(), color=color, marker='o', s=(72. / f1.dpi) ** 2)
                else:
                    plt.scatter(x_vals[i].cpu().numpy(), y[i].cpu().numpy(), color=color, marker='o', s=(72. / f1.dpi) ** 2)
                    
            import time
            ti = time.time()
            plt.savefig("debug_out/test"+str(ti)+".jpg")
            plt.close()
        
        
        vals_arr.append(vals)
    return vals_arr
        
def perspective_loss(imgs, gt_imgs, vpts_batch):
    if imgs.shape[1] > 3:
        imgs = imgs[:,:3,:,:]
    if gt_imgs.shape[1] > 3:
        gt_imgs = gt_imgs[:,:3,:,:]
    imgs = rgb_to_grayscale(imgs)
    gt_imgs = rgb_to_grayscale(gt_imgs)

    h, w = imgs.shape[2:]

    sobel = Sobel().cuda()
    
    N = imgs.shape[0]
    
    loss = 0.0
    
    for idx in range(N):
        img = imgs[idx]
        gt_img = gt_imgs[idx]
        vpts = vpts_batch[idx]
        img_loss = 0.0
        
        out_vals = perspective_loss_img(img, vpts, sobel)
        gt_vals = perspective_loss_img(gt_img, vpts, sobel)
        
        
        for idx, val in enumerate(out_vals):
            img_loss += torch.linalg.norm(val - gt_vals[idx])

        img_loss /= (len(out_vals) if len(out_vals) > 0 else 1)
        
        loss += img_loss
    loss /= N
    return loss