# Enhancing Diffusion Models with 3D Perspective Geometry Constraints

Rishi Upadhyay, Howard Zhang, Yunhao Ba, Ethan Yang, Blake Gella, Sicheng Jiang, Alex Wong, Achuta Kadambi.

| [Project Page](https://visual.ee.ucla.edu/diffusionperspective.htm/) | [Paper](https://arxiv.org/abs/2312.00944) | 


This repository provides code for SIGGRAPH ASIA 2023 paper, Enhancing Diffusion Models with 3D Perspective Geometry Constraints. This code is built on top of [this repo](https://github.com/justinpinkney/stable-diffusion) which is itself built on top of the [original training repo](https://github.com/pesser/stable-diffusion) for Stable Diffusion.

The key addition in this version of the code is the perspective loss as defined in the paper. Code for this loss can be found in ```ldm/modules/losses/perspective.py```

## Perspective Loss

If you are only interested in the code for the perspective loss itself, this can be found in: ```ldm/modules/losses/perspective.py```

The function is named ```perspective_loss(imgs, gt_imgs, vpts_batch)```. The three inputs are:
* imgs: [B, H, W, 3] -- predicted images from the diffusion model
* gt_imgs: [B, H, W, 3] -- ground truth images
* vpts_batch: [B, 3, 2] -- vanishing points (in image coordinates) for each image. If an image has fewer than 3 vanishing points, the extras will be set to [0.0,0.0] and will be ignored.

The output of this function will be a single value equal to L_persp from the paper.

## Fine tuning

This code is setup to fine-tune the StableDiffusion v2 model with the perspective loss. In the paper, we did this training with the Holicity dataset, however any dataset of images and vanishing points will work.

### Data 

Data should be placed in the /data/[dataset name]/. The file structure below this can be modified, but the code as it is right now requires a few things:


* captions_depth.json -- A json file containing a key:value pair for every image in the dataset. The key should be the **full sub-path** to the image and the value should be a caption for this image to be used an input conditioning. (e.g. if the image is at /data/imagenet/day_0/000.png, the key would be "day_0/000.png"). For our paper, these were obtained via BLIP.
* vpts_depth.json -- A json file contain a key:value pair for every image in the dataset. The key should be the same as the captions file, and the value should be a array of vanishing points for this image.
* Image files -- There should be image files corresponding to every key in the captions and vpts files
* Depth files -- There should be a depth file (in .npz format) corresponding to every image. This file should have either the same name as the image file or have the word "imag" in the filename replaced by "dpth".

The path to each of these sets of files should be set in the config file being used in ```/configs```. Code for processing the input dataset is in ```ldm/data/simple.py```.

**Holicity Dataset**: For the Holicity dataset, we have provided the captions_depth and vpts_depth files, assuming the folder structure of the Holicity dataset is not changed. These are located in ```data/holicity/```.

### Configs

Config file options can be found in ```/configs```. We provide a pre-set config that works well for the perspective task in ```configs/vanishing_point_depth.yaml```. We cover some of the key values to be set/modified:

* model.params -- this section contains the weight to be applied to the perspective loss (perspective_weight) as well as defines whether we condition our model on a depth image (depth_cond). The depth_stage_config section selects the depth model to be used if depth maps are not provided.
* model.params.unet_config and model.params.first_stage_config -- these sections contain architectural details. The current values are designed to work with **Stable Diffusion v2** depth-conditioned checkpoints. They may need to be modified for other checkpoints or architectures.
* data -- this contains the path to dataset files as discussed above

### Model Checkpoints

Model checkpoints should be placed in ```/checkpoints``` at the top level. Both stable diffusion and depth model checkpoints can be placed here. For the paper, we used StableDiffusion v2 from [here](https://huggingface.co/stabilityai/stable-diffusion-2-depth) and [MiDaS](https://github.com/isl-org/MiDaS).

A pre-trained model checkpoint used in our paper can be found here:  https://drive.google.com/file/d/12L4oy5Y8Dk5ESuMix-KdDQgUGaEO4oUE.

### Running the Code

Once the data and checkpoints have been placed in the correct folder, the code can be run with the following command:

```
python main.py \
	-t \
	--base configs/[config file] \
	--gpus 0, 1 \
	--finetune_from checkpoints/[ckpt file] \
	--scale_lr False \
	--depth_cond
```

with the config and checkpoint files replaced with your paths. This code will automatically save checkpoints every 1000 steps.

### Generating Images

Once a model has been trained and the checkpoint saved, images can be generated using the ```depth2img.py``` script. The following command will create an image:

```
python depth2img.py \
	--prompt "a city street with cars and pedestrians" \
	--H 512 --W 512 \
	--n_samples 1 \
	--config configs/stable-diffusion/vanishing_point_depth.yaml \
	--ckpt checkpoints/[path to chkpt] \
	--depth [path to depth map] \
	--outdir ./image_renders 
```

## Depth Models

For the second part of our paper, evaluating the models with downstream tasks, we used two depth estimation models: PixelFormer and DPT. 

### PixelFormer

Code to train PixelFormer can be found on the original repository [here](https://github.com/ashutosh1807/PixelFormer).

### DPT

The DPT model is from [this paper](https://github.com/isl-org/DPT/). There is no official training code, so we have provided our own implementation of DPT training code in the ```depth_models``` folder. To fine-tune, first prepare the dataset:

The depth dataset should be in a folder with two sub-folders: ```img/``` and ```depth/```. The img folder should contain images in .jpg format while the depth maps should be in .npy format with corresponding names.

After the dataset is in place, edit the ```params =  {...}``` variable in ```train_DPT.py``` to point to this folder along with any validation and test folders. Training can then be run by calling:

```
python train_DPT.py
```

## Citation

If any of this code is helpful, please cite our paper:

```
@article{upadhyay2023enhancing,
  author = {Upadhyay, Rishi and Zhang, Howard and Ba, Yunhao and Yang, Ethan and Gella, Blake and Jiang, Sicheng and Wong, Alex and Kadambi, Achuta},
  title = {Enhancing Diffusion Models with 3D Perspective Geometry Constraints},
  year = {2023},
  issue_date = {December 2023},
  volume = {42},
  number = {6},
  doi = {10.1145/3618389},
  journal = {ACM Trans. Graph.},
  month = {dec},
  articleno = {237},
  numpages = {15},
}
```

## Acknowledgements

This code is heavily based upon [Justin Pinkney's Implementation](https://github.com/justinpinkney/stable-diffusion) and the [original Stable Diffusion Implementation](https://github.com/pesser/stable-diffusion). We thank them for open-sourcing their code!
