from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset

def make_multi_folder_data(paths, caption_files=None, vp_files=None, depth_path=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths
    if caption_files is not None and vp_files is not None and depth_path is not None:
        datasets = [FolderData(p, caption_file=c, vp_file=v, depth=depth_path[0], **kwargs) for (p, c, v) in zip(paths, caption_files, vp_files)]
    elif caption_files is not None and vp_files is not None:
        datasets = [FolderData(p, caption_file=c, vp_file=v, **kwargs) for (p, c, v) in zip(paths, caption_files, vp_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)
    
def rearr(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        vp_file=None,
        depth=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=True,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None
            
        if vp_file is not None:
            with open(vp_file, "rt") as f:
                vps_data = json.load(f)
                self.vps_data = vps_data
                
        self.depth_path = depth

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(list(self.root_dir.rglob(f"*.{e}")))
        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        
        self.tform2 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Lambda(rearr)])
        self.flip = transforms.RandomHorizontalFlip(1.0)
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms


    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            chosen_name_only = chosen[:chosen.find(".jpg")]

            caption = self.captions.get(chosen, None)
            vps_dat = self.vps_data.get(chosen, None)

            if vps_dat is None:
                vps_dat = self.vps_data.get(chosen_name_only, None)
                if vps_dat is None:
                    vps_dat = [[256.,512.],[256.,0.]]

            vps = torch.tensor(vps_dat)
            if caption is None:
                caption = self.default_caption
                
            if self.depth_path is not None:
                depth_fname = chosen_name_only.replace("imag", "dpth") + ".npz"
                depth_data = np.load(self.depth_path+depth_fname)
                depth_data = torch.tensor(depth_data["depth"][:,:,0])[None, None,:,:]
                depth_data = torch.nn.functional.interpolate(
                    depth_data,
                    size=(64,64),
                    mode="bilinear",
                    align_corners=False
                )
                depth_data[depth_data > 0] = 1 / (depth_data[depth_data > 0])
                depth_min, depth_max = torch.amin(depth_data, dim=[1, 2, 3], keepdim=True), torch.amax(depth_data, dim=[1, 2, 3], keepdim=True)

                depth_data = 2. * (depth_data - depth_min) / (depth_max - depth_min + 0.001) - 1.
                data["depth"] = depth_data
                
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename)
        im, flip = self.process_im(im)
        data["image"] = im

        if flip:
            data["depth"] = self.flip(data["depth"])
        

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption
            
        if flip:
            orig_x = vps[:,0].clone()
            vps[:,0] = 512 - orig_x

        if len(vps) < 3:
            new_vps = torch.zeros((3-len(vps),2))
            vps = torch.cat([vps,new_vps], axis=0)

        data["vps"] = vps
        
        if self.postprocess is not None:
            data = self.postprocess(data)
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        im = self.tform(im)
        #im.save("preflip"+str(index)+".png")
        flip = False
        if np.random.random() < 0.5:
            im = self.flip(im)
            flip = True
        #im.save("postflip"+str(index)+".png")
        return self.tform2(im), flip

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    tform = transforms.Compose(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]
