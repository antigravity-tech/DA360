from __future__ import print_function
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import random
from glob import glob

import torch
from torch.utils import data
from torchvision import transforms

from .util import polar_mask, pitch_rotation


def read_list(list_file):
    rgb_list = []
    depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_list.append(line.strip().split(" ")[0])
            depth_list.append(line.strip().split(" ")[1])
    return [rgb_list, depth_list]

def write_list(list_file, rgb_list, depth_list):
    lines = []
    for rgb, depth in zip(rgb_list, depth_list):
        lines.append(" ".join([rgb, depth])+"\n")
    with open(list_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return 0


class TartanAir(data.Dataset):
    """The TartanAir Dataset"""

    def __init__(self, height=518, width=1036, disable_color_augmentation=False, disable_LR_filp_augmentation=False,
                 disable_pitch_rotation_augmentation=False, disable_yaw_rotation_augmentation=False,
                 disable_polar_mask_augmentation=False, is_training=True):
        """
        Args:
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.v = 1
        self.root_dir = os.environ.get("DA360_TARTANAIR_ROOT", "data/tartanair2")
        self.is_training = is_training
        self.read_list()
        self.w = width
        self.h = height

        self.max_depth_meters = 1024.0
        self.min_depth_meters = 0.1

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.pitch_rotation_augmentation = not disable_pitch_rotation_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation
        self.polar_mask_augmentation = not disable_polar_mask_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
                
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __mul__(self, v):
        self.v = v
        return self

    def __len__(self):
        return len(self.rgb_depth_list[0])*self.v

    def read_list(self):
        list_file = './datasets/tartanair.txt'
        if os.path.exists(list_file):
            self.rgb_depth_list = read_list(list_file)
            return
        
        rgb_list = sorted(glob(os.path.join(self.root_dir, '*/Data_hard/P*/*_rgb.jpg')))
        depth_list = sorted(glob(os.path.join(self.root_dir, '*/Data_hard/P*/*_dep.exr')))

        rgb_list = [os.path.relpath(rgb, self.root_dir)  for rgb in rgb_list]
        depth_list = [os.path.relpath(depth, self.root_dir) for depth in depth_list]
        
        assert len(rgb_list) == len(depth_list), "rgb files are inconsistent with depth files"
        write_list(list_file, rgb_list, depth_list)
        self.rgb_depth_list = [rgb_list, depth_list]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            idx = [i % (len(self.rgb_depth_list[0]) * self.v) for i in idx]
            idx = [i % len(self.rgb_depth_list[0]) for i in idx]
        else:
            idx = idx % (len(self.rgb_depth_list[0]) * self.v)
            idx = idx % len(self.rgb_depth_list[0])

        rgb_path = os.path.join(self.root_dir, self.rgb_depth_list[0][idx])
        depth_path = os.path.join(self.root_dir, self.rgb_depth_list[1][idx])
        inputs = {}

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float32)
        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        
        if self.is_training and self.pitch_rotation_augmentation and random.random() > 0.3:
            # random pitch rotation
            angle = (random.random()-0.5)*np.pi/3
            rgb, gt_depth = pitch_rotation(rgb, gt_depth, angle)

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        
        if self.is_training and self.polar_mask_augmentation and random.random() > 0.3:
            rgb = polar_mask(rgb)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > self.min_depth_meters) & (inputs["gt_depth"] <= self.max_depth_meters)
                               & ~torch.isnan(inputs["gt_depth"]))
    
        inputs["gt_disp"] = -torch.ones_like(inputs["gt_depth"])
        inputs["gt_disp"][inputs["val_mask"]] = 1/inputs["gt_depth"][inputs["val_mask"]]

        return inputs



