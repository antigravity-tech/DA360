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
import torch.nn.functional as F

from .util import polar_mask, pitch_rotation


def cassini2Equirec(cassini):
    if cassini.ndim == 2:
        cassini = np.expand_dims(cassini, axis=-1)
        source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3)
    elif cassini.ndim == 3:
        source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3)
    else:
        source_image = cassini

    erp_h = source_image.shape[-1]
    erp_w = source_image.shape[-2]

    theta_erp_start = np.pi - (np.pi / erp_w)
    theta_erp_end = -np.pi
    theta_erp_step = 2 * np.pi / erp_w
    theta_erp_range = np.arange(theta_erp_start, theta_erp_end, -theta_erp_step)
    theta_erp_map = np.array([theta_erp_range for i in range(erp_h)]).astype(np.float32)

    phi_erp_start = 0.5 * np.pi - (0.5 * np.pi / erp_h)
    phi_erp_end = -0.5 * np.pi
    phi_erp_step = np.pi / erp_h
    phi_erp_range = np.arange(phi_erp_start, phi_erp_end, -phi_erp_step)
    phi_erp_map = np.array([phi_erp_range for j in range(erp_w)]).astype(np.float32).T

    theta_cassini_map = np.arctan2(np.tan(phi_erp_map), np.cos(theta_erp_map))
    phi_cassini_map = np.arcsin(np.cos(phi_erp_map) * np.sin(theta_erp_map))

    grid_x = torch.FloatTensor(np.clip(-phi_cassini_map / (0.5 * np.pi), -1, 1)).unsqueeze(-1)
    grid_y = torch.FloatTensor(np.clip(-theta_cassini_map / np.pi, -1, 1)).unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0).repeat_interleave(source_image.shape[0], dim=0)

    sampled_image = F.grid_sample(source_image, grid, mode='bilinear', align_corners=True, padding_mode='border')  # 1, ch, self.output_h, self.output_w

    if cassini.ndim == 3:
        erp = sampled_image.transpose(1, 3).transpose(1, 2).data.numpy()[0].astype(cassini.dtype)
        return erp.squeeze()
    else:
        erp = sampled_image.numpy()
        return erp.squeeze(1)


class Deep360(data.Dataset):
    """The Deep360 Dataset"""

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
        self.root_dir = os.environ.get("DA360_DEEP360_ROOT", "data/deep360")
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
        """
        rgb_list = sorted(glob(os.path.join(self.root_dir, 'ep*_500frames/*/rgb/*_rgb.png')))
        depth_list = sorted(glob(os.path.join(self.root_dir, 'ep*_500frames/*/depth/*_depth.exr')))
        assert len(rgb_list) == len(depth_list), "rgb files are inconsistent with depth files"
        self.rgb_depth_list = [rgb_list, depth_list]
        """
        if self.is_training:
            rgb_list = sorted(glob(os.path.join(self.root_dir, 'ep[1-5]_500frames/*/rgb/*_12_rgb1.png')))
            depth_list = sorted(glob(os.path.join(self.root_dir, 'ep[1-5]_500frames/*/depth/*_depth.npz')))
        else:
            rgb_list = sorted(glob(os.path.join(self.root_dir, 'ep6_500frames/*/rgb/*_12_rgb1.png')))
            depth_list = sorted(glob(os.path.join(self.root_dir, 'ep6_500frames/*/depth/*_depth.npz')))
        
        assert len(rgb_list) == len(depth_list), "rgb files are inconsistent with depth files"
        self.rgb_depth_list = [rgb_list, depth_list]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            idx = [i % (len(self.rgb_depth_list[0]) * self.v) for i in idx]
            idx = [i % len(self.rgb_depth_list[0]) for i in idx]
        else:
            idx = idx % (len(self.rgb_depth_list[0]) * self.v)
            idx = idx % len(self.rgb_depth_list[0])

        inputs = {}
        
        rgb_name = os.path.join(self.rgb_depth_list[0][idx])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cassini2Equirec(rgb)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        
        depth_name = os.path.join(self.rgb_depth_list[1][idx])
        gt_depth = np.load(depth_name)['arr_0'].astype(np.float32)
        gt_depth = cassini2Equirec(gt_depth)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float32)
        #gt_depth[gt_depth > 200] = -1
        #gt_depth[-gt_depth.shape[0]//2:] = -1
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

