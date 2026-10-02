from __future__ import print_function
import logging
import os
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


class Structured3D(data.Dataset):
    """The Structured3D dataset."""

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
        self.root_dir = os.environ.get("DA360_STRUCTURED3D_ROOT", "data/structured3d")
        self.is_training = is_training
        if self.is_training:
            self.read_train_list()
        else:
            self.read_val_list()
        self.w = width
        self.h = height

        self.max_depth_meters = 1024.0
        self.min_depth_meters = 0.5

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

    def read_train_list(self):
        list_file = './datasets/structured3d_train.txt'
        if os.path.exists(list_file):
            self.rgb_depth_list = read_list(list_file)
            if len(self.rgb_depth_list[0]) > 0:
                return
            logging.warning(
                "%s is empty; scanning Structured3D under %s",
                list_file,
                self.root_dir,
            )

        # official invalid samples
        invalid = {'scene_00043': ['1518', '3128', '474', '732', '856'], 'scene_00240': ['384'], 'scene_00335': ['686'], 
                   'scene_00325': ['970753'], 'scene_00339': ['2193'], 'scene_00501': ['1840'], 'scene_00515': ['277475'],
                   'scene_00917': ['188', '501284'], 'scene_00926': ['2290'], 'scene_00936': ['311'], 'scene_00937': ['1955'], 
                   'scene_00543': ['176'], 'scene_00587': ['9914'], 'scene_00703': ['762455', '771712'], 'scene_01151': ['563'], 
                   'scene_00986': ['141'], 'scene_01009': ['3234', '3571'], 'scene_01021': ['689126'], 'scene_01034': ['222021'], 
                   'scene_00728': ['5662'], 'scene_00828': ['607228'], 'scene_00865': ['1026', '1402'], 'scene_00875': ['739214'], 
                   'scene_02484': ['43003'], 'scene_02499': ['1607', '977359'], 'scene_02509': ['687231'], 'scene_02542': ['671853'], 
                   'scene_01759': ['3584', '3588'], 'scene_01772': ['897997'], 'scene_01781': ['335', '878137'], 'scene_01786': ['5837'], 
                   'scene_01155': [], 'scene_01714': [], 'scene_01816': [], 'scene_03398': [], 'scene_01192': [], 'scene_03399': ['337'], 
                   'scene_01221': ['26619'], 'scene_01222': ['273364'], 'scene_01282': ['1917', '2631', '24057'], 'scene_01530': ['577'], 
                   'scene_01036': ['301'], 'scene_01043': ['2193'], 'scene_01104': ['875'], 'scene_01165': ['204'], 'scene_01745': ['342'], 
                   'scene_01400': ['10576'], 'scene_01445': ['3495'], 'scene_01470': ['1413'], 'scene_01670': ['291'], 'scene_01774': ['143'],
        		   'scene_01852': [], 'scene_01778': ['858455'], 'scene_00010': ['846619'], 'scene_00173': ['4722'], 'scene_03376': ['800900'], 
                   'scene_01916': ['2648'], 'scene_01993': ['849'], 'scene_01998': ['54762'], 'scene_02034': ['921879'], 'scene_02040': ['311'], 
                   'scene_02235': ['799012'], 'scene_02274': ['4093'], 'scene_02326': ['836436'], 'scene_02334': ['869673'], 'scene_02357': ['118319'], 
                   'scene_02580': ['724891'], 'scene_02650': ['877946'], 'scene_02659': ['577142'], 'scene_02690': ['586296'], 'scene_02706': ['823368'], 
                   'scene_02788': ['815473'], 'scene_02889': ['848271'], 'scene_03035': ['631066'], 'scene_03120': ['830640'], 'scene_03327': ['315045'],
                   'scene_02046': ['1014', '834'], 'scene_02047': ['934954'], 'scene_02101': ['255228'], 'scene_02172': ['335'], 'scene_02564': ['702502']}
        scenes = sorted(glob(os.path.join(self.root_dir, 'scene_0[0-2]*')))+sorted(glob(os.path.join(self.root_dir, 'scene_03[0-3]*')))
        rgb_lists = []
        depth_lists = []

        for scene in scenes:
            depth_list = sorted(glob(os.path.join(scene, '2D_rendering/*/panorama/[f-s]*/depth.png')))
            no_exist_list = []
            for i in range(len(depth_list)):
                coldlight = depth_list[i].replace('depth', 'rgb_coldlight')
                rawlight = depth_list[i].replace('depth', 'rgb_rawlight')
                warmlight = depth_list[i].replace('depth', 'rgb_warmlight')
                if os.path.exists(coldlight) or os.path.exists(rawlight) or os.path.exists(warmlight):
                    continue
                else:
                    no_exist_list.append(i)
            for i in no_exist_list[::-1]:
                del depth_list[i]
            rgb_list = [dep.replace('depth', 'rgb_rawlight') for dep in depth_list]
            
            if scene.split('/')[-1] in invalid:
                invalid_folders = invalid[scene.split('/')[-1]]
                if len(invalid_folders) > 0:
                    rgb_lists += [os.path.relpath(rgb, self.root_dir)  for rgb in rgb_list if rgb.split('/')[-4] not in invalid_folders]
                    depth_lists += [os.path.relpath(depth, self.root_dir) for depth in depth_list if depth.split('/')[-4] not in invalid_folders]
            else:
                rgb_lists += [os.path.relpath(rgb, self.root_dir)  for rgb in rgb_list]
                depth_lists += [os.path.relpath(depth, self.root_dir) for depth in depth_list]

        write_list(list_file, rgb_lists, depth_lists)
        self.rgb_depth_list = [rgb_lists, depth_lists]

    def read_val_list(self):
        list_file = './datasets/structured3d_val.txt'
        if os.path.exists(list_file):
            self.rgb_depth_list = read_list(list_file)
            if len(self.rgb_depth_list[0]) > 0:
                return
            logging.warning(
                "%s is empty; scanning Structured3D under %s",
                list_file,
                self.root_dir,
            )

        # official invalid samples
        invalid = {'scene_03478': ['2193']}
        scenes = sorted(glob(os.path.join(self.root_dir, 'scene_034*')))
        rgb_lists = []
        depth_lists = []

        for scene in scenes:
            depth_list = sorted(glob(os.path.join(scene, '2D_rendering/*/panorama/full/depth.png')))
            no_exist_list = []
            for i in range(len(depth_list)):
                rawlight = depth_list[i].replace('depth', 'rgb_rawlight')
                if os.path.exists(rawlight) or cv2.imread(rawlight) is None or cv2.imread(depth_list[i], -1) is None:
                    continue
                else:
                    no_exist_list.append(i)
            for i in no_exist_list[::-1]:
                del depth_list[i]
            rgb_list = [dep.replace('depth', 'rgb_rawlight') for dep in depth_list]
            
            if scene.split('/')[-1] in invalid:
                invalid_folders = invalid[scene.split('/')[-1]]
                if len(invalid_folders) > 0:
                    rgb_lists += [os.path.relpath(rgb, self.root_dir)  for rgb in rgb_list if rgb.split('/')[-4] not in invalid_folders]
                    depth_lists += [os.path.relpath(depth, self.root_dir) for depth in depth_list if depth.split('/')[-4] not in invalid_folders]
            else:
                rgb_lists += [os.path.relpath(rgb, self.root_dir)  for rgb in rgb_list]
                depth_lists += [os.path.relpath(depth, self.root_dir) for depth in depth_list]

        write_list(list_file, rgb_lists, depth_lists)
        self.rgb_depth_list = [rgb_lists, depth_lists]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            idx = [i % (len(self.rgb_depth_list[0]) * self.v) for i in idx]
            idx = [i % len(self.rgb_depth_list[0]) for i in idx]
        else:
            idx = idx % (len(self.rgb_depth_list[0]) * self.v)
            idx = idx % len(self.rgb_depth_list[0])

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[0][idx])
        p = random.random()
        if self.is_training:
            if p > 2/3 and os.path.exists(rgb_name.replace("rawlight", "coldlight")):
                rgb_name = rgb_name.replace("rawlight", "coldlight")
            elif p > 1/3 and os.path.exists(rgb_name.replace("rawlight", "warmlight")):
                rgb_name = rgb_name.replace("rawlight", "warmlight")
            else:
                if not os.path.exists(rgb_name):
                    if os.path.exists(rgb_name.replace("rawlight", "coldlight")):
                        rgb_name = rgb_name.replace("rawlight", "coldlight")
                    if os.path.exists(rgb_name.replace("rawlight", "warmlight")):
                        rgb_name = rgb_name.replace("rawlight", "warmlight")

        rgb = cv2.imread(rgb_name)
        if rgb is None:
            rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[1][idx])
        gt_depth = cv2.imread(depth_name, -1)
        if gt_depth is None:
            gt_depth = -np.ones((self.h, self.w), dtype=np.float32)
        else:
            gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
            gt_depth = gt_depth.astype(np.float32)/1000
            gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters
            gt_depth[gt_depth < self.min_depth_meters] = 0

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



