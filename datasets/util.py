import numpy as np
from scipy.ndimage import map_coordinates
import torch 
import torch.nn as nn
import torch.nn.functional as F
import cv2
from struct import unpack
import os
import random


def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]
        
    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([pad_u, e_img, pad_d], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([pad_l, e_img, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y+1, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img


def pitch_rotation(rgb, depth, angle=0.0):
    h, w = depth.shape
    i_dtype = rgb.dtype
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    theta_y = (1 - (v + 0.5)/h)*np.pi # the angle of the 3D vector to Y-axis
    phi_zx = ((u + 0.5)/w - 0.5)*2*np.pi # the angle of the mapped 2D vector on the Z-X plane to Z-axis
        
    x = np.sin(theta_y)*np.sin(phi_zx)
    y = np.cos(theta_y)
    z = np.sin(theta_y)*np.cos(phi_zx)
        
    c, s = np.cos(angle), np.sin(angle)
        
    r_x = x
    r_y = c*y - s*z
    r_z = s*y + c*z
        
    r_theta_y = np.arccos(r_y)
    r_phi_zx = np.arctan2(r_x, r_z)
        
    r_v = (1-r_theta_y/np.pi)*h-0.5
    r_u = (r_phi_zx/2/np.pi+0.5)*w-0.5

    grid = torch.tensor(np.stack([2 * r_u / w - 1, 2 * r_v / h - 1], axis=2)).unsqueeze(0)

    rgb = torch.FloatTensor(rgb).unsqueeze(0).transpose(1, 3).transpose(2, 3)
    depth = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)

    rgb = F.grid_sample(rgb, grid, mode='bilinear', align_corners=False, padding_mode='border').squeeze().permute([1, 2, 0]).numpy().astype(i_dtype)
    depth = F.grid_sample(depth, grid, mode='nearest', align_corners=False, padding_mode='border').squeeze().numpy()

    return rgb, depth


def polar_mask(rgb):
    h = rgb.shape[0]
    # top 
    if random.random()>0.5:
        crop_h = max(int(0.15*h*random.random()), 5)
        r = random.random()**2
        rgb[:crop_h] = r*rgb[:crop_h]+(1-r)*np.mean(rgb[:crop_h], axis=(0, 1))
    # bottom 
    if random.random()>0.5:
        crop_h = max(int(0.15*h*random.random()), 5)
        r = random.random()**2
        rgb[-crop_h:] = r*rgb[-crop_h:]+(1-r)*np.mean(rgb[-crop_h:], axis=(0, 1))
    return rgb

