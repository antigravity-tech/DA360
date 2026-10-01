import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
import matplotlib.pyplot as plt

import cv2
import open3d as o3d


def kitti_colormap(disparity, maxval=-1):
	"""
	A utility function to reproduce KITTI fake colormap
	Arguments:
	  - disparity: numpy float32 array of dimension HxW
	  - maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)
	
	Returns a numpy uint8 array of shape HxWx3.
	"""
	if maxval < 0:
		maxval = np.max(disparity)

	colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
	weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,8.771929824561404,5.405405405405405,8.771929824561404,0])
	cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

	colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
	values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
	bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disparity.shape[1], axis=1), disparity.shape[0], axis=0)
	diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
	index = np.argmax(diffs, axis=-1)-1

	w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]


	colored_disp[:,:,2] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
	colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
	colored_disp[:,:,0] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

	return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    def save_as_point_cloud(self, depth, rgb, path, mask=None):
        h, w = depth.shape
        Theta = np.pi - np.arange(h).reshape(h, 1) * np.pi / h - np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)
        
        if mask is None:
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            R = rgb[:, :, 0].flatten()
            G = rgb[:, :, 1].flatten()
            B = rgb[:, :, 2].flatten()
        else:
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]
            R = rgb[:, :, 0][mask]
            G = rgb[:, :, 1][mask]
            B = rgb[:, :, 2][mask]

        XYZ = np.stack([X, Y, Z], axis=1)
        RGB = np.stack([R, G, B], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)

    def save_samples(self, rgbs, gt_depths, pred_depths, depth_masks=None, model_name=''):
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.cpu().numpy()
        gt_depths = gt_depths.cpu().numpy()
        if depth_masks is None:
            depth_masks = gt_depths != 0
        else:
            depth_masks = depth_masks.cpu().numpy()

        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1

            gt_depth = gt_depths[i][0]
            min_depth = gt_depth[depth_masks[i][0]].min()/5
            pred_depth = depth_preds[i][0]
            pred_depth[pred_depth<min_depth]=min_depth

            # Save raw depth arrays as npy
            path = os.path.join(self.save_dir, '%04d' % (self.idx) + '_depth_gt.npy')
            np.save(path, gt_depth)
            path = os.path.join(self.save_dir, '%04d' % (self.idx) + f'_depth_pred_{model_name}.npy')
            np.save(path, pred_depth)

            depth = cv2.vconcat([gt_depth, pred_depth])
            depth = depth/depth[depth > 0].max()
            disp = depth
            disp[depth>0] = 1/depth[depth>0]
            disp[depth<=0] = 0
            disp = kitti_colormap(disp)
            h = disp.shape[0]

            gt_disp = disp[:h//2]
            gt_disp[~depth_masks[i][0]] = [230, 230, 230]
            path = os.path.join(self.save_dir, '%04d' % (self.idx)+f'_depth_gt.jpg')
            cv2.imwrite(path, gt_disp)

            pred_disp = disp[h//2:]
            path = os.path.join(self.save_dir, '%04d' % (self.idx)+f'_depth_pred_{model_name}.jpg')
            cv2.imwrite(path, pred_disp)

            path = os.path.join(self.save_dir, '%04d'%(self.idx)+f'_pc_pred_{model_name}.ply')
            self.save_as_point_cloud(pred_depth, rgbs[i], path, pred_depth<200)

            path = os.path.join(self.save_dir, '%04d'%(self.idx)+'_pc_gt.ply')
            self.save_as_point_cloud(gt_depth, rgbs[i], path, depth_masks[i][0])
            
            rgb = (rgbs[i] * 255).astype(np.uint8)
            path = os.path.join(self.save_dir, '%04d'%(self.idx)+'_rgb.jpg')
            cv2.imwrite(path, rgb[:, :, ::-1])
            
    def save_pred_samples(self, rgb, pred_depth, name, model_name=''):
        """
        Saves samples
        """
        rgb = rgb.cpu().numpy().transpose(0, 2, 3, 1)[0]
        pred_depth = pred_depth.cpu().numpy()[0, 0]

        # Save raw depth array as npy
        path = os.path.join(self.save_dir[:-7], name + f'_depth_pred_{model_name}.npy')
        np.save(path, pred_depth)

        depth = pred_depth.copy()
        depth = depth/depth[depth > 0].max()
        disp = depth
        disp[depth>0] = 1/depth[depth>0]
        disp[depth<=0] = 0
        disp = kitti_colormap(disp)
        path = os.path.join(self.save_dir[:-7], name+f'_depth_pred_{model_name}.jpg')
        cv2.imwrite(path, disp)

        rgb1 = (rgb * 255).astype(np.uint8)
        path = os.path.join(self.save_dir[:-7], name+'_rgb.jpg')
        cv2.imwrite(path, rgb1[:, :, ::-1])

        path = os.path.join(self.save_dir[:-7], name + f'_depth_pred_{model_name}.exr')
        cv2.imwrite(path, pred_depth)
        
        path = os.path.join(self.save_dir[:-7], name + f'_pc_pred_{model_name}.ply')
        self.save_as_point_cloud(pred_depth, rgb, path, pred_depth<200)
