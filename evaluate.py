from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm

import torch

import networks
from datasets import fetch_val_dataloaders
from metrics import *
from saver import Saver

parser = argparse.ArgumentParser(description="Evaluation of Depth Anything in 360°.")

parser.add_argument("--val_datasets", default=["matterport3d", "stanford2d3d", "metropolis"], nargs="+", 
                    choices=["matterport3d", "stanford2d3d", "metropolis"],  type=str, help="datasets to validate on.")

parser.add_argument("--model_path", type=str, help="path of model to load")

parser.add_argument("--num_workers", type=int, default=2, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--alignment", type=int, default=1, choices=[0, 1, 2], 
                    help="alignment types in evaluation, 0: No alignment; 1: scale alignment; 2: affine (scale+shift) alignment")

parser.add_argument("--save_samples", action="store_true", help="if set, save the depth maps and point clouds")

parser.add_argument('--model_name', type=str, default='DA360')
parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="root directory for metrics and optional visualizations (default: ./results/)",
)

args = parser.parse_args()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.join(ROOT_DIR, "results")


max_depths = {"matterport3d": 10, "stanford2d3d": 10, "metropolis": 100}
NUM_VIZ_SAMPLES = 10


def viz_batch_indices(num_batches, n_viz=NUM_VIZ_SAMPLES):
    """Evenly spaced batch indices; up to ``n_viz`` saves per dataset."""
    if num_batches <= 0:
        return set()
    if num_batches <= n_viz:
        return set(range(num_batches))
    return {i * num_batches // n_viz for i in range(n_viz)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.model_path)
    
    # data
    args.height = model_dict['height']
    args.width = model_dict['width']
    val_loaders = fetch_val_dataloaders(args)
    if 'net' not in model_dict:
        model_dict['net'] = 'DA360'

    # model
    Net = getattr(networks, model_dict['net'])
    model = Net(model_dict['height'], model_dict['width'], dinov2_encoder = model_dict['dinov2_encoder'])
    
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()
    
    results_root = args.results_dir or DEFAULT_RESULTS_ROOT

    for data_name, val_loader in zip(args.val_datasets, val_loaders):
        max_depth = max_depths[data_name]
        evaluator = Evaluator(0, max_depth)
        
        evaluator.reset_eval_metrics()
        dataset_results_dir = os.path.join(results_root, data_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        saver = Saver(dataset_results_dir, sample_subdir=None)
        save_batch_indices = (
            viz_batch_indices(len(val_loader)) if args.save_samples else set()
        )
        
        pbar = tqdm.tqdm(val_loader)
        pbar.set_description("Evaluating "+data_name)
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                equi_inputs = inputs["normalized_rgb"].to(device)
                outputs = model(equi_inputs)
                pred_disp = outputs["pred_disp"].detach().to('cpu')
                gt_disp = inputs["gt_disp"]
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                pred_depth = pred_disp

                for i in range(gt_depth.shape[0]):
                    mask_i = mask[i]
                    pred_disp_i = pred_disp[i]
                    gt_depth_i = gt_depth[i]
                    gt_disp_i = gt_disp[i]
                    
                    if args.alignment==1:
                        scale = torch.mean(gt_disp_i[mask_i&(pred_disp_i>0)])/torch.mean(pred_disp_i[mask_i&(pred_disp_i>0)]).item()
                        shift = 0
                    elif args.alignment==2:
                        scale, shift = compute_scale_and_shift(pred_disp_i, gt_disp_i, mask_i)
                        scale, shift = scale.item(), shift.item()
                    else:
                        scale, shift = 1, 0

                    align_pred_disp_i = scale*pred_disp_i+shift
                    align_pred_disp_i[align_pred_disp_i<1e-4] = 1e-4
                    pred_depth[i] = 1/align_pred_disp_i

                for i in range(gt_depth.shape[0]):
                    evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])

                if batch_idx in save_batch_indices:
                    saver.save_samples(inputs["rgb"], gt_depth, pred_depth, mask, args.model_name)

        evaluator.print(dataset_results_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
