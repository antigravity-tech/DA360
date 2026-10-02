from __future__ import absolute_import, division, print_function
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from glob import glob
import argparse
import tqdm
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import time

import networks
from datasets import Real
from saver import Saver

parser = argparse.ArgumentParser(description="Test Depth Anything in 360°")

parser.add_argument("--path", default="./data/images", type=str, help="path to test on.")

parser.add_argument("--model_path", type=str, help="path of model to load")
parser.add_argument("--net", type=str, default=None, help="model to use")
parser.add_argument('--model_name', type=str, default='DA360')
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="where to save predictions (default: ./results/images/)",
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_IMAGES = os.path.join(ROOT_DIR, "results", "images")

args = parser.parse_args()


def sort_key(s):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.model_path)
    
    # data
    if 'net' not in model_dict:
        model_dict['net'] = 'DA360'
    if 'dinov2_encoder' not in model_dict:
        model_dict['dinov2_encoder'] = 'vits'
    if 'height' not in model_dict:
        model_dict['height'] = 518
    if 'width' not in model_dict:
        model_dict['width'] = 1036
    if args.net:
        model_dict['net'] = args.net
    
    # model
    Net = getattr(networks, model_dict['net'])
    model = Net(model_dict['height'], model_dict['width'], dinov2_encoder = model_dict['dinov2_encoder'])
    
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()
        
    args.height = model_dict['height']
    args.width = model_dict['width']
    dataset = Real(args.height, args.width)
    dataset.root_dir = args.path
    rgb_list = glob(os.path.join(dataset.root_dir, '*.png')) + glob(os.path.join(dataset.root_dir, '*.jpg'))
    rgb_list = sorted(rgb_list, key=sort_key)
    dataset.rgb_list = rgb_list
    
    data_loader = data.DataLoader(dataset, 1, False, pin_memory=True, drop_last=False)
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description("Testing")

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model_path))[0]
    output_dir = args.output_dir or DEFAULT_RESULTS_IMAGES
    os.makedirs(output_dir, exist_ok=True)
    saver = Saver(output_dir, sample_subdir=None)

    for idx, inputs in enumerate(pbar):
        equi_inputs = inputs["normalized_rgb"].to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model(equi_inputs)
            end = time.time()
        pred_disp = outputs["pred_disp"].detach().cpu()

        pred_depth = 1/pred_disp
        pred_depth = pred_depth/pred_depth.min()

        name = os.path.basename(rgb_list[idx])[:-4]

        saver.save_pred_samples(inputs["rgb"], pred_depth, name, model_name)


if __name__ == "__main__":
    main()
