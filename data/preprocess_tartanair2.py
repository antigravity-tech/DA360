# General imports.
import argparse
import glob
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tartanair.reader import TartanAirImageReader

# Image resampling.
from tartanair.image_resampling.image_sampler import SixPlanarTorch
from tartanair.image_resampling.image_sampler.blend_function import BlendBy2ndOrderGradTorch
from tartanair.image_resampling.mvs_utils.camera_models import Equirectangular
from tartanair.image_resampling.mvs_utils.ftensor import FTensor
from tartanair.image_resampling.mvs_utils.shape_struct import ShapeStruct

try:
    from .download_tartanair2 import DEFAULT_ENVS
except ImportError:
    from download_tartanair2 import DEFAULT_ENVS

# NED -> EDN basis change, and a fixed yaw rotation that uprights the six-planar cube.
R_EDN_NED = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
R_NED_EDN = R_EDN_NED.T
R_UPRIGHT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TartanAir cube maps into upright equirectangular frames.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TARTANAIR_DATA_ROOT", "raw"),
        help="Input TartanAir root (default: raw, or $TARTANAIR_DATA_ROOT)",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("TARTANAIR_OUTPUT_ROOT", "data"),
        help="Output root for upright equirect frames (default: data, or $TARTANAIR_OUTPUT_ROOT)",
    )
    parser.add_argument(
        "--env",
        nargs="+",
        default=DEFAULT_ENVS,
        help="Environment name(s) to process (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        default="hard",
        choices=["easy", "hard"],
        help="Difficulty to process (default: hard)",
    )
    parser.add_argument(
        "--trajectory",
        nargs="+",
        default=None,
        help="Optional trajectory id(s), e.g. P000 P001. Default: all under the env/difficulty",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Equirectangular height (default: 640)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Equirectangular width (default: 1280)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for resampling (default: cuda:0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    return parser.parse_args()


def get_directories(path="."):
    """Return subdirectory names under path."""
    try:
        return [
            item
            for item in os.listdir(path)
            if os.path.isdir(os.path.join(path, item))
        ]
    except FileNotFoundError:
        print(f"路径不存在: {path}")
        return []
    except PermissionError:
        print(f"没有权限访问: {path}")
        return []


def kitti_colormap(disparity, maxval=-1):
    """Reproduce KITTI fake colormap. Returns HxWx3 uint8."""
    if maxval < 0:
        maxval = np.max(disparity)

    colormap = np.asarray(
        [
            [0, 0, 0, 114],
            [0, 0, 1, 185],
            [1, 0, 0, 114],
            [1, 0, 1, 174],
            [0, 1, 0, 114],
            [0, 1, 1, 185],
            [1, 1, 0, 114],
            [1, 1, 1, 0],
        ]
    )
    weights = np.asarray(
        [
            8.771929824561404,
            5.405405405405405,
            8.771929824561404,
            5.747126436781609,
            8.771929824561404,
            5.405405405405405,
            8.771929824561404,
            0,
        ]
    )
    cumsum = np.asarray(
        [0, 0.114, 0.299, 0.413, 0.587, 0.701, 0.8859999999999999, 0.9999999999999999]
    )

    colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
    values = np.expand_dims(np.minimum(np.maximum(disparity / maxval, 0.0), 1.0), -1)
    bins = np.repeat(
        np.repeat(np.expand_dims(np.expand_dims(cumsum, axis=0), axis=0), disparity.shape[1], axis=1),
        disparity.shape[0],
        axis=0,
    )
    diffs = np.where(
        (np.repeat(values, 8, axis=-1) - bins) > 0,
        -1000,
        (np.repeat(values, 8, axis=-1) - bins),
    )
    index = np.argmax(diffs, axis=-1) - 1
    w = 1 - (values[:, :, 0] - cumsum[index]) * np.asarray(weights)[index]

    colored_disp[:, :, 2] = w * colormap[index][:, :, 0] + (1.0 - w) * colormap[index + 1][:, :, 0]
    colored_disp[:, :, 1] = w * colormap[index][:, :, 1] + (1.0 - w) * colormap[index + 1][:, :, 1]
    colored_disp[:, :, 0] = w * colormap[index][:, :, 2] + (1.0 - w) * colormap[index + 1][:, :, 2]

    return (colored_disp * np.expand_dims((disparity > 0), -1) * 255).astype(np.uint8)


def check_image(image):
    """Return False for low-contrast / near-black / near-white frames."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dx = np.abs(np.diff(image, axis=1))
    dy = np.abs(np.diff(image, axis=0))
    mv = (np.mean(dx) + np.mean(dy)) / image.max()
    mean_value = image.mean()
    return not (mv < 0.2 or mean_value < 20 or mean_value > 235)


def check_depth(depth):
    h, w = depth.shape
    hs = h//2+3
    num_th = int(h*w/100)

    max_depth = max(50, depth.max()*0.95)

    num = (depth[hs:]>max_depth).sum()
    return not (num > num_th)


def ned_pose_to_edn_upright(pose_ned):
    """
    Pose of the upright equirect after six-planar resampling.

    Input / output encoding (same as TartanAir pose_*.txt):
        tx ty tz qx qy qz qw

    Resampling applies an upright correction in the original camera frame:
        R_new_to_raw_ned = (R_upright @ R_w_ned).T

    Chaining with the original camera-to-world (NED):
        X_w_ned = R_w_ned @ X_raw + t_ned
        X_raw   = R_new_to_raw_ned @ X_new
    gives:
        X_w_ned = R_w_ned @ (R_upright @ R_w_ned).T @ X_new + t_ned

    Because R_w_ned @ (R_upright @ R_w_ned).T = R_upright.T, the upright
    equirect has a fixed world orientation (R_upright.T in NED); only the
    camera center moves with the trajectory.

    Output pose in EDN:
        t_edn       = R_edn_ned @ t_ned
        R_w_new_edn = R_edn_ned @ R_upright.T @ R_ned_edn
    """
    p_ned = np.asarray(pose_ned[:3], dtype=np.float64)
    q_ned = np.asarray(pose_ned[3:7], dtype=np.float64)  # xyzw
    R_w_ned = Rotation.from_quat(q_ned).as_matrix()

    R_raw_new = (R_UPRIGHT @ R_w_ned).T 
    R_w_new_ned = R_w_ned @ R_raw_new

    p_edn = R_EDN_NED @ p_ned
    R_w_new_edn = R_EDN_NED @ R_w_new_ned @ R_NED_EDN
    q_edn = Rotation.from_matrix(R_w_new_edn).as_quat()  # xyzw
    return np.concatenate([p_edn, q_edn])


def write_pose_txt(path, poses):
    """Write poses with the same encoding as TartanAir pose files."""
    with open(path, "w", encoding="utf-8") as f:
        for pose in poses:
            line = " ".join(f"{v:.18e}" for v in pose)
            f.write(line + "\n")


def sample_image_worker(argslist):
    (
        frame_ix,
        q_w_raw,
        front_rgb_file,
        new_data_dir_path,
        height,
        width,
        device,
    ) = argslist

    R_w_raw_ned = Rotation.from_quat(q_w_raw).as_matrix()
    R_raw_new = R_EDN_NED @ (R_UPRIGHT @ R_w_raw_ned).T @ R_NED_EDN
    R_raw_new = FTensor(R_raw_new, f0="raw", f1="fisheye", rotation=True)

    new_cam_model_object = Equirectangular(shape_struct=ShapeStruct(H=height, W=width))
    sampler = SixPlanarTorch(new_cam_model_object, R_raw_new)
    sampler.device = device

    reader = TartanAirImageReader()
    modality_to_reader = {
        "image": reader.read_bgr,
        "depth": reader.read_dist,
        "seg": reader.read_seg,
    }
    modality_to_interpolation = {"image": "linear", "seg": "nearest", "depth": "blend"}
    blend_func = BlendBy2ndOrderGradTorch(0.01)

    raw_images = {}
    raw_depths = {}
    front_depth_file = front_rgb_file.replace("image", "depth").replace(".png", "_depth.png")
    for side in ["front", "back", "left", "right", "top", "bottom"]:
        raw_images[side] = modality_to_reader["image"](front_rgb_file.replace("front", side))
        raw_depths[side] = modality_to_reader["depth"](front_depth_file.replace("front", side))

    new_image, _ = sampler(
        raw_images,
        interpolation=modality_to_interpolation["image"],
        invalid_pixel_value=0,
    )
    new_depth, _ = sampler.blend_interpolation(
        raw_depths, blend_func, invalid_pixel_value=0
    )
    new_depth[new_depth > 1024] = 1024

    if check_depth(new_depth) and check_image(new_image):
        new_image_path = os.path.join(new_data_dir_path, f"{frame_ix:06d}_rgb.jpg")
        new_depth_path = os.path.join(new_data_dir_path, f"{frame_ix:06d}_dep.exr")
        new_disp_path = os.path.join(new_data_dir_path, f"{frame_ix:06d}_disp.jpg")
        new_disp = kitti_colormap(new_depth.min() / new_depth)
        cv2.imwrite(new_image_path, new_image)
        cv2.imwrite(new_depth_path, new_depth)
        cv2.imwrite(new_disp_path, new_disp)

    return 0


def process_trajectory(
    tartanair_data_root,
    result_data_root,
    env_name,
    difficulty_dir,
    trajectory,
    height,
    width,
    device,
    num_workers,
):
    new_data_dir_path = os.path.join(result_data_root, env_name, difficulty_dir, trajectory)
    os.makedirs(new_data_dir_path, exist_ok=True)

    front_rgb_list = sorted(
        glob.glob(
            os.path.join(
                tartanair_data_root,
                env_name,
                difficulty_dir,
                trajectory,
                "image_lcam_front",
                "*.png",
            )
        )
    )
    pose_fp = os.path.join(
        tartanair_data_root, env_name, difficulty_dir, trajectory, "pose_lcam_front.txt"
    )
    if not front_rgb_list:
        print(f"Skip {env_name}/{difficulty_dir}/{trajectory}: no front images")
        return
    if not os.path.isfile(pose_fp):
        print(f"Skip {env_name}/{difficulty_dir}/{trajectory}: missing {pose_fp}")
        return

    poses_ned = np.loadtxt(pose_fp)
    if poses_ned.ndim == 1:
        poses_ned = poses_ned.reshape(1, -1)
    num_frames = len(front_rgb_list)
    if len(poses_ned) != num_frames:
        print(
            f"Warning: pose count ({len(poses_ned)}) != image count ({num_frames}) "
            f"in {env_name}/{difficulty_dir}/{trajectory}"
        )
        num_frames = min(len(poses_ned), num_frames)

    # Upright equirect poses in EDN, same tx ty tz qx qy qz qw encoding.
    poses_edn = np.stack([ned_pose_to_edn_upright(poses_ned[i]) for i in range(num_frames)])
    out_pose_path = os.path.join(new_data_dir_path, "pose.txt")
    write_pose_txt(out_pose_path, poses_edn)
    print(f"Wrote upright EDN poses: {out_pose_path} ({num_frames} frames)")

    sample_image_worker_args = [
        [
            frame_ix,
            poses_ned[frame_ix][3:7],
            front_rgb_list[frame_ix],
            new_data_dir_path,
            height,
            width,
            device,
        ]
        for frame_ix in range(num_frames)
    ]

    if num_workers <= 1:
        print(f"Running sequentially: {env_name}/{difficulty_dir}/{trajectory}")
        for arglist in sample_image_worker_args:
            sample_image_worker(arglist)
    else:
        print(
            f"Running in parallel with {num_workers} workers: "
            f"{env_name}/{difficulty_dir}/{trajectory}"
        )
        try:
            with Pool(num_workers) as pool:
                pool.map(sample_image_worker, sample_image_worker_args)
        except KeyboardInterrupt:
            raise SystemExit(130)


def main():
    args = parse_args()

    # Delayed import of tartanair-side packages that may pull opencv incorrectly
    # is already handled by importing cv2 first above.
    tartanair_data_root = os.path.abspath(args.data_root)
    result_data_root = os.path.abspath(args.output_root)
    difficulty_dir = f"Data_{args.difficulty}"
    envs = args.env

    os.makedirs(result_data_root, exist_ok=True)

    print(f"data-root   : {tartanair_data_root}")
    print(f"output-root : {result_data_root}")
    print(f"env         : {envs}")
    print(f"difficulty  : {args.difficulty} ({difficulty_dir})")
    print(f"trajectory  : {args.trajectory or 'all'}")
    print(f"equirect    : {args.width}x{args.height}")
    print(f"device      : {args.device}")
    print(f"num-workers : {args.num_workers}")

    for env_name in envs:
        traj_root = os.path.join(tartanair_data_root, env_name, difficulty_dir)
        if args.trajectory is None:
            trajectories = sorted(get_directories(traj_root))
        else:
            trajectories = args.trajectory

        if not trajectories:
            print(f"No trajectories under {traj_root}")
            continue
            
        for trajectory in trajectories:
            process_trajectory(
                tartanair_data_root=tartanair_data_root,
                result_data_root=result_data_root,
                env_name=env_name,
                difficulty_dir=difficulty_dir,
                trajectory=trajectory,
                height=args.height,
                width=args.width,
                device=args.device,
                num_workers=args.num_workers,
            )


if __name__ == "__main__":
    main()
