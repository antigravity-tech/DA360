import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import argparse
import socket
import textwrap
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from urllib.parse import quote

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from torchvision import transforms

import networks
from checkpoint import checkpoint_meta, load_checkpoint, state_dict_from_checkpoint
from saver import Saver, depth_rgb_to_point_cloud, kitti_colormap

DEFAULT_MODEL = "./checkpoints/DA360_large.pth"
CACHE_DIR = "cache"
CACHE_DIR_ABS = str((Path(__file__).resolve().parent / CACHE_DIR).resolve())
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
CHECKPOINT_REPO_ID = "hljiang/DA360"
CHECKPOINT_FILENAMES = {
    "./checkpoints/DA360_small.pth": "DA360_small.pth",
    "./checkpoints/DA360_base.pth": "DA360_base.pth",
    "./checkpoints/DA360_large.pth": "DA360_large.pth",
}

IS_HF_SPACE = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE") == "1")
_ALLOW_CHECKPOINT_DOWNLOAD = IS_HF_SPACE
INDOOR_EXAMPLE_COUNT = 3
MAX_POINT_CLOUD_POINTS = None
POINT_CLOUD_STRIDE = 1

_gradio_major = int(gr.__version__.split(".")[0])
FILE_ROUTE_PREFIX = "/gradio_api/file=" if _gradio_major >= 5 else "/file="

_model = None
_model_meta = None
_saver = None
_device = None

VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #d9d9d9; }
  #app { width: 100vw; height: 100vh; display: block; }
  #ui {
    position: absolute; top: 12px; left: 12px; z-index: 10;
    font: 13px/1.4 system-ui, sans-serif; color: #222;
    background: rgba(255,255,255,0.85); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.2); user-select: none;
  }
  #ui label { display: block; margin: 2px 0; }
  #ui input[type=range] { width: 160px; vertical-align: middle; }
  #size_val { display: inline-block; width: 34px; text-align: right; }
  #hint { color: #666; font-size: 11px; margin-top: 6px; }
  #err { position:absolute; top:12px; right:12px; color:#b00; background:#fff;
         padding:6px 10px; border-radius:6px; font:12px sans-serif; display:none; }
</style>
</head>
<body>
<div id="ui">
  <label>Point Size
    <input id="size" type="range" min="0.005" max="0.2" step="0.001" value="0.1">
    <span id="size_val"></span>
  </label>
  <label><input id="attn" type="checkbox" checked> Size attenuation (perspective)</label>
  <label><button id="reset" type="button">Reset view</button></label>
  <div id="hint">Left-drag rotate · right-drag pan · wheel zoom</div>
</div>
<div id="err"></div>
<canvas id="app"></canvas>
<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const params = new URLSearchParams(location.search);
const cloudUrl = params.get('cloud');
const initSize = parseFloat(params.get('size') || '0.1');

const canvas = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd9d9d9);
const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 10000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomSpeed = 0.25;
controls.rotateSpeed = 0.85;
controls.panSpeed = 0.85;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN,
};

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
}

const sizeInput = document.getElementById('size');
const sizeVal = document.getElementById('size_val');
const attn = document.getElementById('attn');
let material = null, home = null;

function showErr(msg) {
  const e = document.getElementById('err'); e.textContent = msg; e.style.display = 'block';
}

const loader = new PLYLoader();
loader.load(cloudUrl, (geometry) => {
  geometry.computeBoundingSphere();
  const bs = geometry.boundingSphere;
  material = new THREE.PointsMaterial({
    size: initSize, sizeAttenuation: true, vertexColors: true,
  });
  scene.add(new THREE.Points(geometry, material));

  const r = bs.radius || 1;
  const dist = r / Math.sin((camera.fov * Math.PI / 180) / 2) * 1.15;
  const c = bs.center;
  controls.target.copy(c);
  camera.position.set(c.x, c.y, c.z + dist);
  controls.minDistance = Math.max(r * 0.15, dist * 0.12);
  controls.maxDistance = Math.max(dist * 6, r * 15);
  camera.near = r / 100; camera.far = r * 100; camera.updateProjectionMatrix();
  home = { pos: camera.position.clone(), tgt: controls.target.clone() };

  sizeInput.value = initSize; sizeVal.textContent = initSize.toFixed(3);
}, undefined, (e) => showErr('Failed to load point cloud: ' + e));

sizeInput.addEventListener('input', () => {
  const v = parseFloat(sizeInput.value);
  sizeVal.textContent = v.toFixed(3);
  if (material) material.size = v;
});
attn.addEventListener('change', () => {
  if (material) { material.sizeAttenuation = attn.checked; material.needsUpdate = true; }
});
document.getElementById('reset').addEventListener('click', () => {
  if (home) { camera.position.copy(home.pos); controls.target.copy(home.tgt); }
});

function animate() {
  requestAnimationFrame(animate);
  resize(); controls.update(); renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""

DEFAULT_POINT_SIZE = 0.1
VIEWER_HEIGHT = 720
POINT_CLOUD_PLACEHOLDER = (
    f'<div class="point-cloud-placeholder" '
    f'style="width:100%;height:{VIEWER_HEIGHT}px;border:1px dashed #bbb;'
    f'border-radius:8px;background:#f5f5f5;display:flex;align-items:center;'
    f'justify-content:center;color:#666;font:14px/1.5 system-ui,sans-serif">'
    "Point cloud viewer — click <b>Run</b> to generate"
    "</div>"
)
DEMO_CSS = """
.point-cloud-square iframe {
    display: block;
    width: 100%;
}
.examples-panel .gallery {
    min-height: 360px;
}
.panorama-panel, .depth-panel {
    width: 100%;
}
.panorama-panel .image-container,
.panorama-panel .wrap,
.depth-panel .image-container,
.depth-panel .wrap {
    aspect-ratio: 2 / 1 !important;
    height: auto !important;
    max-height: 360px;
}
.panorama-panel img,
.depth-panel img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}
"""


def file_serve_url(abs_path):
    return f"{FILE_ROUTE_PREFIX}{os.path.abspath(abs_path)}"


def ensure_viewer_html():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "viewer.html")
    with open(path, "w") as f:
        f.write(VIEWER_HTML)
    return os.path.abspath(path)


def sort_key(s):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", s)]


def scene_type_from_path(path):
    name = os.path.basename(path).lower()
    if name.startswith("indoor"):
        return "Indoor"
    if name.startswith("outdoor"):
        return "Outdoor"
    return "Outdoor"


def ensure_checkpoints(paths=None, allow_download=None):
    """Use local ./checkpoints when present; otherwise download from Hugging Face."""
    global _ALLOW_CHECKPOINT_DOWNLOAD
    if paths is None:
        paths = list(CHECKPOINT_FILENAMES.keys())
    if allow_download is None:
        allow_download = _ALLOW_CHECKPOINT_DOWNLOAD
    missing = [p for p in paths if p in CHECKPOINT_FILENAMES and not os.path.exists(p)]
    if not missing:
        return
    if not allow_download:
        names = ", ".join(os.path.basename(p) for p in missing)
        raise gr.Error(
            f"Checkpoint not found locally: {names}. "
            "Place weights under ./checkpoints/ or run "
            "`bash scripts/download_models.sh` / `python app.py --download`."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            f"Missing checkpoints. Install huggingface_hub and download from "
            f"https://huggingface.co/{CHECKPOINT_REPO_ID}"
        ) from exc
    os.makedirs("checkpoints", exist_ok=True)
    for path in missing:
        filename = CHECKPOINT_FILENAMES[path]
        print(f"[checkpoints] fetching {CHECKPOINT_REPO_ID}/{filename} ...", flush=True)
        t0 = time.time()
        hf_hub_download(
            repo_id=CHECKPOINT_REPO_ID,
            filename=filename,
            local_dir="checkpoints",
        )
        print(f"[checkpoints] ready {path} ({time.time() - t0:.1f}s)", flush=True)


def get_example_images():
    paths = glob(os.path.join(EXAMPLE_DIR, "*.jpg")) + glob(os.path.join(EXAMPLE_DIR, "*.png"))
    paths = [p for p in paths if "depth" not in os.path.basename(p).lower()]
    examples = []
    for path in sorted(paths, key=sort_key):
        if os.path.getsize(path) < 1024:
            with open(path, "rb") as f:
                if f.read(16).startswith(b"version https://"):
                    continue
        examples.append(os.path.abspath(path))
    return examples


def build_example_rows(default_model):
    rows = []
    examples = get_example_images()
    for i, path in enumerate(examples):
        scene = scene_type_from_path(path)
        if scene == "Outdoor" and i >= len(examples) - INDOOR_EXAMPLE_COUNT:
            scene = "Indoor"
        rows.append([path, default_model, scene])
    return rows


def load_model(model_path):
    global _model, _model_meta, _saver, _device

    if _model is not None and _model_meta.get("path") == model_path:
        return _model, _model_meta, _saver, _device

    ensure_checkpoints([model_path])
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_checkpoint(model_path, map_location="cpu")
    meta = checkpoint_meta(model_dict)

    Net = getattr(networks, meta["net"])
    model = Net(
        meta["height"],
        meta["width"],
        dinov2_encoder=meta["dinov2_encoder"],
    )
    model.to(_device)
    model.load_state_dict(
        state_dict_from_checkpoint(model_dict, model.state_dict()),
        strict=False,
    )
    model.eval()

    _model = model
    _model_meta = {
        "path": model_path,
        "height": meta["height"],
        "width": meta["width"],
        "model_name": os.path.basename(model_path)[:-4],
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    _saver = Saver(os.path.join(CACHE_DIR, "demo"))
    return _model, _model_meta, _saver, _device


def _as_rgb_numpy(image):
    if image is None:
        raise gr.Error("Please upload a panoramic image.")
    if isinstance(image, dict):
        image = image.get("path") or image.get("url")
    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise gr.Error(f"Failed to read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def preprocess_rgb(rgb, height, width):
    rgb = cv2.resize(rgb, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    rgb_tensor = to_tensor(rgb.copy()).unsqueeze(0)
    normalized_rgb = normalize(to_tensor(rgb.copy())).unsqueeze(0)
    return rgb_tensor, normalized_rgb, rgb


def depth_to_colormap(pred_depth):
    depth = pred_depth.copy()
    depth = depth / depth[depth > 0].max()
    disp = depth.copy()
    disp[depth > 0] = 1 / depth[depth > 0]
    disp[depth <= 0] = 0
    return kitti_colormap(disp)


def filter_indoor_ceiling(points, colors):
    y = points[:, 1]
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    keep = y_norm >= 0.1
    return points[keep], colors[keep]


def clip_radial_outliers(points, colors, keep_percentile=98.0):
    if len(points) == 0:
        return points, colors
    centroid = np.median(points, axis=0)
    radius = np.linalg.norm(points - centroid, axis=1)
    threshold = np.percentile(radius, keep_percentile)
    keep = radius <= threshold
    if keep.sum() == 0:
        return points, colors
    return points[keep], colors[keep]


def normalize_point_cloud_scale(points, target_extent=10.0):
    if len(points) == 0:
        return points
    points = points.astype(np.float32, copy=True)
    points -= np.median(points, axis=0)
    extent = np.ptp(points, axis=0).max()
    if extent > 1e-8:
        points *= target_extent / extent
    return points


def build_point_cloud_mask(pred_depth_raw, pred_depth_norm, scene_type):
    valid = pred_depth_raw > 0
    if scene_type == "Outdoor":
        return valid & (pred_depth_norm < 200)
    return valid


def flip_point_cloud_axes(points):
    out = points.copy()
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


def subsample_point_cloud(points, colors, max_points=MAX_POINT_CLOUD_POINTS):
    if max_points is None or len(points) <= max_points:
        return points, colors
    idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
    return points[idx], colors[idx]


def postprocess_point_cloud(pred_depth_raw, pred_depth_np, rgb_np, scene_type):
    stride = max(int(POINT_CLOUD_STRIDE), 1)
    if stride > 1:
        pred_depth_raw = pred_depth_raw[::stride, ::stride]
        pred_depth_np = pred_depth_np[::stride, ::stride]
        rgb_np = rgb_np[::stride, ::stride]
    mask = build_point_cloud_mask(pred_depth_raw, pred_depth_np, scene_type)
    points, colors = depth_rgb_to_point_cloud(
        pred_depth_raw, rgb_np / 255.0, mask
    )
    if scene_type == "Indoor":
        points, colors = filter_indoor_ceiling(points, colors)
    points, colors = clip_radial_outliers(points, colors, keep_percentile=98.0)
    points, colors = subsample_point_cloud(points, colors)
    points = flip_point_cloud_axes(points)
    points = normalize_point_cloud_scale(points, target_extent=10.0)
    return points, colors


def export_ply_point_cloud(points, colors, path):
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")

    if colors.max() <= 1.0:
        colors = (colors * 255.0).clip(0, 255).astype(np.uint8)
    else:
        colors = colors.clip(0, 255).astype(np.uint8)

    if colors.shape[1] == 4:
        colors = colors[:, :3]

    cloud = trimesh.points.PointCloud(vertices=points.astype(np.float32), colors=colors)
    cloud.export(path, file_type="ply")
    return path


def render_point_cloud(points, colors, point_size=DEFAULT_POINT_SIZE):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ply_path = os.path.join(CACHE_DIR, f"pc_{timestamp}.ply")
    export_ply_point_cloud(points, colors, os.path.abspath(ply_path))

    viewer_abs = ensure_viewer_html()
    cloud_url = quote(file_serve_url(ply_path), safe="/=")
    src = f"{file_serve_url(viewer_abs)}?cloud={cloud_url}&size={point_size}&_={timestamp}"
    return (
        f'<iframe src="{src}" '
        f'style="width:100%;height:{VIEWER_HEIGHT}px;border:0;border-radius:8px" '
        f'allow="fullscreen"></iframe>'
    )


def run_inference(image_path, model_path, scene_type="Outdoor"):
    model, meta, _saver, device = load_model(model_path)
    rgb = _as_rgb_numpy(image_path)
    _, normalized_rgb, rgb_np = preprocess_rgb(rgb, meta["height"], meta["width"])
    equi_inputs = normalized_rgb.to(device)

    with torch.no_grad():
        outputs = model(equi_inputs)

    pred_disp = outputs["pred_disp"].detach().cpu()
    pred_depth_raw = (1 / pred_disp).numpy()[0, 0].astype(np.float32)
    pred_depth_np = pred_depth_raw / pred_depth_raw[pred_depth_raw > 0].min()
    depth_vis = np.ascontiguousarray(depth_to_colormap(pred_depth_np))
    points, colors = postprocess_point_cloud(
        pred_depth_raw, pred_depth_np, rgb_np, scene_type
    )
    return depth_vis, points, colors


def predict(image_path, model_path, scene_type="Outdoor"):
    depth_vis, points, colors = run_inference(image_path, model_path, scene_type)
    viewer_html = render_point_cloud(points, colors)
    return depth_vis, viewer_html


def build_demo(model_path):
    model_choices = list(CHECKPOINT_FILENAMES.keys())
    existing_models = [p for p in model_choices if os.path.exists(p)]
    default_model = model_path if os.path.exists(model_path) else (
        existing_models[0] if existing_models else DEFAULT_MODEL
    )
    example_rows = build_example_rows(default_model)

    with gr.Blocks(title="DA360 Demo") as demo:
        gr.HTML(f"<style>{DEMO_CSS}</style>")
        gr.Markdown(
            textwrap.dedent(
                """
                # Depth Anything in 360°
                Scale-invariant panoramic depth (KITTI colormap) and interactive 3D point cloud.

                [Project Page](https://insta360-research-team.github.io/DA360/) |
                [Paper](https://arxiv.org/abs/2512.22819) |
                [GitHub](https://github.com/insta360-research-team/DA360) |
                [Weights](https://huggingface.co/hljiang/DA360)

                **Run** estimates depth and builds the point cloud in one step.
                Checkpoints are read from `./checkpoints/` when present.
                """
            ).strip()
        )

        panel_height = 360

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Panoramic Input",
                    type="filepath",
                    height=panel_height,
                    elem_classes=["panorama-panel"],
                )
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Checkpoint",
                    info="Switch among small / base / large weights under ./checkpoints/.",
                )
                scene_type = gr.Radio(
                    choices=["Outdoor", "Indoor"],
                    value="Outdoor",
                    label="Point cloud post-processing",
                    info=(
                        "Outdoor: mask sky/far pixels (norm. depth ≥ 200× nearest). "
                        "Indoor: remove ceiling band. Does **not** change depth estimation."
                    ),
                )
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                depth_output = gr.Image(
                    label="Depth (KITTI colormap)",
                    type="numpy",
                    height=panel_height,
                    elem_classes=["depth-panel"],
                )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["examples-panel"]) as examples_col:
                pass
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Point cloud\n"
                    "Full resolution from the depth map (no stride / random subsample)."
                )
                point_cloud_output = gr.HTML(
                    value=POINT_CLOUD_PLACEHOLDER,
                    elem_classes=["point-cloud-square"],
                )

        def run(image_path, selected_model, selected_scene):
            if image_path is None:
                return None, POINT_CLOUD_PLACEHOLDER
            try:
                return predict(image_path, selected_model, selected_scene)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        run_btn.click(
            fn=run,
            inputs=[image_input, model_dropdown, scene_type],
            outputs=[depth_output, point_cloud_output],
        )

        with examples_col:
            if example_rows:
                gr.Examples(
                    examples=example_rows,
                    inputs=[image_input, model_dropdown, scene_type],
                    outputs=[depth_output, point_cloud_output],
                    fn=run,
                    cache_examples=False,
                    label="Examples",
                )

    return demo


def resolve_server_port(preferred_port):
    if preferred_port == 0:
        return 0
    for port in range(preferred_port, preferred_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                if port != preferred_port:
                    print(f"Port {preferred_port} is in use, using {port} instead.")
                return port
            except OSError:
                continue
    print(f"No free port found near {preferred_port}, letting Gradio choose one.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing checkpoints from Hugging Face (hljiang/DA360)",
    )
    args = parser.parse_args()

    if args.download or IS_HF_SPACE:
        globals()["_ALLOW_CHECKPOINT_DOWNLOAD"] = True
        ensure_checkpoints(list(CHECKPOINT_FILENAMES.keys()), allow_download=True)

    ensure_viewer_html()
    demo = build_demo(args.model_path)
    server_port = resolve_server_port(args.server_port)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=server_port,
        share=args.share,
        allowed_paths=[CACHE_DIR_ABS],
        show_error=True,
    )
