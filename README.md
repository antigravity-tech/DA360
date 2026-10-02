# Depth Anything in 360°: Towards Scale Invariance in the Wild

**Hualie Jiang**<sup>1,2</sup> · **Ziyang Song**<sup>1,2</sup> · **Zhiqiang Lou**<sup>1,2</sup> · **Rui Xu**<sup>1,2</sup> · **Minglang Tan**<sup>2</sup>

<sup>1</sup> Antigravity Tech &nbsp;&nbsp; <sup>2</sup> Insta360 Research

[**Paper**](https://arxiv.org/abs/2512.22819) · [**Project Page**](https://antigravity-tech.github.io/DA360/) · [**Code**](https://github.com/antigravity-tech/DA360) · [**Demo**](https://huggingface.co/spaces/hljiang/DA360) · [**Dataset**](https://huggingface.co/datasets/hljiang/DA360)

Panoramic depth estimation lags in open-world generalization compared to perspective models. **DA360** adapts [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to $360°$ imagery: a ViT-predicted shift turns scale-and-shift-invariant disparity into **scale-invariant** disparity for direct 3D reconstruction, and **circular padding** in the DPT decoder removes seam artifacts. On indoor benchmarks and our outdoor **Metropolis** dataset, DA360 cuts relative depth error substantially vs. its base model and prior panoramic methods (e.g. PanDA). This repo is the official PyTorch implementation.

<p align="center">
<img src='assets/teaser.jpg' width=980>
</p>

## News

- **2026-10-02:** **Training datasets** — PyTorch loaders plus Structured3D / Deep360 layout notes and TartanAir v2 download & preprocessing (`data/download_tartanair2.py`, `data/preprocess_tartanair2.py`; [Training datasets](#training-datasets)).
- **2026-09-30:** **Metropolis** outdoor test set released with Matterport3D and Stanford2D3D on [**hljiang/DA360**](https://huggingface.co/datasets/hljiang/DA360) ([Evaluation](#evaluation)).
- **2026-09-13:** **Demo** on [Hugging Face Space](https://huggingface.co/spaces/hljiang/DA360) ([Demo](#demo)); local Gradio via `app.py`.

## Preparation

### Installation

Create the environment

```bash
bash scripts/setup_da360_env.sh
conda activate da360
```

## Pre-trained Weights

Checkpoints are on Hugging Face: [**hljiang/DA360**](https://huggingface.co/hljiang/DA360) (`DA360_small.pth`, `DA360_base.pth`, `DA360_large.pth`).

Download into `./checkpoints/`:

```bash
bash scripts/download_models.sh
```

Or a single file:

```bash
hf download hljiang/DA360 DA360_large.pth --local-dir checkpoints
```


## Demo

**Online:** [hljiang/DA360](https://huggingface.co/spaces/hljiang/DA360) on Hugging Face Spaces.

**Local:**

```bash
python app.py
```

Open the printed URL (default http://localhost:7860). Upload a 2:1 panorama or use **Examples** from `./data/images/`, choose *Indoor* / *Outdoor* point-cloud post-processing if needed, then **Run**. Use the **Checkpoint** dropdown to switch among small / base / large weights under `./checkpoints/`.

### Test on panoramic images (CLI)

Put panoramas in `./data/images/` (six examples are included), then run:

```bash
python test.py --model_path ./checkpoints/DA360_large.pth --model_name DA360_large
```

Outputs go under **`./results/images/`** (not under `data/`):

- `{name}_depth_pred_{model_name}.jpg` — KITTI colormap  
- `{name}_depth_pred_{model_name}.exr` — raw depth  
- `{name}_pc_pred_{model_name}.ply` — point cloud  
- `{name}_rgb.jpg` — input RGB  

Override with `--output_dir`.


## Evaluation

Benchmark RGB–depth test sets (Matterport3D, Stanford2D3D, Metropolis) are on Hugging Face: [**hljiang/DA360**](https://huggingface.co/datasets/hljiang/DA360).

Download and extract into `./data/` (`Matterport3D/`, `Stanford2D3D/`, `Metropolis/`):

```bash
bash scripts/download_eval_data.sh
```

### Perform Evaluation

```bash
bash scripts/evaluate.sh
```

**Metrics** (always): `./results/<dataset>/result_<model_name>.txt`  
(e.g. `results/matterport3d/result_DA360_large.txt`).

**Visualization** (with `--save_samples`, 10 evenly spaced samples per dataset):  
`./results/<dataset>/` — e.g. `0001_rgb.jpg`, `0001_depth_gt.jpg`, `0001_depth_pred_<model_name>.jpg`, `0001_pc_gt.ply`, `0001_pc_pred_<model_name>.ply`.

Change the root with `--results_dir` (default `./results/`).


## Training datasets

PyTorch loaders live under `datasets/` (`structured3d.py`, `deep360.py`, `tartanair.py`). Use them after preparing the raw data below.

1. **Structured3D** — [dataset](https://structured3d-dataset.org/) (panorama zips) → `data/structured3d/scene_*/2D_rendering/.../panorama/`. Override the root with `DA360_STRUCTURED3D_ROOT` if needed.

2. **Deep360** (episode 6) — [ep6_500frames.zip](https://drive.google.com/file/d/19f0SREILDE3X1OIaGP_5NnWyHXyTPWsg/view?usp=drive_link):

   ```bash
   mkdir -p data/deep360
   pip install gdown   # once, if needed
   gdown 19f0SREILDE3X1OIaGP_5NnWyHXyTPWsg -O data/deep360/ep6_500frames.zip
   unzip -q data/deep360/ep6_500frames.zip -d data/deep360
   ```

3. **TartanAir v2** — use env `tartanair2` for download and panoramic preprocessing. **This can take hours to days** for all environments; pass `--env` to both scripts to try one scene first.

   ```bash
   bash scripts/setup_tartanair2_env.sh
   conda activate tartanair2
   cd data

   python download_tartanair2.py \
     --data-root tartanair2_raw \
     --data-source huggingface \
     --difficulty hard \
     --modality image depth

   python preprocess_tartanair2.py \
     --data-root tartanair2_raw \
     --output-root tartanair2 \
     --difficulty hard \
     --height 640 --width 1280 \
     --device cuda:0
   ```


## Acknowledgements

The project is partially based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [PanDA](https://github.com/caozidong/PanDA) and [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion).
 

## Citation

Please cite our paper if you find our work useful in your research.

```
@article{jiang2025depth,
  title={Depth Anything in $360\^{}$\backslash$circ $: Towards Scale Invariance in the Wild},
  author={Jiang, Hualie and Song, Ziyang and Lou, Zhiqiang and Xu, Rui and Tan, Minglang},
  journal={arXiv preprint arXiv:2512.22819},
  year={2025}
}
```
