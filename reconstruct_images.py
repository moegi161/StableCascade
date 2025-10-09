
# -*- coding: utf-8 -*-
# Auto-generated from `reconstruct_images.ipynb` with GUI display replaced by image saving.
# Generated: 2025-10-01T11:01:25.198179
# Notes:
# - Matplotlib backend set to 'Agg' (no GUI).
# - All `plt.show()` and custom `_save_show()` calls are replaced with `save_current_figure()`.
# - Figures will be written to `./saved_figures/fig_XXX.png` (relative to where you run the script).
# - Any function named `showimages` is preserved but will save figures instead of showing them.

import os, pathlib, contextlib

# Force headless matplotlib BEFORE importing pyplot
import matplotlib

from modules.effnet import EfficientNetEncoderBlend
matplotlib.use("Agg")  # no GUI

import matplotlib.pyplot as plt
import math
import PIL.Image as PILImage
from io import BytesIO
import torch
import torchvision

import yaml
import torch
from tqdm import tqdm


from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreB

_SAVEDIR = pathlib.Path("./saved_figures")
_SAVEDIR.mkdir(parents=True, exist_ok=True)
_show_images_counter = 0  # auto-increment if no filename is given

def save_images(images, rows=None, cols=None, return_images=False, **kwargs):
    """
    Save a grid of images to disk (headless).
    - images: torch.Tensor [N, C, H, W], values typically in [0,1]
    - rows/cols: optional grid size; if one is None, it's inferred.
    - return_images: if True, returns the PIL image object of the grid.
    Extra kwargs:
      - out_dir (str | Path): directory to save into (default: ./saved_figures)
      - filename (str): exact filename to use (default: auto-increment)
      - prefix (str): filename prefix if auto (default: "show_images_")
    """
    global _show_images_counter

    # Allow caller to customize output location/name
    out_dir = pathlib.Path(kwargs.get("out_dir", _SAVEDIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = kwargs.get("prefix", "show_images_")
    filename = kwargs.get("filename", None)

    if not isinstance(images, torch.Tensor):
        raise TypeError("`images` must be a torch.Tensor of shape [N, C, H, W].")

    # Bring to 3 channels for visualization
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]

    n = images.size(0)
    if rows is None and cols is None:
        # square-ish layout
        rows = int(math.sqrt(n)) or 1
        cols = math.ceil(n / rows)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)

    # Tensor -> PIL grid
    _, _, h, w = images.shape
    grid = PILImage.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        # Ensure on CPU and in [0,1]
        img_cpu = img.detach().cpu().clamp(0, 1)
        pil = torchvision.transforms.functional.to_pil_image(img_cpu)
        grid.paste(pil, box=((i % cols) * w, (i // cols) * h))

    # Choose output filename
    if filename is None:
        _show_images_counter += 1
        filename = f"{prefix}{_show_images_counter:03d}.png"
    if not filename.lower().endswith(".png"):
        filename += ".png"

    out_path = out_dir / filename
    grid.save(out_path, format="PNG")

    # If a caller depended on the old BytesIO side-effect, keep a minimal equivalent
    # (not displayed, just created for compatibility—safe to remove if unneeded)
    _ = BytesIO()
    grid.save(_, format="PNG")

    # Optional: print path for logs
    print(f"Saved grid -> {out_path}")

    if return_images:
        return grid

def save_current_figure(basename: str | None = None):
    """ Save the current matplotlib figure and close it.
    If `basename` is None, use an incrementing pattern fig_001.png, fig_002.png, ...
    """
    global _fig_counter
    _fig_counter += 1
    if basename is None:
        fname = f"fig_{_fig_counter:03d}.png"
    else:
        # sanitize basename
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_",".") else "_" for ch in basename)
        if not safe.endswith(".png"):
            safe += ".png"
        fname = safe
    out = _SAVEDIR / fname
    # Use bbox_inches='tight' to reduce extra whitespace
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return str(out)

# Backward compatibility in case the notebook used a helper
def _save_show():
    return save_current_figure()

# A context manager to capture accidental plt.show() via interactive libs (rare)
@contextlib.contextmanager
def _noninteractive():
    yield


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# SETUP STAGE B & A
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
    
core = WurstCoreB(config_dict=config_file_b, device=device, training=False)


extras = core.setup_extras_pre()
data = core.setup_data(extras)
models = core.setup_models(extras)
models.generator.bfloat16()
print("STAGE B READY")

batch = next(data.iterator)
print("ORIG SIZE:", batch['images'].shape)

print(batch['captions'])

extras.sampling_configs['cfg'] = 1.1
extras.sampling_configs['shift'] = 1
extras.sampling_configs['timesteps'] = 10
extras.sampling_configs['t_start'] = 1.0

print("Original Size:", batch['images'].shape)
factor = 3/4
scaled_image = downscale_images(batch['images'], factor)
print("[Optional] Downscaled Size:", scaled_image.shape)

effnet_latents = models.effnet(extras.effnet_preprocess(scaled_image.to(device)))
print("Encoded Size:", effnet_latents.shape)

conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False)
unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True)    
conditions['effnet'] = effnet_latents
unconditions['effnet'] = torch.zeros_like(effnet_latents)

"""
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    sampling_b = extras.gdf.sample(
        models.generator, conditions, (batch['images'].size(0), 4, batch['images'].size(-2)//4, batch['images'].size(-1)//4),
        unconditions, device=device, **extras.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models.stage_a.decode(sampled_b).float()
    print("Decoded Size:", sampled.shape)

save_images(batch['images'])
save_images(sampled)
"""

# Stage B Parameters
# Ianna: New code for blending with EfficientNet
extras.sampling_configs['cfg'] = 1.1
extras.sampling_configs['shift'] = 1
extras.sampling_configs['timesteps'] = 10
extras.sampling_configs['t_start'] = 1.0


with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)

    print("Original Size:", batch['images'].shape)
    factor = 3/4
    scaled_image = downscale_images(batch['images'], factor)
    print("[Optional] Downscaled Size:", scaled_image.shape)
    reference_image = downscale_images(batch['ref_images'], factor)
    print("[Optional] Downscaled Reference Size:", reference_image.shape)
    mask = downscale_images(batch['ref_masks'], factor)
    print("[Optional] Downscaled Mask Size:", mask.shape)
    
    # Prepare inputs (you already have preprocess)
    t_img = extras.effnet_preprocess(scaled_image.to(device))     # [B,3,H,W]
    r_img = extras.effnet_preprocess(reference_image.to(device))  # [B,3,H,W]
    m = mask.to(device)  # [B,1,H,W] in [0,1]
    
    # Optional: affine theta (normalize to [-1,1] coords as PyTorch expects)
    # theta = affine_matrix.to(device) if affine_matrix is not None else None  # [B,2,3]
    
    effnet_latents = models.effnet_blend(
        target=t_img,
        reference=r_img,
        mask=m,
        affine_theta=None,       # or None
        video_length=None,        # or an int for videos
        #per_block_alpha=None,     # or {idx: 0.5, ...} for per-layer control
    )
    
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True)    
    conditions['effnet'] = effnet_latents
    unconditions['effnet'] = torch.zeros_like(effnet_latents)

    sampling_b = extras.gdf.sample(
        models.generator, conditions, (batch['images'].size(0), 4, batch['images'].size(-2)//4, batch['images'].size(-1)//4),
        unconditions, device=device, **extras.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models.stage_a.decode(sampled_b).float()
    print("Decoded Size:", sampled.shape)

save_images(batch['images'])
save_images(sampled, prefix="blended_0_")
