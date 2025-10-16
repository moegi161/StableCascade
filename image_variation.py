import os
import yaml
import torch
from tqdm import tqdm
import pathlib
import math
from PIL import Image as PILImage
import torchvision

from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreC, WurstCoreB

# Ianna:
from core.utils import blend_clip_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

def slerp(a, b, t, eps=1e-8):
    # a,b: [B,1,D]
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
    theta = torch.arccos(cos)
    sin = torch.sin(theta)
    w1 = torch.sin((1 - t) * theta) / (sin + eps)
    w2 = torch.sin(t * theta)       / (sin + eps)
    return w1 * a + w2 * b



# SETUP STAGE C
config_file = 'configs/inference/stage_c_3b.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

core = WurstCoreC(config_dict=loaded_config, device=device, training=False)

# SETUP STAGE B
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
    
core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

# SETUP MODELS & DATA
extras = core.setup_extras_pre()
models = core.setup_models(extras)
models.generator.eval().requires_grad_(False)
print("STAGE C READY")

extras_b = core_b.setup_extras_pre()
models_b = core_b.setup_models(extras_b, skip_clip=True)
models_b = WurstCoreB.Models(
   **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
)
models_b.generator.bfloat16().eval().requires_grad_(False)
print("STAGE B READY")

"""
### Image Variation Example ###

batch_size = 4
file = "/mnt/workspace2025/chan/StableCascade/figures/image.png"
images = resize_image(PIL.Image.open(file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

batch = {'images': images}

caption = ""
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras.sampling_configs['cfg'] = 4
extras.sampling_configs['shift'] = 2
extras.sampling_configs['timesteps'] = 20
extras.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    sampling_c = extras.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
        sampled_c = sampled_c
        
    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()
    
save_images(batch['images'], filename="input.png")
save_images(sampled, filename="variation.png")
"""

### Feature blending Example  ###

batch_size = 4
target_file = "/mnt/workspace2024/chan/celebv-hq/celebv-hq-100/frames_align/_hBFmCNNviE_8_0/frame_0001.png"
target_images = resize_image(PIL.Image.open(target_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
ref_file = "/mnt/workspace2024/chan/celebv-hq/reference_100/ref_img/eyeglasses_133.png"
ref = resize_image(PIL.Image.open(ref_file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
mask = "/mnt/workspace2024/chan/celebv-hq/reference_100/ref_mask/eyeglasses_000.png"
ref_masks = resize_image(PIL.Image.open(mask).convert("L")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
images = extras.clip_preprocess(target_images)
ref_images = extras.clip_preprocess(ref)
ref_masks = torchvision.transforms.functional.resize(ref_masks, (images.size(-2), images.size(-1)), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

batch = {'images': images, "ref_images": ref_images, "ref_masks": ref_masks}
target_images_eff = extras_b.effnet_preprocess(target_images)
ref_images_eff = extras_b.effnet_preprocess(ref)
ref_masks_eff = torchvision.transforms.functional.resize(ref_masks, (target_images_eff.size(-2), target_images_eff.size(-1)), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

caption = ""
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras.sampling_configs['cfg'] = 4
extras.sampling_configs['shift'] = 2
extras.sampling_configs['timesteps'] = 20
extras.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size


with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=True)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    img_embs= blend_clip_img.clip_image_embed_preproj_blend(models.image_model, batch['images'], batch['ref_images'], batch['ref_masks'], alpha=1.0)
    img_embs = img_embs.unsqueeze(1) # [B,1,C]
    
    conditions['clip_img'] = img_embs    #
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    # Ianna: New code for blending with EfficientNet
    print(target_images_eff.shape, ref_images_eff.shape, ref_masks_eff.shape)
    effnet_latents = models_b.effnet_blend(
        target=target_images_eff,
        reference=ref_images_eff,
        mask=ref_masks_eff,
        affine_theta=None,       # or None
        video_length=None,        # or an int for videos
        #per_block_alpha=None,     # or {idx: 0.5, ...} for per-layer control
    )
    conditions_b['effnet'] = effnet_latents
    
    sampling_c = extras.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
        sampled_c = sampled_c
        
    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c  # halfway blend
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()
    
save_images(target_images, filename="input.png")
save_images(sampled, filename="feature_blend_clip_img.png")


"""
### Image-to-Image Example ###
batch_size = 4
file = "/mnt/workspace2025/chan/StableCascade/figures/image.png"
images = resize_image(PIL.Image.open(file).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

batch = {'images': images}

show_images(batch['images'])

caption = "a person riding a rodent"
noise_level = 0.8
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

effnet_latents = core.encode_latents(batch, models, extras)
t = torch.ones(effnet_latents.size(0), device=device) * noise_level
noised = extras.gdf.diffuse(effnet_latents, t=t)[0]

# Stage C Parameters
extras.sampling_configs['cfg'] = 4
extras.sampling_configs['shift'] = 2
extras.sampling_configs['timesteps'] = int(20 * noise_level)
extras.sampling_configs['t_start'] = noise_level
extras.sampling_configs['x_init'] = noised

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    sampling_c = extras.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
        sampled_c = sampled_c
        
    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()

save_images(batch['images'], filename="input2.png")
save_images(sampled, filename="image_to_image.png")
"""
