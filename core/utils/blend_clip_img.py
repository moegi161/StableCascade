import torch
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
import torch.nn as nn

@torch.inference_mode()
def clip_image_embed_with_mask_blend(
    model: CLIPVisionModelWithProjection,
    target_pixels: torch.Tensor,    # [B,3,H,W] *after CLIP preprocess*
    ref_pixels: torch.Tensor,       # [B,3,H,W] *after CLIP preprocess*
    ref_mask: torch.Tensor | None,  # [B,1,H,W] in [0,1], spatially aligned to target_pixels/ref_pixels
    alpha: float = 1.0,
    blend_cls: bool = False,
):
    dev  = next(model.parameters()).device
    dtyp = next(model.parameters()).dtype

    target_pixels = target_pixels.to(dev, dtyp)
    ref_pixels    = ref_pixels.to(dev, dtyp)
    if ref_mask is not None:
        ref_mask = ref_mask.to(dev, target_pixels.dtype)

    vm = model.vision_model  # CLIPVisionTransformer

    # ---- robust forward (works across HF versions) ----
    def _last_hidden_state(x: torch.Tensor):
        out = vm(x, output_hidden_states=False)  # don't pass return_dict
        if isinstance(out, (tuple, list)):
            return out[0]  # last_hidden_state
        return out.last_hidden_state

    h_t = _last_hidden_state(target_pixels)  # [B, 1+N, C]
    h_r = _last_hidden_state(ref_pixels)     # [B, 1+N, C]

    # ---- masked blending on patch tokens ----
    if ref_mask is not None:
        cfg   = vm.config
        Hp    = cfg.image_size // cfg.patch_size
        Wp    = cfg.image_size // cfg.patch_size
        m     = F.interpolate(ref_mask, size=(Hp, Wp), mode="bilinear", align_corners=True).clamp(0, 1)
        m     = m.flatten(2).transpose(1, 2)  # [B, Np, 1]

        cls_t, pt_t = h_t[:, :1, :], h_t[:, 1:, :]
        cls_r, pt_r = h_r[:, :1, :], h_r[:, 1:, :]

        a = torch.as_tensor(alpha, device=pt_t.device, dtype=pt_t.dtype)
        while a.dim() < 3:
            a = a.unsqueeze(0)  # -> [1,1,1]

        pt_blend = (1 - a * m) * pt_t + (a * m) * pt_r
        if blend_cls:
            cls_m    = m.mean(dim=1, keepdim=True)  # [B,1,1]
            cls_blend= (1 - a * cls_m) * cls_t + (a * cls_m) * cls_r
        else:
            cls_blend= cls_t

        h_blend = torch.cat([cls_blend, pt_blend], dim=1)
    else:
        h_blend = h_t

    # ---- post-norm + projection (matches HF) ----
    h_norm       = vm.post_layernorm(h_blend)
    pooled       = h_norm[:, 0, :]               # CLS
    image_embeds = model.visual_projection(pooled)  # [B, D]
    return image_embeds



def _as_pos_tensor(pos_any: torch.nn.Module | torch.Tensor | None) -> torch.Tensor:
    if pos_any is None:
        raise RuntimeError("Positional embeddings not found.")
    if isinstance(pos_any, nn.Embedding):
        return pos_any.weight
    if isinstance(pos_any, (torch.Tensor, torch.nn.Parameter)):
        return pos_any
    # some modules keep it as an attribute named 'weight'
    if hasattr(pos_any, "weight") and isinstance(pos_any.weight, (torch.Tensor, torch.nn.Parameter)):
        return pos_any.weight
    raise TypeError(f"Unsupported positional embedding type: {type(pos_any)}")

@torch.inference_mode()
def clip_image_embed_preproj_blend(
    model: "CLIPVisionModelWithProjection",
    target_pixels: torch.Tensor,    # [B,3,H,W] after CLIP preprocess
    ref_pixels: torch.Tensor,       # [B,3,H,W] after CLIP preprocess
    ref_mask: torch.Tensor | None,  # [B,1,H,W] in [0,1], same resize/crop as pixels (no normalize)
    alpha: float = 1.0,
):
    dev  = next(model.parameters()).device
    dtyp = next(model.parameters()).dtype

    target_pixels = target_pixels.to(dev, dtyp)
    ref_pixels    = ref_pixels.to(dev, dtyp)
    if ref_mask is not None:
        ref_mask = ref_mask.to(dev, target_pixels.dtype).clamp(0, 1)

    vm  = model.vision_model
    emb = vm.embeddings  # CLIPVisionEmbeddings

    # 1) Patch conv (still 2D)
    conv   = emb.patch_embedding
    t_feat = conv(target_pixels)  # [B, C, Hp, Wp]
    r_feat = conv(ref_pixels)     # [B, C, Hp, Wp]

    # 2) Blend in patch-embedding space
    if ref_mask is not None:
        m = F.interpolate(ref_mask, size=t_feat.shape[-2:], mode="bilinear", align_corners=True)  # [B,1,Hp,Wp]
        a = torch.as_tensor(alpha, device=t_feat.device, dtype=t_feat.dtype)
        t_feat = (1 - a * m) * t_feat + (a * m) * r_feat

    # 3) Flatten to tokens
    x = t_feat.flatten(2).permute(0, 2, 1)  # [B, Np, C]
    B, Np, C = x.shape

    # ---- class token (handle [hidden], [1,hidden], or [1,1,hidden]) ----
    cls = emb.class_embedding.to(dev, dtyp)
    if cls.dim() == 1:
        cls = cls.unsqueeze(0).unsqueeze(0)   # [1,1,C]
    elif cls.dim() == 2:
        cls = cls.unsqueeze(1)                # [1,1,C]
    elif cls.dim() != 3:
        raise ValueError(f"Unexpected class_embedding dim {cls.dim()} with shape {tuple(cls.shape)}")

    cls = cls.expand(B, -1, -1)               # [B,1,C]
    x   = torch.cat([cls, x], dim=1)          # [B, 1+Np, C]

    # ---- positional embeddings (normalize to Tensor and align) ----
    # Prefer HF's CLIPVisionEmbeddings.position_embedding if present.
    pos_any = getattr(emb, "position_embedding", None)
    if pos_any is None:
        # fallbacks for other CLIP/ViT variants
        pos_any = getattr(vm, "positional_embedding", None) or getattr(vm, "pos_embed", None)

    pos = _as_pos_tensor(pos_any).to(dev, dtyp)  # usually [1, 1+Np, C] or [1+Np, C] or [1+Np, 1, C]
    # Make pos shape [1, L, C]
    if pos.dim() == 2:                   # [L, C]
        pos = pos.unsqueeze(0)           # [1, L, C]
    elif pos.dim() == 3:                 # [Bpos, L, C] or [1, L, C]
        if pos.size(0) != 1:
            # broadcast-first dimension to 1 for addition with [B, L, C]
            pos = pos[:1, ...]
    else:
        raise ValueError(f"Unexpected position_embedding dim {pos.dim()} with shape {tuple(pos.shape)}")

    Lx = x.size(1)
    Lp = pos.size(1)

    # Handle common +1/-1 CLS length mismatches
    if Lp == Lx + 1:
        # drop CLS pos if present
        pos = pos[:, 1:, :]
        Lp = pos.size(1)
    elif Lp + 1 == Lx:
        # x likely has an extra token (rare); drop the first token in x to match
        x = x[:, 1:, :]
        Lx = x.size(1)

    if Lp != Lx:
        raise RuntimeError(f"Token length mismatch: x has {Lx}, pos has {Lp}. "
                           f"Check input resolution/patch grid or interpolate pos.")

    # Channel/hidden dim must match
    if pos.size(-1) != C:
        raise RuntimeError(f"Hidden dim mismatch: pos {pos.size(-1)} vs tokens {C}.")

    x = x + pos  # [B, L, C]

    # 4) Pre-LN → Encoder → Post-LN
    pre_ln = getattr(vm, "pre_layernorm", getattr(vm, "pre_layrnorm", None))
    if pre_ln is not None:
        x = pre_ln(x)

    enc_out = vm.encoder(x)
    x = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out.last_hidden_state

    post_ln = getattr(vm, "post_layernorm", getattr(vm, "post_layrnorm", None))
    if post_ln is not None:
        x = post_ln(x)

    # 5) CLS pool + projection
    pooled = x[:, 0, :]                       # [B, C]
    image_embeds = model.visual_projection(pooled)  # [B, D]
    return image_embeds
