import torchvision
from torch import nn
import torch
import torch.nn.functional as F
import torchvision


# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))
    

# Ianna: A new class for blending features from target and reference images using EfficientNet
def _maybe_repeat_temporal(x, video_length: int | None):
    if video_length is None:
        return x
    # x: [B, C, H, W] -> [(B*F), C, H, W]
    B, C, H, W = x.shape
    x = x.unsqueeze(1).expand(B, video_length, C, H, W).reshape(B * video_length, C, H, W)
    return x

def _resize_mask(mask, to_hw, align_corners=True):
    # mask: [B,1,H,W] (or [B,C,H,W]) → resized to to_hw
    return F.interpolate(mask, size=to_hw, mode="bilinear", align_corners=align_corners)

def _affine_warp(feat, theta, mode="bilinear", align_corners=True, padding_mode="zeros"):
    """
    feat: [B,C,H,W]
    theta: [B,2,3] in normalized coordinates (like affine_grid expects)
    """
    B, C, H, W = feat.shape
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=align_corners)
    return F.grid_sample(feat, grid, mode=mode, align_corners=align_corners, padding_mode=padding_mode)

# Ianna: A tiny bottleneck layer to add after EfficientNet features
class TinyBottleneck(nn.Module):
    def __init__(self, c_in=1280, c_mid=512, gain=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(c_in, elementwise_affine=False, eps=1e-6)
        self.down = nn.Linear(c_in, c_mid, bias=False)
        self.up   = nn.Linear(c_mid, c_in, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)  # start as identity via zero residual
        self.gain = nn.Parameter(torch.tensor(gain), requires_grad=False)

    @torch.inference_mode()
    def forward(self, feat):           # feat: [B,C,H,W] from late EffNet block
        B,C,H,W = feat.shape
        x = feat.permute(0,2,3,1).reshape(-1, C)       # [BHW,C]
        x = self.ln(x)
        z = self.down(x)
        x = self.up(z)
        x = x.view(B,H,W,C).permute(0,3,1,2)
        return feat + self.gain * x

# Ianna: New class for blending with EfficientNet
class EfficientNetEncoderBlend(nn.Module):
    """
    EfficientNet-v2-s encoder with masked alpha-blending of target/reference features at chosen blocks.

    Args:
      c_latent: output channels after 1x1 mapping (same as your baseline).
      blend_points: list[int] | "all" — which feature blocks to blend (indices in .features).
      alpha: float in [0,1]; we do y = (1 - alpha*mask)*y_t + (alpha*mask)*y_r.
      warp_mask: if True, apply affine to mask before blending.
      warp_reference: if True, apply affine to reference features before blending.
      keep_eval: keep backbone in eval() (recommended since you don't retrain).

    Forward:
      target: [B,3,H,W]
      reference: [B,3,H,W]
      mask: [B,1,H,W] (values in [0,1]); if None, no blending.
      affine_theta: optional [B,2,3] (normalized coords) used if warp_* is True
      video_length: optional int to tile per-frame (for your temporal case)
      per_block_alpha: optional dict {idx: float or tensor} to override alpha at specific blocks
    """
    def __init__(
        self,
        c_latent: int = 16,
        blend_points="all",
        alpha: float = 1.0,
        warp_mask: bool = False,
        warp_reference: bool = False,
        keep_eval: bool = True,
    ):
        super().__init__()
        base = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        self.backbone = base.features  # nn.Sequential
        if keep_eval:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.out_channels = 1280  # v2_s last feature planes
        self.mapper = nn.Sequential(
            nn.Conv2d(self.out_channels, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),
        )

        self.alpha_default = alpha
        self.warp_mask = warp_mask
        self.warp_reference = warp_reference
        self.c_latent = c_latent
        
        #Ianna: Test a no-training bottleneck once at init
        self.bottleneck = TinyBottleneck(c_in=1280, c_mid=512, gain=0.3).eval()
        for p in self.bottleneck.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            self.bottleneck.up.weight.copy_(self.bottleneck.down.weight.t())  # <— the key line


        # Decide where to blend
        if blend_points == "all":
            self.blend_indices = set(range(len(self.backbone)))
        else:
            self.blend_indices = set(blend_points or [])

    @torch.inference_mode()
    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor | None = None,           # [B,1,H,W]
        affine_theta: torch.Tensor | None = None,   # [B,2,3] normalized (optional)
        video_length: int | None = None,
        per_block_alpha: dict[int, float | torch.Tensor] | None = None,
    ):
        # Pre: pass your own effnet_preprocess before calling this encoder
        # Shapes:
        #   target/ref: [B,3,H,W], mask: [B,1,H,W]
        h_t = target
        h_r = reference

        # Temporal tiling if needed
        if video_length is not None:
            h_t = _maybe_repeat_temporal(h_t, video_length)
            h_r = _maybe_repeat_temporal(h_r, video_length)
            if mask is not None:
                mask = _maybe_repeat_temporal(mask, video_length)
            if affine_theta is not None:
                # Repeat B->B*F
                Bf = h_t.shape[0]
                repeat_factor = Bf // affine_theta.shape[0]
                affine_theta = affine_theta.repeat_interleave(repeat_factor, dim=0)

        # Run through feature blocks; blend at chosen indices
        for idx, block in enumerate(self.backbone):
            h_t = block(h_t)
            h_r = block(h_r)

            if (mask is not None) and (idx in self.blend_indices):
                # Resize mask to current spatial res
                m = _resize_mask(mask, to_hw=h_t.shape[-2:])  # [B,1,h,w]
                # Optional: warp mask &/or reference features by affine
                if self.warp_mask and (affine_theta is not None):
                    m = _affine_warp(m, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")
                hr = h_r
                if self.warp_reference and (affine_theta is not None):
                    hr = _affine_warp(h_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

                # Alpha (block-specific or default). Allow scalar or broadcastable tensor
                a = self.alpha_default
                if per_block_alpha is not None and idx in per_block_alpha:
                    a = per_block_alpha[idx]
                if not torch.is_tensor(a):
                    a = torch.as_tensor(a, dtype=h_t.dtype, device=h_t.device)
                # Make a broadcastable [B,1,1,1] tensor
                while a.dim() < 4:
                    a = a.unsqueeze(0)
                a = a.to(h_t.dtype).to(h_t.device)

                # y = (1 - a*m)*target_feat + (a*m)*ref_feat
                # mask may be [B,1,h,w], auto-broadcast over channels
                h_t = (1 - a * m) * h_t + (a * m) * hr

        # in forward, after your last blend point:
        h_t = self.bottleneck(h_t)                           # low-rank per-location
        z = self.mapper(h_t)  # [B,c_latent,h_last,w_last]
        return z

class EfficientNetEncoderBlendPostProj(EfficientNetEncoderBlend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mapper = nn.Sequential(
            nn.Conv2d(self.out_channels, self.c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_latent, affine=False),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, target, reference, mask=None, affine_theta=None, video_length=None, alpha=None):
        # 1) run EfficientNet features independently (NO in-backbone blending)
        h_t = target
        h_r = reference
        for block in self.backbone:
            h_t = block(h_t)
            h_r = block(h_r)

        # 2) map both to low-d Csc
        z_t = self.mapper(h_t)    # [B, c_latent, 24, 24]
        z_r = self.mapper(h_r)    # [B, c_latent, 24, 24]

        # 3) mask handling
        if mask is None:
            return z_t

        m = torch.nn.functional.interpolate(mask, size=z_t.shape[-2:], mode="bilinear", align_corners=True).clamp_(0, 1)

        if affine_theta is not None:
            # warp both mask and reference Csc in the same way
            m  = _affine_warp(m,  affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")
            z_r = _affine_warp(z_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

        # feather mask a bit at 24×24 (1–2 cells)
        # (implement your own tiny blur/dilate if you already have one)
        # m = gaussian_blur(m, sigma≈0.5)  # optional

        a = self.alpha_default if alpha is None else alpha
        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=z_t.dtype, device=z_t.device)
        while a.dim() < 4:  # broadcastable
            a = a.unsqueeze(0)

        # 4) simple post-proj α-blend
        z = (1 - a * m) * z_t + (a * m) * z_r


        return z


# Ianna: Combined code for blending with EfficientNet
class EfficientNetEncoderBlendCombined(nn.Module):
    """
    EfficientNet-v2-s encoder with optional masked alpha-blending:
      (1) inside the backbone at chosen block indices, and/or
      (2) after the 1x1 projection (Csc space).

    Args:
      c_latent: output channels after 1x1 mapping (Stage-B expects 16 by default).
      backbone_blend_points: list[int] | "all" | None. If None, no backbone blending.
      postproj_blend: bool — if True, also blend after 1x1 projection.
      alpha_backbone: default alpha for backbone blending (can override via per_block_alpha).
      alpha_post: alpha for post-projection blending.
      warp_mask / warp_reference: apply affine warp (theta) to mask and/or reference stream before blending.
      keep_eval: keep backbone+mapper in eval() and frozen.
    """
    def __init__(
        self,
        c_latent: int = 16,
        backbone_blend_points=None,   # e.g., [6,7] or "all" or None
        postproj_blend: bool = False,
        alpha_backbone: float = 1.0,
        alpha_post: float = 1.0,
        warp_mask: bool = False,
        warp_reference: bool = False,
        keep_eval: bool = True,
    ):
        super().__init__()
        base = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        self.backbone = base.features  # nn.Sequential of MBConv blocks
        self.out_channels = 1280
        self.c_latent = c_latent

        # 1x1 projection to Csc space (keep BatchNorm2d as requested)
        self.mapper = nn.Sequential(
            nn.Conv2d(self.out_channels, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),
        )

        # config
        if backbone_blend_points == "all":
            self.backbone_blend_indices = set(range(len(self.backbone)))
        elif backbone_blend_points is None:
            self.backbone_blend_indices = set()
        else:
            self.backbone_blend_indices = set(backbone_blend_points)

        self.postproj_blend = postproj_blend
        self.alpha_backbone = alpha_backbone
        self.alpha_post = alpha_post
        self.warp_mask = warp_mask
        self.warp_reference = warp_reference

        # freeze
        if keep_eval:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        target: torch.Tensor,                  # [B,3,H,W]
        reference: torch.Tensor,               # [B,3,H,W]
        mask: torch.Tensor | None = None,      # [B,1,H,W] in [0,1]
        affine_theta: torch.Tensor | None = None,  # [B,2,3], normalized coords
        video_length: int | None = None,
        per_block_alpha: dict[int, float | torch.Tensor] | None = None,
        alpha_post: float | None = None,
    ):
        # optional temporal tiling
        h_t = target
        h_r = reference
        if video_length is not None:
            h_t = _maybe_repeat_temporal(h_t, video_length)
            h_r = _maybe_repeat_temporal(h_r, video_length)
            if mask is not None:
                mask = _maybe_repeat_temporal(mask, video_length)
            if affine_theta is not None:
                Bf = h_t.shape[0]
                repeat_factor = Bf // affine_theta.shape[0]
                affine_theta = affine_theta.repeat_interleave(repeat_factor, dim=0)

        # -------- backbone forward with optional blending at chosen blocks --------
        for idx, block in enumerate(self.backbone):
            h_t = block(h_t)
            h_r = block(h_r)

            if mask is not None and idx in self.backbone_blend_indices:
                m = _resize_mask(mask, to_hw=h_t.shape[-2:])  # [B,1,h,w], bilinear inside helper
                if self.warp_mask and (affine_theta is not None):
                    m = _affine_warp(m, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

                hr = h_r
                if self.warp_reference and (affine_theta is not None):
                    hr = _affine_warp(h_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

                # alpha for this block
                a = self.alpha_backbone
                if per_block_alpha is not None and idx in per_block_alpha:
                    a = per_block_alpha[idx]
                if not torch.is_tensor(a):
                    a = torch.as_tensor(a, dtype=h_t.dtype, device=h_t.device)
                while a.dim() < 4:
                    a = a.unsqueeze(0)

                # masked alpha blend in backbone feature space
                h_t = (1 - a * m) * h_t + (a * m) * hr

        # -------- project both streams to Csc --------
        z_t = self.mapper(h_t)  # [B, c_latent, 24, 24] (for 1024→786 input, ends ~24x24)
        if self.postproj_blend or (mask is None):
            # we only need z_r if we plan to blend post-proj (or skip if no mask)
            z_r = self.mapper(h_r)
        else:
            z_r = None  # save a tiny compute, though you likely won’t care

        # -------- optional post-projection blending in Csc space --------
        if self.postproj_blend and (mask is not None):
            m = torch.nn.functional.interpolate(
                mask, size=z_t.shape[-2:], mode="bilinear", align_corners=True
            ).clamp_(0, 1)

            if affine_theta is not None:
                if self.warp_mask:
                    m = _affine_warp(m, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")
                if self.warp_reference:
                    z_r = _affine_warp(z_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

            a_post = self.alpha_post if alpha_post is None else alpha_post
            if not torch.is_tensor(a_post):
                a_post = torch.tensor(a_post, dtype=z_t.dtype, device=z_t.device)
            while a_post.dim() < 4:
                a_post = a_post.unsqueeze(0)

            z = (1 - a_post * m) * z_t + (a_post * m) * z_r
            return z

        # else: only backbone-blended (or no blending at all)
        return z_t


