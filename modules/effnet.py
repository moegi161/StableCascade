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


# helpers you already have:
# _resize_mask(mask, to_hw)
# _affine_warp(tensor, theta, mode="bilinear", align_corners=True, padding_mode="zeros")
# _maybe_repeat_temporal(x, video_length)

class TinyCscBottleneck(nn.Module):
    """
    No-training per-location low-rank projector in Csc space (B,C,H,W).
    Use up = down^T so it acts like a PCA-ish projector without training.
    """
    def __init__(self, c_in: int, c_mid: int = 8, gain: float = 0.3, add_noise: bool = False, sigma: float = 0.06):
        super().__init__()
        self.ln = nn.GroupNorm(1, c_in, affine=False, eps=1e-6)  # LN-style, BN-free
        self.down = nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False)
        self.up   = nn.Conv2d(c_mid, c_in, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.down.weight)
        with torch.no_grad():
            self.up.weight.copy_(self.down.weight.permute(1,0,2,3))  # up = down^T
        self.register_buffer("gain", torch.tensor(gain), persistent=False)
        self.add_noise = add_noise
        self.sigma = float(sigma)

    @torch.inference_mode()
    def forward(self, z):  # z: [B,C,H,W]
        x = self.ln(z)
        x = self.down(x)
        x = self.up(x)
        out = z + self.gain * x
        if self.add_noise and self.sigma > 0:
            out = out + self.sigma * torch.randn_like(out)
        return out

class GlobalFlattenBottleneck(nn.Module):
    def __init__(self, C:int, H:int, W:int, d:int=256, gain:float=0.2, add_noise:bool=False, sigma:float=0.06):
        super().__init__()
        D = C*H*W
        self.C, self.H, self.W = C, H, W
        self.ln = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)  # BN-free, stable across batch sizes
        self.down = nn.Linear(D, d, bias=False)
        self.up   = nn.Linear(d, D, bias=False)
        nn.init.orthogonal_(self.down.weight)
        with torch.no_grad():
            self.up.weight.copy_(self.down.weight.t())  # up = down^T  (low-rank projector)
        self.register_buffer("gain", torch.tensor(gain), persistent=False)
        self.add_noise = add_noise
        self.sigma = float(sigma)

    @torch.inference_mode()
    def forward(self, z):  # z: [B,C,H,W]
        B, C, H, W = z.shape
        x = z.view(B, -1)                 # [B, D]
        x = self.ln(x)
        y = self.up(self.down(x))         # [B, D]
        y = y.view(B, C, H, W)
        out = z + self.gain * y
        if self.add_noise and self.sigma > 0:
            out = out + self.sigma * torch.randn_like(out)
        return out


class EfficientNetEncoderBlendCombined(nn.Module):
    """
    EfficientNet-v2-s encoder with:
      (A) optional masked alpha-blend at chosen backbone indices (shallow recommended),
      (B) 1x1 projection to Csc (BatchNorm2d kept),
      (C) optional post-projection bottleneck (no-training low-rank) in Csc,
      (D) optional post-projection masked blend (z-space).

    Typical use for your findings:
      backbone_blend_points=[0], alpha_backbone≈0.3–0.6
      post_bottleneck=True with c_mid=8 or 12, gain=0.2–0.4, add_noise=True, sigma≈0.06
      postproj_blend=True with alpha_post≈0.4–0.8
    """
    def __init__(
        self,
        c_latent: int = 16,
        backbone_blend_points=None,              # e.g., [0] or None
        alpha_backbone: float = 0.5,
        postproj_blend: bool = True,
        alpha_post: float = 0.6,
        post_bottleneck: bool = True,
        c_mid_bottleneck: int = 8,
        bottleneck_gain: float = 0.3,
        bottleneck_add_noise: bool = False,
        bottleneck_sigma: float = 0.06,
        warp_mask: bool = False,
        warp_reference: bool = False,
        keep_eval: bool = True,
    ):
        super().__init__()
        base = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        self.backbone = base.features
        self.out_channels = 1280
        self.c_latent = c_latent

        # 1x1 projection to Csc (keep BatchNorm2d)
        self.mapper = nn.Sequential(
            nn.Conv2d(self.out_channels, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),
        )

        # config
        self.backbone_blend_indices = set(backbone_blend_points or [])
        self.alpha_backbone = alpha_backbone
        self.postproj_blend = postproj_blend
        self.alpha_post = alpha_post
        self.warp_mask = warp_mask
        self.warp_reference = warp_reference

        self.post_bottleneck = post_bottleneck
        if post_bottleneck:
            
            self.csc_bottleneck = GlobalFlattenBottleneck(
                C=c_latent, H=24, W=24,  # fixed for EfficientNet-v2-s
            )

        if keep_eval:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        target: torch.Tensor,                  # [B,3,H,W]
        reference: torch.Tensor,               # [B,3,H,W]
        mask: torch.Tensor | None = None,      # [B,1,H,W]
        affine_theta: torch.Tensor | None = None,  # [B,2,3]
        video_length: int | None = None,
        per_block_alpha: dict[int, float | torch.Tensor] | None = None,
        alpha_post: float | None = None,
        cfg_scale: float | None = None,        # optional CFG scaling on Csc
    ):
        # optional temporal tiling
        h_t, h_r = target, reference
        if video_length is not None:
            h_t = _maybe_repeat_temporal(h_t, video_length)
            h_r = _maybe_repeat_temporal(h_r, video_length)
            if mask is not None:
                mask = _maybe_repeat_temporal(mask, video_length)
            if affine_theta is not None:
                Bf = h_t.shape[0]
                affine_theta = affine_theta.repeat_interleave(Bf // affine_theta.shape[0], dim=0)

        # ---------- shallow backbone with optional blend ----------
        for idx, block in enumerate(self.backbone):
            h_t = block(h_t)
            h_r = block(h_r)

            if mask is not None and idx in self.backbone_blend_indices:
                m = _resize_mask(mask, to_hw=h_t.shape[-2:])
                if self.warp_mask and (affine_theta is not None):
                    m = _affine_warp(m, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")
                hr = h_r
                if self.warp_reference and (affine_theta is not None):
                    hr = _affine_warp(h_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

                a = self.alpha_backbone
                if per_block_alpha is not None and idx in per_block_alpha:
                    a = per_block_alpha[idx]
                if not torch.is_tensor(a):
                    a = torch.as_tensor(a, dtype=h_t.dtype, device=h_t.device)
                while a.dim() < 4: a = a.unsqueeze(0)

                h_t = (1 - a * m) * h_t + (a * m) * hr

        # ---------- map to Csc ----------
        z_t = self.mapper(h_t)
        z_r = self.mapper(h_r) if (self.postproj_blend or (mask is None)) else None

        # ---------- post-projection bottleneck (no-training) ----------
        if self.post_bottleneck:
            print(f"z_t ori norm: {z_t.norm().item():.4f}, z_t ori var: {z_t.var().item():.4f}")
            z_t = self.csc_bottleneck(z_t)
            print("Applied Csc bottleneck")
            print(f"z_t norm: {z_t.norm().item():.4f}, z_t var: {z_t.var().item():.4f}")
        

        # ---------- optional post-projection blend in Csc ----------
        if self.postproj_blend and (mask is not None):
            m = torch.nn.functional.interpolate(mask, size=z_t.shape[-2:], mode="bilinear", align_corners=True).clamp_(0, 1)
            if affine_theta is not None:
                if self.warp_mask:
                    m = _affine_warp(m, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")
                if self.warp_reference:
                    z_r = _affine_warp(z_r, affine_theta, mode="bilinear", align_corners=True, padding_mode="zeros")

            a_post = self.alpha_post if alpha_post is None else alpha_post
            if not torch.is_tensor(a_post):
                a_post = torch.tensor(a_post, dtype=z_t.dtype, device=z_t.device)
            while a_post.dim() < 4: a_post = a_post.unsqueeze(0)

            z = (1 - a_post * m) * z_t + (a_post * m) * z_r
        else:
            z = z_t

        # ---------- optional CFG on Csc ----------
        if cfg_scale is not None and cfg_scale != 1.0:
            z_null = torch.zeros_like(z)
            z = z_null + float(cfg_scale) * (z - z_null)

        return z
