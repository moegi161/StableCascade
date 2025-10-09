#!/usr/bin/env python3
import os
import io
import re
import math
import json
import argparse
from pathlib import Path
from PIL import Image
import webdataset as wds

# ----------------------------
# Parser for your captions.txt
# ----------------------------
def parse_captions(captions_path: Path):
    """
    Expects blocks like:
      Image: eyeglasses_000.png
      Caption: The eyeglasses ...

    Returns dict: { "eyeglasses_000.png": "The eyeglasses ..." , ... }
    """
    text = captions_path.read_text(encoding="utf-8", errors="ignore")
    # Split on lines; be robust to extra blank lines / spaces
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    mapping = {}
    img_pat = re.compile(r"^Image:\s*(.+)$", re.IGNORECASE)
    cap_pat = re.compile(r"^Caption:\s*(.+)$", re.IGNORECASE)

    i = 0
    while i < len(lines):
        m_img = img_pat.match(lines[i])
        if not m_img:
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        m_cap = cap_pat.match(lines[i + 1])
        if not m_cap:
            # Try to find the next caption line within a couple of lines
            found = False
            for j in range(i + 1, min(i + 5, len(lines))):
                m_cap = cap_pat.match(lines[j])
                if m_cap:
                    i_cap = j
                    found = True
                    break
            if not found:
                i += 1
                continue
        else:
            i_cap = i + 1

        fname = m_img.group(1).strip()
        caption = m_cap.group(1).strip()
        mapping[fname] = caption
        i = i_cap + 1

    return mapping


# ----------------------------
# JPEG encode utility
# ----------------------------
def png_to_jpeg_bytes(png_path: Path, quality=95):
    img = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

# --- replace the JPEG function with a PNG one ---
def image_to_png_bytes(img_path: Path, optimize=True):
    from PIL import Image
    import io
    img = Image.open(img_path).convert("RGB")  # drop alpha for consistency
    buf = io.BytesIO()
    # optimize=True shrinks file a bit without loss
    img.save(buf, format="PNG", optimize=optimize)
    return buf.getvalue()




# ----------------------------
# Build WebDataset shards
# ----------------------------
def build_shards(
    images_dir: Path,
    captions_path: Path,
    out_dir: Path,
    shard_size: int = 1000,
    quality: int = 95,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = parse_captions(captions_path)

    # Collect (image_path, caption) pairs that exist
    items = []
    for fname, cap in captions.items():
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"[WARN] Missing image for caption: {img_path}")
            continue
        items.append((img_path, cap))

    if not items:
        raise RuntimeError("No valid (image, caption) pairs found.")

    print(f"[INFO] Found {len(items)} items. Writing shards to: {out_dir}")

    # Pattern like: eyeglasses-%06d.tar
    pattern = str(out_dir / "eyeglasses_%06d.tar")
    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        for idx, (img_path, caption) in enumerate(items):
            stem = img_path.stem  # e.g., "eyeglasses_000"
            try:
                png_bytes = image_to_png_bytes(img_path)
            except Exception as e:
                print(f"[ERROR] Failed to convert {img_path}: {e}")
                continue

            sample = {
                "__key__": stem,  # results in stem.png, stem.txt inside the tar
                "png": png_bytes,
                "txt": caption.encode("utf-8"),
            }
            sink.write(sample)

            if (idx + 1) % 100 == 0:
                print(f"[INFO] Wrote {idx + 1} samples...")

    # Write a simple manifest for reproducibility
    manifest = {
        "images_dir": str(images_dir),
        "captions_path": str(captions_path),
        "out_dir": str(out_dir),
        "total_items": len(items),
        "shard_size": shard_size,
        "jpeg_quality": quality,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DONE] Wrote shards and manifest to {out_dir}")


# ----------------------------
# Optional: quick validator
# ----------------------------
def quick_validate(out_dir: Path, take: int = 3):
    """
    Try a tiny read to verify the format.
    """
    urls = [str(p) for p in sorted(out_dir.glob("*.tar"))]
    if not urls:
        print("[VALIDATE] No .tar shards found.")
        return
    print(f"[VALIDATE] Using {urls[0]} (showing {take} samples)")
    ds = (
        wds.WebDataset(urls[0], resampled=False)
        .decode("pilrgb")               
        .to_tuple("png", "txt")
    )
    n = 0
    for img, txt in ds:
        n += 1
        # no GUI: just print shapes / first 60 chars
        size = getattr(img, "size", None)
        print(f"  sample#{n}: image_size={size}, caption={txt[:60]!r}")
        if n >= take:
            break


def main():
    ap = argparse.ArgumentParser(description="Build WebDataset shards from eyeglasses PNGs + captions.txt")
    ap.add_argument("--images_dir", type=Path, required=True,
                    help="Folder with input images (PNGs)")
    ap.add_argument("--captions", type=Path, required=True,
                    help="captions.txt file (Image:/Caption: blocks)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Output folder for .tar shards")
    ap.add_argument("--shard_size", type=int, default=1000,
                    help="Max samples per shard (.tar)")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="JPEG quality for conversion")
    ap.add_argument("--validate", action="store_true",
                    help="Run a quick read test on the first shard")
    args = ap.parse_args()

    build_shards(
        images_dir=args.images_dir,
        captions_path=args.captions,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        quality=args.jpeg_quality,
    )

    if args.validate:
        quick_validate(args.out_dir)


if __name__ == "__main__":
    main()
