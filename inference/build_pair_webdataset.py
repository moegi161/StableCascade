#!/usr/bin/env python3
import os, io, csv, json, math, argparse, re
from pathlib import Path
from PIL import Image
import webdataset as wds

# ---------- helpers ----------

def pil_png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_mask(path: str) -> Image.Image:
    # store masks as single-channel PNG
    return Image.open(path).convert("L")

def read_dataset_pairs(pairs_txt_path: str):
    """
    Lines look like:
    ./celebv-hq-100/frames_align/-1eKufUP5XQ_2, ./reference_100/ref_img/eyeglasses_000.png
    Return list of tuples: (target_folder_rel, ref_img_rel)
    """
    pairs = []
    with open(pairs_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            # split by comma
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            t_folder, r_img = parts
            # remove leading "./"
            if t_folder.startswith("./"): t_folder = t_folder[2:]
            if r_img.startswith("./"):    r_img = r_img[2:]
            pairs.append((t_folder, r_img))
    return pairs

def parse_target_captions(captions_txt_path: str):
    """
    File format:
      Video_ID: -1eKufUP5XQ_2
      Caption:  ...
    Returns dict {video_id: caption_str}
    """
    mapping = {}
    vid = None
    cap_lines = []
    with open(captions_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Video_ID:"):
                # flush previous
                if vid is not None and cap_lines:
                    mapping[vid] = " ".join(cap_lines).strip()
                vid = line.split(":", 1)[1].strip()
                cap_lines = []
            elif line.startswith("Caption:"):
                # first caption line
                cap = line.split(":", 1)[1].strip()
                cap_lines = [cap] if cap else []
            else:
                # possible continuation lines
                if line.strip():
                    cap_lines.append(line.strip())
        # flush last
        if vid is not None and cap_lines:
            mapping[vid] = " ".join(cap_lines).strip()
    return mapping

def parse_ref_captions(captions_txt_path: str):
    """
    File format:
      Image: eyeglasses_000.png
      Caption: ...
    Returns dict {filename: caption_str}
    """
    mapping = {}
    fname = None
    cap_lines = []
    with open(captions_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Image:"):
                if fname is not None and cap_lines:
                    mapping[fname] = " ".join(cap_lines).strip()
                fname = line.split(":", 1)[1].strip()
                cap_lines = []
            elif line.startswith("Caption:"):
                cap = line.split(":", 1)[1].strip()
                cap_lines = [cap] if cap else []
            else:
                if line.strip():
                    cap_lines.append(line.strip())
        if fname is not None and cap_lines:
            mapping[fname] = " ".join(cap_lines).strip()
    return mapping

def ensure_exists(p: str) -> bool:
    try:
        return p and os.path.exists(p)
    except Exception:
        return False

# ---------- main builder ----------

def build(
    repo_root: str,
    pairs_txt: str,
    target_captions_txt: str,
    ref_captions_txt: str,
    out_dir: str,
    shard_size: int = 1000,
):
    """
    repo_root: /mnt/workspace2024/chan/celebv-hq
    pairs_txt: celebv-hq-100/dataset_pair.txt (relative to repo_root or absolute)
    """
    repo_root = os.path.abspath(repo_root)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # read inputs
    pairs_path = pairs_txt if os.path.isabs(pairs_txt) else os.path.join(repo_root, pairs_txt)
    pairs = read_dataset_pairs(pairs_path)

    tgt_caps_path = target_captions_txt if os.path.isabs(target_captions_txt) else os.path.join(repo_root, target_captions_txt)
    ref_caps_path = ref_captions_txt  if os.path.isabs(ref_captions_txt)  else os.path.join(repo_root, ref_captions_txt)
    tgt_caps = parse_target_captions(tgt_caps_path)
    ref_caps = parse_ref_captions(ref_caps_path)

    # fixed subpaths
    ref_mask_root = os.path.join(repo_root, "reference_100", "ref_mask")

    # write shards
    nsamples = len(pairs)
    nshards = math.ceil(nsamples / shard_size)
    print(f"Found {nsamples} pairs -> {nshards} shard(s)")

    idx = 0
    shard_paths = []
    for s in range(nshards):
        shard_path = os.path.join(out_dir, f"pairs-{s:05d}.tar")
        shard_paths.append(shard_path)
        with wds.TarWriter(shard_path) as sink:
            for _ in range(shard_size):
                if idx >= nsamples:
                    break
                t_folder_rel, ref_img_rel = pairs[idx]; idx += 1

                # target folder id (e.g., "-1eKufUP5XQ_2")
                video_id = Path(t_folder_rel).name
                # target image: frame_0001.png
                target_img_abs = os.path.join(repo_root, t_folder_rel, "frame_0001.png")
                # reference image abs
                ref_img_abs = os.path.join(repo_root, ref_img_rel)
                # reference mask (same file name) under reference_100/ref_mask
                ref_fname = Path(ref_img_abs).name
                ref_mask_abs = os.path.join(ref_mask_root, ref_fname)

                # existence checks
                if not ensure_exists(target_img_abs) or not ensure_exists(ref_img_abs):
                    print(f"[skip] missing target or ref: {target_img_abs} | {ref_img_abs}")
                    continue
                if not ensure_exists(ref_mask_abs):
                    print(f"[warn] missing ref_mask, continuing without it: {ref_mask_abs}")
                    ref_mask_abs = None

                # load images
                t_img = load_rgb(target_img_abs)
                r_img = load_rgb(ref_img_abs)

                # merged caption: target + reference
                tcap = tgt_caps.get(video_id, "").strip()
                rcap = ref_caps.get(ref_fname, "").strip()
                merged_caption = (tcap + (" " if tcap and rcap else "") + rcap).strip()
                if not merged_caption:
                    # fall back to at least reference filename
                    merged_caption = f"{video_id} with {ref_fname}"

                # pack sample
                key = f"{video_id}"
                sample = {"__key__": key}
                # store images as PNG to avoid JPEG artifacts
                sample["png"] = pil_png_bytes(t_img)         # target
                sample["ref.png"] = pil_png_bytes(r_img)     # reference

                if ref_mask_abs:
                    rm = load_mask(ref_mask_abs)
                    sample["ref_mask.png"] = pil_png_bytes(rm)

                sample["txt"] = merged_caption.encode("utf-8")

                sink.write(sample)
        print(f"Wrote {shard_path}")

    # write YAML list of shards for DataCore.webdataset_path()
    yaml_path = os.path.join(out_dir, "pairs.yml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        for p in shard_paths:
            yf.write(f"- {p}\n")
    print(f"Wrote shard list: {yaml_path}")

    # quick peek info
    print("\nSample config.webdataset_path to use:")
    print(yaml_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root (e.g., /mnt/workspace2024/chan/celebv-hq)")
    ap.add_argument("--pairs", default="celebv-hq-100/dataset_pairs.txt")
    ap.add_argument("--tcap", default="celebv-hq-100/captions.txt")
    ap.add_argument("--rcap", default="reference_100/captions.txt")
    ap.add_argument("--out", required=True, help="Output dir for shards")
    ap.add_argument("--shard_size", type=int, default=1000)
    args = ap.parse_args()
    build(args.root, args.pairs, args.tcap, args.rcap, args.out, shard_size=args.shard_size)
