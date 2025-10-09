import webdataset as wds
from PIL import Image
import torch

tar_path = "/mnt/workspace2025/chan/StableCascade/data/celebv-100/pairs-00000.tar"

# decode PIL images automatically
dataset = wds.WebDataset(tar_path).decode("pil")

for i, sample in enumerate(dataset):
    print(f"--- Sample {i} ---")
    print("Keys:", list(sample.keys()))
    if "png" in sample:
        print("  target:", sample["png"].size)
    if "ref.png" in sample:
        print("  ref:", sample["ref.png"].size)
    if "ref_mask.png" in sample:
        print("  ref_mask:", sample["ref_mask.png"].size)
    if "txt" in sample:
        print("  caption:", sample["txt"])   # print first 200 chars
    if i >= 2:
        break
