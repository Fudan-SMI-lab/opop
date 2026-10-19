"""Download pinned public assets; run once on a machine with Hugging Face access.

Requires numpy and Pillow. These are asset-preparation dependencies only.
The GPU runner consumes pixels.npy and needs neither Pillow nor torchvision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
    with urllib.request.urlopen("https://api.github.com/repos/pytorch/vision/commits/main", timeout=30) as response:
        vision_revision = json.load(response)["sha"]
    sources = {
        name: f"https://huggingface.co/microsoft/resnet-18/resolve/{revision}/{name}"
        for name in ("config.json", "model.safetensors", "preprocessor_config.json")
    }
    sources.update({
        name: f"https://raw.githubusercontent.com/pytorch/vision/{vision_revision}/gallery/assets/{name}"
        for name in ("dog1.jpg", "dog2.jpg", "astronaut.jpg")
    })
    records = []
    for name, url in sources.items():
        target = args.output / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as writer:
                while chunk := response.read(1024 * 1024):
                    writer.write(chunk)
            temporary.replace(target)
        records.append({"file": name, "source": url, "bytes": target.stat().st_size,
                        "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest()})
        print(name, target.stat().st_size, flush=True)
    import numpy as np
    import PIL
    from PIL import Image

    pixels = []
    for name in ("dog1.jpg", "dog2.jpg", "astronaut.jpg"):
        with Image.open(args.output / name) as source:
            img = source.convert("RGB")
            width, height = img.size
            size = (int(width * 256 / min(width, height)), int(height * 256 / min(width, height)))
            img = img.resize(size, resample=Image.Resampling.BILINEAR)
            left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
            array = np.asarray(img, dtype=np.float32) / 255.0
            array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
            pixels.append(array.transpose(2, 0, 1))
    target = args.output / "pixels.npy"
    np.save(target, np.stack(pixels))
    records.append({"file": target.name, "bytes": target.stat().st_size,
                    "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest()})
    manifest = {"model": "microsoft/resnet-18", "model_revision": revision,
                "image_revision": vision_revision, "pillow_version": PIL.__version__, "numpy_version": np.__version__,
                "preprocessing": "RGB; bilinear shortest edge 256; floor-centered crop 224; ImageNet mean/std; float32 NCHW",
                "scope": "Three public example images; correctness fixture, not an ImageNet accuracy evaluation.", "files": records}
    (args.output / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
