"""Download phototourism benchmark scenes for evaluation.

Scenes available from the Image Matching Challenge:
  - sacre_coeur, reichstag, st_peters_square (validation set)

Usage:
    python scripts/download_benchmark.py --scene sacre_coeur
    python scripts/download_benchmark.py --scene all
"""

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# Known scenes from the IMW2020 benchmark
SCENES = {
    "sacre_coeur": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/sacre_coeur.tar.gz",
    "reichstag": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/reichstag.tar.gz",
    "st_peters_square": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/st_peters_square.tar.gz",
}

DEFAULT_OUTPUT = Path("data") / "phototourism"


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  Downloading: {mb:.1f} MB", end="", flush=True)


def download_scene(scene_name, output_dir):
    if scene_name not in SCENES:
        print(f"Unknown scene: {scene_name}")
        print(f"Available: {', '.join(SCENES.keys())}")
        sys.exit(1)

    url = SCENES[scene_name]
    output_dir = Path(output_dir)
    scene_dir = output_dir / scene_name

    if scene_dir.is_dir() and (
        (scene_dir / "calibration.h5").is_file()
        or (scene_dir / "calibration.json").is_file()
        or (scene_dir / "dense" / "sparse" / "cameras.bin").is_file()
    ):
        print(f"  Scene {scene_name} already exists at {scene_dir}")
        return scene_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = url.split("/")[-1]
    archive_path = output_dir / archive_name

    print(f"Downloading {scene_name} from {url}")
    urlretrieve(url, str(archive_path), reporthook=download_progress)
    print()

    print(f"  Extracting {archive_name}...")
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(str(output_dir))
    elif archive_name.endswith(".zip"):
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            zf.extractall(str(output_dir))

    # Clean up archive
    archive_path.unlink()

    print(f"  Scene ready at {scene_dir}")
    return scene_dir


def main():
    p = argparse.ArgumentParser(description="Download phototourism benchmark scenes")
    p.add_argument("--scene", type=str, default="sacre_coeur",
                   choices=list(SCENES.keys()) + ["all"],
                   help="Scene to download (default: sacre_coeur)")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                   help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = p.parse_args()

    if args.scene == "all":
        for name in SCENES:
            download_scene(name, args.output)
    else:
        download_scene(args.scene, args.output)

    print("\nDone. Run the benchmark with:")
    print(f"  python src/main.py --benchmark --benchmark-scene {args.output}/<scene_name>")


if __name__ == "__main__":
    main()
