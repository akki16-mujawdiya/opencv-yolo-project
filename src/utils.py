"""
utils.py - small helpers that every task script uses.

Nothing clever in here: finding the input files, loading/resizing images,
saving figures and writing CSV/JSON results.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # no pop-up windows, we only save figures to disk
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


# ----------------------------------------------------------------------------
# files and folders
# ----------------------------------------------------------------------------
def find_input(stem, exts):
    """Look inside data/ for a file called <stem>.<any allowed extension>."""
    if not DATA_DIR.exists():
        return None
    for p in sorted(DATA_DIR.iterdir()):
        if p.stem.lower() == stem and p.suffix.lower() in exts:
            return p
    return None


def need_input(stem, exts, what):
    """Same as find_input but stops with a friendly message if it is missing."""
    p = find_input(stem, exts)
    if p is None:
        raise SystemExit(
            f"\n[!] Could not find {what}.\n"
            f"    Put a file named '{stem}' (one of {', '.join(exts)}) inside the data/ folder.\n"
            f"    See data/README.md for details.\n"
        )
    return p


def task_dir(name):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config():
    cfg = {
        "student_name": "Student",
        "prn": "",
        "course": "Computer Vision - CCA 3",
        "git_url": "",
        "explanation_video_url": "",
        "drive_folder_url": "",
    }
    path = ROOT / "config.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------
def load_image(path, max_side=900):
    """Read an image (BGR). Big phone photos are shrunk so everything runs fast."""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"[!] OpenCV could not read '{path}'. Is it a valid jpg/png?")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def to_rgb(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(path, img):
    cv2.imwrite(str(path), img)


def save_grid(images, titles, path, cols=3, cell=(5, 4), suptitle=None):
    """Put several images into one labelled figure and save it."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cell[0] * cols, cell[1] * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255 if im.dtype == np.uint8 else None)
        else:
            ax.imshow(to_rgb(im))
        ax.set_title(t, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# video
# ----------------------------------------------------------------------------
def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(
            f"[!] Could not open video '{path}'.\n"
            "    If it is an iPhone .MOV, convert a copy with ffmpeg (see data/README.md)."
        )
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0
    return cap, info


def read_frame_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
