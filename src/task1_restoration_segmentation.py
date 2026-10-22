"""
Task 1 (Unit 3) - Image restoration and segmentation

Part A : add Gaussian + salt-and-pepper noise, restore with median and Gaussian
         filters, compare with the clean photo using PSNR and SSIM.
Part B : thresholding, erosion / dilation / opening / closing, Canny edges and
         watershed segmentation.

Run:  python src/task1_restoration_segmentation.py
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils import (IMAGE_EXTS, find_input, load_image, need_input, save_grid,
                   save_img, task_dir, write_csv, write_json)


# ----------------------------------------------------------------------------
# Part A - noise and restoration
# ----------------------------------------------------------------------------
def add_gaussian_noise(img, sigma=25, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.05, seed=1):
    """amount = fraction of pixels that get corrupted (half salt, half pepper)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0                       # pepper
    out[(r >= amount / 2) & (r < amount)] = 255   # salt
    return out


FILTERS = {
    "Median 3x3": lambda x: cv2.medianBlur(x, 3),
    "Median 5x5": lambda x: cv2.medianBlur(x, 5),
    "Gaussian 5x5": lambda x: cv2.GaussianBlur(x, (5, 5), 1.2),
}


def center_crop(img, size=160):
    h, w = img.shape[:2]
    y0, x0 = max(0, h // 2 - size // 2), max(0, w // 2 - size // 2)
    crop = img[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)


def run_restoration(img, out):
    noisy_versions = {
        "Gaussian noise": ("gaussian", add_gaussian_noise(img)),
        "Salt & pepper noise": ("saltpepper", add_salt_pepper(img)),
    }
    rows = []
    for noise_name, (tag, noisy) in noisy_versions.items():
        images = [img, noisy]
        titles = ["Original (clean)"]
        p0, s0 = psnr(img, noisy), ssim(img, noisy, channel_axis=2)
        titles.append(f"{noise_name}\nPSNR {p0:.2f} dB | SSIM {s0:.3f}")
        rows.append([noise_name, "Noisy (no filter)", round(p0, 2), round(s0, 4)])
        save_img(out / f"{tag}_noisy.png", noisy)

        for fname, fn in FILTERS.items():
            restored = fn(noisy)
            p, s = psnr(img, restored), ssim(img, restored, channel_axis=2)
            rows.append([noise_name, fname, round(p, 2), round(s, 4)])
            images.append(restored)
            titles.append(f"{fname}\nPSNR {p:.2f} dB | SSIM {s:.3f}")
            save_img(out / f"{tag}_{fname.lower().replace(' ', '_')}.png", restored)

        save_grid(images, titles, out / f"restoration_{tag}.png", cols=3,
                  suptitle=f"Restoration of {noise_name.lower()}")
        # zoomed crops make the difference between filters much easier to see
        save_grid([center_crop(i) for i in images],
                  ["Original", "Noisy"] + list(FILTERS.keys()),
                  out / f"restoration_{tag}_zoom.png", cols=5, cell=(3.2, 3.4),
                  suptitle=f"Zoomed centre crop - {noise_name.lower()}")

    write_csv(out / "restoration_metrics.csv",
              ["Noise type", "Image", "PSNR (dB)", "SSIM"], rows)
    return rows


# ----------------------------------------------------------------------------
# Part B - thresholding, morphology, edges
# ----------------------------------------------------------------------------
def run_threshold_morph_edges(img, out):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_val, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
    save_grid([gray, fixed, otsu, adaptive],
              ["Grayscale", "Global threshold (T=127)",
               f"Otsu threshold (T={otsu_val:.0f})", "Adaptive Gaussian threshold"],
              out / "thresholding.png", cols=4, cell=(4.2, 3.6),
              suptitle="Binary images from different thresholding methods")
    save_img(out / "binary_otsu.png", otsu)

    # morphology works on white objects, so make the smaller region white
    binary = otsu if np.mean(otsu) < 127 else cv2.bitwise_not(otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    save_grid([binary, erosion, dilation, opening, closing],
              ["Binary mask (Otsu)", "Erosion", "Dilation",
               "Opening (erode->dilate)", "Closing (dilate->erode)"],
              out / "morphology.png", cols=5, cell=(3.6, 3.2),
              suptitle="Morphological operations (5x5 elliptical kernel)")
    for name, im in [("erosion", erosion), ("dilation", dilation),
                     ("opening", opening), ("closing", closing)]:
        save_img(out / f"morph_{name}.png", im)

    # Canny: one fixed pair of thresholds and one picked automatically from the median
    canny_fixed = cv2.Canny(blur, 100, 200)
    med = np.median(blur)
    lo, hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    canny_auto = cv2.Canny(blur, lo, hi)
    save_grid([canny_fixed, canny_auto],
              ["Canny (100, 200)", f"Canny auto ({lo}, {hi})"],
              out / "canny_edges.png", cols=2, cell=(6, 4.5),
              suptitle="Canny edge detection")
    save_img(out / "canny_fixed.png", canny_fixed)

    return {
        "otsu_threshold": float(otsu_val),
        "canny_auto_low": lo,
        "canny_auto_high": hi,
        "edge_pixels_fixed_pct": round(100 * np.count_nonzero(canny_fixed) / canny_fixed.size, 2),
        "edge_pixels_auto_pct": round(100 * np.count_nonzero(canny_auto) / canny_auto.size, 2),
        "foreground_pct_after_opening": round(100 * np.count_nonzero(opening) / opening.size, 2),
    }


# ----------------------------------------------------------------------------
# Part B - watershed
# ----------------------------------------------------------------------------
def run_watershed(img, out):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # decide which side is "object": the image border is usually background
    border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:
        bw = cv2.bitwise_not(bw)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

    # a blob covering a big chunk of the image is almost certainly uneven
    # lighting / background, not an object - drop it
    n_cc, cc, cc_stats, _ = cv2.connectedComponentsWithStats(opening)
    for i in range(1, n_cc):
        if cc_stats[i, cv2.CC_STAT_AREA] > 0.25 * opening.size:
            opening[cc == i] = 0

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # threshold the distance map separately inside every blob (50% of that blob's
    # own maximum) so small and large objects both get a marker
    sure_fg = np.zeros_like(opening)
    n_cc, cc = cv2.connectedComponents(opening)
    for i in range(1, n_cc):
        region = cc == i
        sure_fg[region & (dist > 0.5 * dist[region].max())] = 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background becomes 1, objects 2, 3, ...
    markers[unknown == 255] = 0      # 0 = "let watershed decide"
    markers = cv2.watershed(img.copy(), markers)

    result = img.copy()
    result[markers == -1] = (0, 0, 255)  # watershed lines in red
    n_segments = len([m for m in np.unique(markers) if m > 1])

    # random colours for the label map
    rng = np.random.default_rng(3)
    colours = rng.integers(60, 255, size=(markers.max() + 2, 3), dtype=np.uint8)
    label_img = np.zeros_like(img)
    for m in range(2, markers.max() + 1):
        label_img[markers == m] = colours[m]
    label_img[markers == -1] = (255, 255, 255)

    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_grid([img, bw, dist_vis, sure_fg, label_img, result],
              ["Input image", "Binary mask (objects white)", "Distance transform",
               "Sure foreground (markers)", "Segment labels",
               f"Watershed result ({n_segments} segments)"],
              out / "watershed_steps.png", cols=3, cell=(5, 4),
              suptitle="Watershed segmentation step by step")
    save_img(out / "watershed_result.png", result)
    return n_segments


# ----------------------------------------------------------------------------
def main():
    out = task_dir("task1")
    photo_path = need_input("photo", IMAGE_EXTS, "your original photograph")
    img = load_image(photo_path)
    print(f"[task1] using {photo_path.name}  ({img.shape[1]}x{img.shape[0]})")
    save_img(out / "input_photo.png", img)

    print("[task1] part A - noise and restoration")
    rows = run_restoration(img, out)

    print("[task1] part B - threshold, morphology, canny")
    info = run_threshold_morph_edges(img, out)

    print("[task1] part B - watershed")
    obj_path = find_input("objects", IMAGE_EXTS)
    ws_img = load_image(obj_path, max_side=700) if obj_path else img
    ws_source = obj_path.name if obj_path else photo_path.name
    if obj_path:
        save_img(out / "input_objects.png", ws_img)
    n_segments = run_watershed(ws_img, out)

    info.update({
        "photo_file": photo_path.name,
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "watershed_source": ws_source,
        "watershed_segments": n_segments,
    })
    write_json(out / "summary.json", info)
    print("[task1] done ->", out)


if __name__ == "__main__":
    main()
