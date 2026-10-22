"""
Task 2 (Unit 4) - Feature extraction and image matching

1. HOG features + visualisation
2. SIFT keypoints and descriptors
3. SIFT matching between the two photos (Lowe ratio test + RANSAC)
4. Repeat after rotating / resizing the second photo
5. Compare the matching quality in a table

Run:  python src/task2_features.py
"""
import cv2
import numpy as np
from skimage import exposure
from skimage.feature import hog

from utils import (IMAGE_EXTS, find_input, load_image, need_input, save_grid,
                   save_img, task_dir, write_csv, write_json)

RATIO = 0.75  # Lowe's ratio test threshold


# ----------------------------------------------------------------------------
# HOG
# ----------------------------------------------------------------------------
def hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))  # fixed size -> fixed length descriptor
    vec, vis = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=True, channel_axis=None)
    vis = exposure.rescale_intensity(vis, in_range=(0, vis.max() + 1e-9), out_range=(0, 255))
    return gray, vec, vis.astype(np.uint8)


# ----------------------------------------------------------------------------
# SIFT + matching
# ----------------------------------------------------------------------------
def sift_detect(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_sift(kp1, des1, kp2, des2):
    """Ratio-test matching, then RANSAC homography to count geometric inliers."""
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return [], 0, np.zeros(0, dtype=np.uint8)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    good.sort(key=lambda m: m.distance)

    inliers, mask = 0, np.zeros(len(good), dtype=np.uint8)
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, m = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if m is not None:
            mask = m.ravel().astype(np.uint8)
            inliers = int(mask.sum())
    return good, inliers, mask


def rotate_keep_all(img, angle):
    """Rotate around the centre and enlarge the canvas so nothing gets cut off."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, M, (new_w, new_h))


def draw_matches_colored(img1, kp1, img2, kp2, matches, inlier_mask, max_draw=80):
    """Side-by-side canvas; RANSAC inliers drawn in green, outliers in red."""
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((h, img1.shape[1] + img2.shape[1], 3), np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2
    offset = img1.shape[1]
    for m, ok in list(zip(matches, inlier_mask))[:max_draw]:
        x1, y1 = map(int, kp1[m.queryIdx].pt)
        x2, y2 = kp2[m.trainIdx].pt
        x2, y2 = int(x2) + offset, int(y2)
        colour = (0, 200, 0) if ok else (0, 0, 255)
        cv2.line(canvas, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, colour, -1)
        cv2.circle(canvas, (x2, y2), 3, colour, -1)
    return canvas


def make_fallback_second_image(img):
    """Only used when the student has not supplied photo2 yet (so the code still runs)."""
    h, w = img.shape[:2]
    crop = img[int(0.05 * h):int(0.95 * h), int(0.08 * w):int(0.98 * w)]
    crop = cv2.convertScaleAbs(crop, alpha=0.9, beta=15)
    return cv2.resize(crop, (int(w * 0.9), int(h * 0.9)))


# ----------------------------------------------------------------------------
def main():
    out = task_dir("task2")
    img1 = load_image(need_input("photo", IMAGE_EXTS, "your original photograph"), max_side=800)
    p2 = find_input("photo2", IMAGE_EXTS)
    if p2 is None:
        print("\n[!] WARNING: data/photo2.* not found - creating a fake second image from photo.\n"
              "    Add your real second photo of the same object/scene before submitting!\n")
        img2 = make_fallback_second_image(img1)
        second_name = "(auto-generated placeholder)"
    else:
        img2 = load_image(p2, max_side=800)
        second_name = p2.name
    save_img(out / "input_photo1.png", img1)
    save_img(out / "input_photo2.png", img2)

    # ---- 1. HOG -----------------------------------------------------------
    print("[task2] HOG")
    g1, v1, hog1 = hog_features(img1)
    g2, v2, hog2 = hog_features(img2)
    save_grid([g1, hog1, g2, hog2],
              ["Photo 1 (gray, 256x256)", "Photo 1 - HOG visualisation",
               "Photo 2 (gray, 256x256)", "Photo 2 - HOG visualisation"],
              out / "hog_visualisation.png", cols=4, cell=(4, 4),
              suptitle="Histogram of Oriented Gradients (9 bins, 16x16 cells, 2x2 blocks)")
    save_img(out / "hog_photo1.png", hog1)
    save_img(out / "hog_photo2.png", hog2)
    hog_similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

    # ---- 2. SIFT keypoints --------------------------------------------------
    print("[task2] SIFT keypoints")
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift_detect(img1, sift)
    kp2, des2 = sift_detect(img2, sift)
    vis_kp1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_kp2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_grid([vis_kp1, vis_kp2],
              [f"Photo 1 - {len(kp1)} SIFT keypoints", f"Photo 2 - {len(kp2)} SIFT keypoints"],
              out / "sift_keypoints.png", cols=2, cell=(7, 5.5),
              suptitle="SIFT keypoints (circle size = scale, line = orientation)")

    # ---- 3 + 4. matching experiments ---------------------------------------
    experiments = [
        ("A. Original vs original", img2),
        ("B. Photo 2 rotated 45 deg", rotate_keep_all(img2, 45)),
        ("C. Photo 2 resized to 50%", cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
        ("D. Photo 2 rotated 30 deg + 70% size",
         cv2.resize(rotate_keep_all(img2, 30), None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)),
    ]
    rows, vis_imgs, vis_titles = [], [], []
    for i, (name, im) in enumerate(experiments):
        print(f"[task2] matching - {name}")
        kpb, desb = sift_detect(im, sift)
        good, inliers, mask = match_sift(kp1, des1, kpb, desb)
        avg_dist = float(np.mean([m.distance for m in good])) if good else 0.0
        inlier_ratio = inliers / len(good) if good else 0.0
        rows.append([name, len(kp1), len(kpb), len(good), inliers,
                     f"{100 * inlier_ratio:.1f}%", round(avg_dist, 1)])

        canvas = draw_matches_colored(img1, kp1, im, kpb, good, mask)
        save_img(out / f"matches_{chr(ord('A') + i)}.png", canvas)
        vis_imgs.append(canvas)
        vis_titles.append(f"{name}\n{len(good)} good matches, {inliers} inliers "
                          f"({100 * inlier_ratio:.0f}%)")

    save_grid(vis_imgs, vis_titles, out / "sift_matching_comparison.png", cols=2,
              cell=(9, 5), suptitle="SIFT matching (green = RANSAC inlier, red = outlier)")

    write_csv(out / "matching_results.csv",
              ["Experiment", "Keypoints (photo 1)", "Keypoints (photo 2 version)",
               "Good matches (ratio test)", "RANSAC inliers", "Inlier ratio",
               "Avg match distance"], rows)
    write_json(out / "summary.json", {
        "photo2_file": second_name,
        "hog_vector_length": int(len(v1)),
        "hog_cosine_similarity": round(hog_similarity, 3),
        "ratio_test": RATIO,
        "keypoints_photo1": len(kp1),
        "keypoints_photo2": len(kp2),
    })
    print("[task2] done ->", out)


if __name__ == "__main__":
    main()
