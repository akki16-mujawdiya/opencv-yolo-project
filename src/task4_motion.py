"""
Task 4 (Unit 5) - Video motion estimation

1. extract consecutive frames from the video
2. frame differencing
3. GMM background subtraction (OpenCV MOG2)
4. Lucas-Kanade sparse optical flow
5. save foreground masks and motion-vector images
6. compare the three methods (table + side-by-side video)

Run:  python src/task4_motion.py
"""
import time

import cv2
import numpy as np

from utils import (VIDEO_EXTS, need_input, open_video, resize_to_width, save_grid,
                   save_img, task_dir, write_csv, write_json)

WORK_WIDTH = 640      # frames are shrunk to this width for speed
START_FRACTION = 0.4  # start analysing 40% into the clip
N_PAIRS = 4           # consecutive frame pairs shown in the figure
DEMO_FRAMES = 150     # length of the side-by-side demo video
WARMUP = 120          # frames fed to the background model before the start frame
MIN_AREA = 150        # ignore blobs smaller than this many pixels

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FEATURE_PARAMS = dict(maxCorners=250, qualityLevel=0.01, minDistance=8, blockSize=7)


def prep(frame):
    return resize_to_width(frame, WORK_WIDTH)


# ----------------------------------------------------------------------------
# the three methods
# ----------------------------------------------------------------------------
def frame_difference(prev_gray, gray, thresh=25):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return mask


def gmm_mask(subtractor, frame):
    raw = subtractor.apply(frame)
    _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)  # 127 = shadow, drop it
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def lucas_kanade(prev_gray, gray):
    """Track corner features between two frames.
    Returns old points, new points, per-point motion after removing camera shake."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    if p0 is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
    st = st.ravel() == 1
    old, new = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
    if len(old) == 0:
        return old, new, np.zeros(0)
    flow = new - old
    camera = np.median(flow, axis=0)  # most points are background -> median = camera motion
    residual = np.linalg.norm(flow - camera, axis=1)
    return old, new, residual


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def count_blobs(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA)


def overlay_mask(frame, mask, colour):
    out = frame.copy()
    out[mask > 0] = (0.4 * out[mask > 0] + 0.6 * np.array(colour)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return out


def draw_flow(frame, old, new, residual, moving_thresh=1.0, arrow_scale=5):
    """Arrows are stretched x5 because motion between consecutive frames is tiny."""
    out = frame.copy()
    for (x0, y0), (x1, y1), r in zip(old, new, residual):
        moving = r > moving_thresh
        colour = (0, 0, 255) if moving else (0, 200, 0)
        end = (int(x0 + (x1 - x0) * arrow_scale), int(y0 + (y1 - y0) * arrow_scale))
        cv2.arrowedLine(out, (int(x0), int(y0)), end, colour, 1, tipLength=0.3)
        cv2.circle(out, (int(x0), int(y0)), 2, colour, -1)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
def main():
    out = task_dir("task4")
    for sub in ("frames", "masks_diff", "masks_gmm", "flow"):
        (out / sub).mkdir(exist_ok=True)

    cap, info = open_video(need_input("video", VIDEO_EXTS, "your video"))
    total = info["frames"]
    start = int(total * START_FRACTION)
    last = min(total - 2, start + max(DEMO_FRAMES, N_PAIRS + 1))
    warm_start = max(0, start - WARMUP)
    print(f"[task4] video: {info['width']}x{info['height']} @ {info['fps']:.1f} fps, {total} frames")
    print(f"[task4] analysing frames {start} to {last} (background model warm-up from {warm_start})")

    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)

    writer = None
    prev_gray = None
    stats = {"diff": [], "gmm": [], "lk": []}  # per-frame measurements
    still_rows = []

    idx = warm_start
    while idx <= last:
        ok, raw = cap.read()
        if not ok:
            break
        frame = prep(raw)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        t0 = time.perf_counter()
        g_mask = gmm_mask(subtractor, frame)
        t_gmm = (time.perf_counter() - t0) * 1000

        if idx >= start and prev_gray is not None:
            pair_no = idx - start  # 1, 2, 3 ... for the first pairs

            t0 = time.perf_counter()
            d_mask = frame_difference(prev_gray, gray)
            t_diff = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            old, new, residual = lucas_kanade(prev_gray, gray)
            t_lk = (time.perf_counter() - t0) * 1000

            area = frame.shape[0] * frame.shape[1]
            moving_pts = int(np.sum(residual > 1.0)) if len(residual) else 0
            stats["diff"].append((100 * np.count_nonzero(d_mask) / area, count_blobs(d_mask), t_diff))
            stats["gmm"].append((100 * np.count_nonzero(g_mask) / area, count_blobs(g_mask), t_gmm))
            stats["lk"].append((len(old), moving_pts,
                                float(np.mean(residual)) if len(residual) else 0.0, t_lk))

            flow_img = draw_flow(frame, old, new, residual)
            diff_view = overlay_mask(frame, d_mask, (0, 0, 255))
            gmm_view = overlay_mask(frame, g_mask, (255, 0, 0))

            if 1 <= pair_no <= N_PAIRS:
                save_img(out / "masks_diff" / f"diff_mask_{idx}.png", d_mask)
                save_img(out / "masks_gmm" / f"gmm_mask_{idx}.png", g_mask)
                save_img(out / "flow" / f"flow_{idx}.png", flow_img)
                still_rows.append([frame, d_mask, g_mask, flow_img, idx])

            # side-by-side demo video (2x2 grid)
            if idx - start <= DEMO_FRAMES:
                grid = np.vstack([
                    np.hstack([label(frame, f"Original  (frame {idx})"),
                               label(diff_view, "Frame differencing")]),
                    np.hstack([label(gmm_view, "GMM background subtraction (MOG2)"),
                               label(flow_img, "Lucas-Kanade optical flow (red = moving)")]),
                ])
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out / "motion_demo.mp4"), fourcc,
                                             min(info["fps"], 30), (grid.shape[1], grid.shape[0]))
                writer.write(grid)

        if start <= idx <= start + N_PAIRS:
            save_img(out / "frames" / f"frame_{idx}.png", frame)
        prev_gray = gray
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not still_rows:
        raise SystemExit("[!] Video too short to analyse - record at least a few seconds.")

    # ---- the comparison figure: one row per frame pair --------------------
    imgs, titles = [], []
    for frame, d_mask, g_mask, flow_img, i in still_rows:
        imgs += [frame, d_mask, g_mask, flow_img]
        titles += [f"Frame {i}", f"Frame differencing mask ({i - 1}->{i})",
                   f"GMM foreground mask ({i})", f"Lucas-Kanade vectors ({i - 1}->{i}, arrows x5)"]
    save_grid(imgs, titles, out / "motion_comparison_grid.png", cols=4, cell=(4.6, 3.2),
              suptitle="Consecutive frames - differencing vs GMM vs optical flow")

    # ---- numbers for the comparison table --------------------------------
    d = np.array(stats["diff"])
    g = np.array(stats["gmm"])
    lk = np.array(stats["lk"])
    rows = [
        ["Frame differencing", "Binary motion mask (changed pixels)",
         round(d[:, 0].mean(), 2), round(d[:, 1].mean(), 1), round(d[:, 2].mean(), 2),
         "Very fast, no model; only edges of the mover show up and it reacts to any camera shake"],
        ["GMM background subtraction (MOG2)", "Binary foreground mask (full object)",
         round(g[:, 0].mean(), 2), round(g[:, 1].mean(), 1), round(g[:, 2].mean(), 2),
         "Fills the whole object and ignores static background, but needs warm-up and a steady camera"],
        ["Lucas-Kanade optical flow", "Sparse motion vectors (direction + speed)",
         f"{lk[:, 1].mean():.1f} moving of {lk[:, 0].mean():.0f} tracked pts", "-",
         round(lk[:, 3].mean(), 2),
         "Only method that gives direction and speed, but only at corner points, not whole regions"],
    ]
    write_csv(out / "method_comparison.csv",
              ["Method", "Output", "Avg foreground area (% of frame) / moving points",
               "Avg blobs per frame", "Avg time per frame (ms)", "Strengths / weaknesses"], rows)
    write_json(out / "summary.json", {
        "video_resolution": f"{info['width']}x{info['height']}",
        "fps": round(info["fps"], 2),
        "total_frames": total,
        "duration_s": round(info["duration"], 1),
        "start_frame": start,
        "frames_analysed": len(stats["diff"]),
        "work_width": WORK_WIDTH,
        "mean_flow_residual_px": round(float(lk[:, 2].mean()), 3),
    })
    print("[task4] done ->", out)


if __name__ == "__main__":
    main()
