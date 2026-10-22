"""
Task 3 (Unit 5) - Object recognition with Viola-Jones (Haar cascade) and YOLO

* Haar cascade: pretrained frontal-face model that ships with OpenCV
* YOLO: pretrained YOLOv8-nano from Ultralytics (trained on COCO, 80 classes)

The detectors are run on the photograph and on a few frames taken from the video.
Boxes, labels and scores are drawn, then the two approaches are compared.

Run:  python src/task3_detection.py
"""
import time

import cv2
import numpy as np

from utils import (IMAGE_EXTS, VIDEO_EXTS, load_image, need_input, open_video,
                   read_frame_at, resize_to_width, save_grid, save_img, task_dir,
                   write_csv, write_json)

N_VIDEO_FRAMES = 5          # how many frames to take from the video
YOLO_WEIGHTS = "yolov8n.pt"  # downloaded automatically the first time
YOLO_CONF = 0.25


# ----------------------------------------------------------------------------
# Haar (Viola-Jones)
# ----------------------------------------------------------------------------
def load_haar():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise SystemExit("[!] Could not load the Haar cascade file from OpenCV.")
    return cascade


def run_haar(cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    t0 = time.perf_counter()
    rects, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), outputRejectLevels=True)
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for (x, y, w, h), wt in zip(rects, weights):
        dets.append({"label": "face", "conf": float(wt), "box": (int(x), int(y), int(x + w), int(y + h))})
    return dets, ms


# ----------------------------------------------------------------------------
# YOLO
# ----------------------------------------------------------------------------
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_WEIGHTS)
    except Exception as e:  # no internet, package missing, ...
        raise SystemExit(
            f"[!] Could not load YOLO ({e}).\n"
            "    Check 'pip install ultralytics' and that you are online for the first run\n"
            f"    (it downloads {YOLO_WEIGHTS} automatically).")


def run_yolo(model, img):
    t0 = time.perf_counter()
    res = model.predict(img, conf=YOLO_CONF, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        dets.append({"label": res.names[int(b.cls[0])], "conf": float(b.conf[0]),
                     "box": (x1, y1, x2, y2)})
    return dets, ms


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def draw(img, dets, colour):
    out = img.copy()
    thick = max(2, img.shape[1] // 400)
    scale = max(0.5, img.shape[1] / 1400)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        text = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + 6), colour, -1)
        cv2.putText(out, text, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def describe_case(name, haar, yolo):
    """Spot places where the two detectors disagree - these are the interesting cases."""
    faces = len(haar)
    persons = len([d for d in yolo if d["label"] == "person"])
    notes = []
    if faces > 0 and persons == 0:
        notes.append("Haar found a face but YOLO found no person (possible Haar false positive "
                     "or YOLO miss)")
    if persons > 0 and faces == 0:
        notes.append("YOLO found a person but Haar found no face (face not frontal, too small, "
                     "or partly hidden)")
    if faces > persons and persons > 0:
        notes.append("Haar reported more faces than YOLO reported people (check for a false positive)")
    if faces == 0 and persons == 0 and not yolo:
        notes.append("Neither detector found anything")
    return "; ".join(notes) if notes else "Both detectors roughly agree"


# ----------------------------------------------------------------------------
def collect_inputs(out):
    """Photo + a few evenly spaced video frames, all saved to disk."""
    items = []
    photo = resize_to_width(load_image(need_input("photo", IMAGE_EXTS, "your photograph"), 1280), 960)
    items.append(("photo", photo))

    cap, info = open_video(need_input("video", VIDEO_EXTS, "your video"))
    frames_dir = out / "input_frames"
    frames_dir.mkdir(exist_ok=True)
    picks = np.linspace(0.1, 0.9, N_VIDEO_FRAMES) * max(info["frames"] - 1, 1)
    for k, idx in enumerate(picks, start=1):
        frame = read_frame_at(cap, int(idx))
        if frame is None:
            continue
        frame = resize_to_width(frame, 960)
        items.append((f"video_frame_{k}_(#{int(idx)})", frame))
        save_img(frames_dir / f"frame_{k}_{int(idx)}.png", frame)
    cap.release()
    return items, info


def main():
    out = task_dir("task3")
    items, info = collect_inputs(out)
    haar = load_haar()
    yolo = load_yolo()
    print(f"[task3] {len(items)} images to process")

    det_rows, cmp_rows = [], []
    for name, img in items:
        h_dets, h_ms = run_haar(haar, img)
        y_dets, y_ms = run_yolo(yolo, img)
        safe = name.replace("(", "").replace(")", "").replace("#", "")
        h_img = draw(img, h_dets, (0, 160, 0))
        y_img = draw(img, y_dets, (200, 80, 0))
        save_img(out / f"{safe}_haar.png", h_img)
        save_img(out / f"{safe}_yolo.png", y_img)
        save_grid([h_img, y_img],
                  [f"Haar cascade - {len(h_dets)} face(s) - {h_ms:.0f} ms",
                   f"YOLOv8n - {len(y_dets)} object(s) - {y_ms:.0f} ms"],
                  out / f"{safe}_comparison.png", cols=2, cell=(8, 5.5), suptitle=name)

        for d in h_dets:
            det_rows.append([name, "Haar", d["label"], round(d["conf"], 3), *d["box"]])
        for d in y_dets:
            det_rows.append([name, "YOLO", d["label"], round(d["conf"], 3), *d["box"]])

        classes = {}
        for d in y_dets:
            classes[d["label"]] = classes.get(d["label"], 0) + 1
        class_text = ", ".join(f"{k} x{v}" for k, v in sorted(classes.items())) or "-"
        note = describe_case(name, h_dets, y_dets)
        cmp_rows.append([name, len(h_dets), round(h_ms), len(y_dets), class_text, round(y_ms), note])
        print(f"   {name}: haar={len(h_dets)}  yolo={len(y_dets)} ({class_text})")

    write_csv(out / "all_detections.csv",
              ["Image", "Detector", "Label", "Confidence / weight", "x1", "y1", "x2", "y2"], det_rows)
    write_csv(out / "detector_comparison.csv",
              ["Image", "Haar faces", "Haar time (ms)", "YOLO objects", "YOLO classes",
               "YOLO time (ms)", "Observation"], cmp_rows)

    # one overview figure with every image for the report
    overview, titles = [], []
    for name, _ in items:
        safe = name.replace("(", "").replace(")", "").replace("#", "")
        for tag, label in (("haar", "Haar"), ("yolo", "YOLO")):
            overview.append(cv2.imread(str(out / f"{safe}_{tag}.png")))
            titles.append(f"{label} - {name}")
    save_grid(overview, titles, out / "overview_all.png", cols=4, cell=(5, 3.6),
              suptitle="Haar cascade vs YOLO on the photo and video frames")

    write_json(out / "summary.json", {
        "yolo_model": YOLO_WEIGHTS,
        "yolo_conf_threshold": YOLO_CONF,
        "images_processed": len(items),
        "video_frames_total": info["frames"],
        "video_fps": round(info["fps"], 2),
    })
    print("[task3] done ->", out)


if __name__ == "__main__":
    main()
