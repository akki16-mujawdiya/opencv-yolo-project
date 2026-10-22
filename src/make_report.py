"""
make_report.py - builds the main technical report (Word .docx) from the results
that the four task scripts saved in outputs/.

    python src/make_report.py

Before running: fill in config.json (name, PRN, Git URL, video link, Drive link).
The numbers in the text are read from the CSV/JSON result files, so the report
always matches what your code actually produced.
Open the .docx afterwards, read it once, and tweak any wording you like.
"""
import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from utils import (DATA_DIR, IMAGE_EXTS, OUT_DIR, ROOT, VIDEO_EXTS, find_input,
                   load_config, read_csv, read_json)

fig_counter = 0
table_counter = 0


# ----------------------------------------------------------------------------
# small docx helpers
# ----------------------------------------------------------------------------
def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def para(doc, text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def figure(doc, path, caption, width=6.2):
    """Insert an image with a numbered caption. Skips quietly if the file is missing."""
    global fig_counter
    path = Path(path)
    if not path.exists():
        para(doc, f"[image not found: {path.name} - run the task script first]", italic=True)
        return
    fig_counter += 1
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {fig_counter}: {caption}")
    r.italic, r.font.size = True, Pt(9.5)


def table(doc, header, rows, caption, widths=None, font=9):
    global table_counter
    table_counter += 1
    cap = doc.add_paragraph()
    r = cap.add_run(f"Table {table_counter}: {caption}")
    r.bold, r.font.size = True, Pt(10)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(h))
        run.bold, run.font.size = True, Pt(font)
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def code_block(doc, text):
    for line in text.splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(line if line else " ")
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        run.font.size = Pt(7)


def read_or_none(fn, path):
    try:
        return fn(path)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# text that depends on the results
# ----------------------------------------------------------------------------
def best_filter(rows, noise):
    sub = [r for r in rows if r["Noise type"] == noise and "Noisy" not in r["Image"]]
    if not sub:
        return None
    best_p = max(sub, key=lambda r: float(r["PSNR (dB)"]))
    best_s = max(sub, key=lambda r: float(r["SSIM"]))
    base = [r for r in rows if r["Noise type"] == noise and "Noisy" in r["Image"]][0]
    return best_p, best_s, base


# ----------------------------------------------------------------------------
def main():
    cfg = load_config()
    t1, t2 = OUT_DIR / "task1", OUT_DIR / "task2"
    t3, t4 = OUT_DIR / "task3", OUT_DIR / "task4"
    s1 = read_or_none(read_json, t1 / "summary.json") or {}
    s2 = read_or_none(read_json, t2 / "summary.json") or {}
    s3 = read_or_none(read_json, t3 / "summary.json") or {}
    s4 = read_or_none(read_json, t4 / "summary.json") or {}

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.9)
        sec.top_margin = sec.bottom_margin = Inches(0.9)

    # ---------------- cover -------------------------------------------------
    para(doc, "Computer Vision", size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "CCA 3 - Practical Coding Assignment", bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Units 3, 4 and 5: Restoration & Segmentation, Feature Extraction, "
              "Object Recognition & Motion Estimation", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    table(doc, ["Item", "Details"], [
        ["Student", cfg["student_name"]],
        ["PRN", cfg["prn"]],
        ["Course", cfg["course"]],
        ["Date", datetime.date.today().strftime("%d %B %Y")],
        ["Git repository", cfg["git_url"]],
        ["Project explanation video", cfg["explanation_video_url"]],
        ["Google Drive folder (originals + outputs)", cfg["drive_folder_url"]],
        ["Shared with", "rahul.inchal.inchal24@gmail.com, rahul.inchal.vf@mitwpu.edu.in"],
    ], "Submission details", widths=[2.3, 4.4], font=10)
    doc.add_page_break()

    # ---------------- 0. intro ------------------------------------------------
    doc.add_heading("Introduction", 1)
    para(doc, "This report documents the practical assignment for Units 3, 4 and 5. "
              "Using my own photograph and my own video I implemented image restoration, "
              "morphological processing and segmentation (Task 1), feature extraction and "
              "image matching (Task 2), object recognition with Haar cascades and YOLO (Task 3) "
              "and video motion estimation (Task 4). The code is written in Python with OpenCV, "
              "NumPy, Matplotlib, scikit-image and Ultralytics YOLO. All numbers and images in this "
              "report were produced by the code in the Git repository.")

    # ---------------- 1. input ------------------------------------------------
    doc.add_heading("1. Input", 1)
    photo = find_input("photo", IMAGE_EXTS)
    photo2 = find_input("photo2", IMAGE_EXTS)
    video = find_input("video", VIDEO_EXTS)
    objects = find_input("objects", IMAGE_EXTS)
    vid_txt = (f"{s4.get('video_resolution', '?')} at {s4.get('fps', '?')} fps, "
               f"{s4.get('total_frames', '?')} frames (about {s4.get('duration_s', '?')} s)")
    table(doc, ["Input", "File", "Details / how it was used"], [
        ["Original photograph", photo.name if photo else "missing",
         f"Captured with my own phone/camera, not edited. Used in Tasks 1, 2 and 3. "
         f"Working size {s1.get('image_size', '?')} after resizing a copy in memory."],
        ["Second photograph", photo2.name if photo2 else "missing",
         "Same object/scene from a different position. Used for SIFT matching in Task 2."],
        ["Object photograph (optional)", objects.name if objects else "not used",
         "Several touching objects, used for the watershed segmentation in Task 1."],
        ["Original video", video.name if video else "missing",
         f"Self-recorded, unedited. {vid_txt}. Used in Tasks 3 and 4."],
    ], "Input data used", widths=[1.6, 1.3, 3.8])
    figure(doc, t1 / "input_photo.png", "Original photograph used for Tasks 1-3 (resized copy).", 4.5)
    figure(doc, t2 / "input_photo2.png", "Second photograph of the same object/scene used for SIFT matching.", 3.6)
    if objects:
        figure(doc, t1 / "input_objects.png", "Image with multiple touching objects used for watershed.", 4.0)
    frames_dir = t4 / "frames"
    first_frames = sorted(frames_dir.glob("frame_*.png"))[:1] if frames_dir.exists() else []
    for f in first_frames:
        figure(doc, f, "Sample frame from the self-recorded video.", 4.2)
    para(doc, "Folder layout and how to place the input files is described in data/README.md "
              "in the repository. Large raw videos are kept on Google Drive instead of Git.")

    # ---------------- 2. source code -------------------------------------------
    doc.add_heading("2. Source Code", 1)
    para(doc, f"Repository: {cfg['git_url']}")
    table(doc, ["File", "What it does"], [
        ["src/utils.py", "Shared helpers: finding inputs, loading images, saving figures and CSV/JSON."],
        ["src/task1_restoration_segmentation.py", "Noise, median/Gaussian restoration, PSNR/SSIM, thresholding, morphology, Canny, watershed."],
        ["src/task2_features.py", "HOG visualisation, SIFT keypoints, ratio-test matching + RANSAC, rotation/resize experiments."],
        ["src/task3_detection.py", "Haar cascade face detection and YOLOv8 detection with boxes, labels and scores."],
        ["src/task4_motion.py", "Frame differencing, MOG2 background subtraction, Lucas-Kanade optical flow."],
        ["src/run_all.py / src/make_report.py", "Run all tasks / build this report."],
    ], "Source files", widths=[2.6, 4.1])
    para(doc, "The complete source listing is given in Appendix A.")

    # ---------------- 3. Task 1 ---------------------------------------------------
    doc.add_page_break()
    doc.add_heading("3. Task 1 - Image Restoration and Segmentation (Unit 3)", 1)
    doc.add_heading("3.1 Image restoration", 2)
    para(doc, "Explanation. I added two kinds of noise to the clean photograph: Gaussian noise "
              "(sigma = 25) which changes every pixel a little, and salt-and-pepper noise (5% of pixels "
              "set to pure black or white). Each noisy image was restored with a 3x3 median filter, a 5x5 median "
              "filter and a 5x5 Gaussian filter (sigma = 1.2). Quality was measured against the clean image "
              "with PSNR (higher is better) and SSIM (closer to 1 is better).")
    for tag, name in (("gaussian", "Gaussian"), ("saltpepper", "salt-and-pepper")):
        figure(doc, t1 / f"restoration_{tag}.png", f"Original, {name} noisy image and the three restored versions.")
        figure(doc, t1 / f"restoration_{tag}_zoom.png", f"Zoomed centre crop for {name} noise (easier to compare detail).")
    metrics = read_or_none(read_csv, t1 / "restoration_metrics.csv")
    if metrics:
        table(doc, ["Noise type", "Image", "PSNR (dB)", "SSIM"],
              [[r["Noise type"], r["Image"], r["PSNR (dB)"], r["SSIM"]] for r in metrics],
              "PSNR and SSIM compared with the clean photograph", widths=[2.0, 2.0, 1.3, 1.3])
        para(doc, "Comparison.", bold=True)
        for noise in ("Gaussian noise", "Salt & pepper noise"):
            res = best_filter(metrics, noise)
            if res:
                bp, bs, base = res
                bullet(doc, f"{noise}: the noisy image scored {base['PSNR (dB)']} dB / SSIM {base['SSIM']}. "
                            f"The best PSNR came from {bp['Image']} ({bp['PSNR (dB)']} dB) and the best SSIM "
                            f"from {bs['Image']} ({bs['SSIM']}).")
        para(doc, "The median filter is clearly the right tool for salt-and-pepper noise because the extreme "
                  "black/white pixels are outliers that the median simply ignores, while the Gaussian filter "
                  "averages them into their neighbours and leaves grey blotches. For Gaussian noise the two "
                  "filters are much closer: Gaussian smoothing matches the noise statistics and gives a "
                  "good PSNR, while the larger median kernel keeps edges a bit better (higher SSIM) at the "
                  "cost of a slightly 'painted' look. Bigger kernels remove more noise but blur fine detail.")

    doc.add_heading("3.2 Thresholding, morphology and edge detection", 2)
    para(doc, f"Explanation. The photo was converted to grayscale, smoothed with a 5x5 Gaussian and "
              f"binarised using a fixed threshold (127), Otsu's automatic threshold "
              f"(T = {s1.get('otsu_threshold', 0):.0f}) and an adaptive Gaussian threshold. "
              "Otsu's binary mask was then cleaned with erosion, dilation, opening and closing using a "
              "5x5 elliptical structuring element. Canny edge detection was run with fixed thresholds "
              f"(100/200) and with thresholds chosen from the image median "
              f"({s1.get('canny_auto_low', '?')}/{s1.get('canny_auto_high', '?')}).")
    figure(doc, t1 / "thresholding.png", "Binary images from global, Otsu and adaptive thresholding.")
    figure(doc, t1 / "morphology.png", "Erosion, dilation, opening and closing of the Otsu mask.")
    figure(doc, t1 / "canny_edges.png", "Canny edge maps with fixed and automatic thresholds.")
    para(doc, "Comparison.", bold=True)
    bullet(doc, "Otsu picks the threshold from the histogram itself, so it adapts to each photo better than "
                "a hand-picked 127; adaptive thresholding handles uneven lighting but also picks up texture/noise.")
    bullet(doc, "Erosion shrinks white regions and removes small specks; dilation grows them and fills small gaps. "
                "Opening removes isolated noise without shrinking big objects much; closing fills small holes inside objects.")
    bullet(doc, f"Canny with fixed thresholds marked {s1.get('edge_pixels_fixed_pct', '?')}% of pixels as edges, the "
                f"automatic version {s1.get('edge_pixels_auto_pct', '?')}%. The automatic version needs no tuning "
                "when the lighting changes.")

    doc.add_heading("3.3 Watershed segmentation", 2)
    para(doc, f"Explanation. Watershed was applied to '{s1.get('watershed_source', '?')}', which contains "
              "several objects that sit close to or touch each other. Steps: Otsu threshold, opening to remove noise, "
              "distance transform, threshold of the distance map to get one marker per object, then "
              "cv2.watershed floods from these markers and draws boundary lines where the regions meet. "
              f"The algorithm found {s1.get('watershed_segments', '?')} segments.")
    figure(doc, t1 / "watershed_steps.png", "Watershed pipeline: mask, distance transform, markers, labels and result.")
    figure(doc, t1 / "watershed_result.png", "Final watershed boundaries (red) on the input image.", 4.6)
    para(doc, "Comparison. Plain thresholding would merge touching objects into one blob. The distance "
              "transform turns each blob into a 'hill' and the markers are placed on the hill tops, so "
              "touching objects get separated. Watershed is sensitive to the marker threshold: too low "
              "merges objects, too high splits one object into several (over-segmentation).")

    # ---------------- 4. Task 2 -------------------------------------------------
    doc.add_page_break()
    doc.add_heading("4. Task 2 - Feature Extraction and Image Matching (Unit 4)", 1)
    doc.add_heading("4.1 HOG features", 2)
    para(doc, f"Explanation. Both photos were converted to grayscale and resized to 256x256. HOG divides the "
              "image into 16x16 pixel cells, builds a 9-bin histogram of gradient orientations in each "
              "cell and normalises over 2x2 cell blocks. The result is a descriptor vector of length "
              f"{s2.get('hog_vector_length', '?')}. The visualisation shows the dominant edge direction in each cell. "
              f"The cosine similarity between the HOG vectors of my two photos is {s2.get('hog_cosine_similarity', '?')} "
              "(different viewpoints, so HOG is not expected to be identical).")
    figure(doc, t2 / "hog_visualisation.png", "HOG visualisation for both photographs.")

    doc.add_heading("4.2 SIFT keypoints", 2)
    para(doc, f"Explanation. SIFT finds scale-invariant keypoints and describes each with a 128-number descriptor. "
              f"It found {s2.get('keypoints_photo1', '?')} keypoints in photo 1 and {s2.get('keypoints_photo2', '?')} in photo 2.")
    figure(doc, t2 / "sift_keypoints.png", "SIFT keypoints on both photographs (circle size = scale).")

    doc.add_heading("4.3 Matching, rotation and resizing", 2)
    para(doc, "Explanation. Descriptors were matched with a brute-force matcher and Lowe's ratio test "
              f"(ratio {s2.get('ratio_test', 0.75)}). RANSAC then fits a homography; matches that agree with it are "
              "'inliers' (green) and the rest are outliers (red). The experiment was repeated after rotating "
              "photo 2 by 45 degrees, resizing it to 50%, and doing both together.")
    figure(doc, t2 / "sift_matching_comparison.png", "SIFT matches for the original pair and the rotated/resized versions.")
    mrows = read_or_none(read_csv, t2 / "matching_results.csv")
    if mrows:
        table(doc, list(mrows[0].keys()), [list(r.values()) for r in mrows],
              "Matching quality for each experiment", font=8)
        best = max(mrows, key=lambda r: float(r["Inlier ratio"].strip("%")))
        worst = min(mrows, key=lambda r: int(r["RANSAC inliers"]))
        para(doc, "Comparison.", bold=True)
        para(doc, f"The highest inlier ratio was in '{best['Experiment']}' ({best['Inlier ratio']}) and the fewest "
                  f"inliers were found in '{worst['Experiment']}' ({worst['RANSAC inliers']}). SIFT is built to "
                  "be invariant to scale and rotation, so matching still works after rotating or resizing the image, "
                  "although the number of matches drops when the image is shrunk (fewer pixels means fewer detectable "
                  "keypoints). Rotation adds black borders that can create a few extra, unmatched keypoints.")

    # ---------------- 5. Task 3 -------------------------------------------------
    doc.add_page_break()
    doc.add_heading("5. Task 3 - Object Recognition with Viola-Jones and YOLO (Unit 5)", 1)
    para(doc, "Explanation. Viola-Jones uses Haar-like features and a cascade of classifiers; I used OpenCV's "
              "pretrained frontal-face model (scaleFactor 1.1, minNeighbors 5). It gives no true confidence, so "
              "the 'weight' from detectMultiScale3 is shown instead. YOLO "
              f"({s3.get('yolo_model', 'yolov8n.pt')}) is a pretrained deep network that detects 80 COCO classes "
              f"in a single pass; the confidence threshold was {s3.get('yolo_conf_threshold', 0.25)}. "
              f"Both were run on the photo and {max(int(s3.get('images_processed', 1)) - 1, 0)} frames from the video.")
    figure(doc, t3 / "overview_all.png", "Haar cascade (left of each pair) vs YOLO (right) on all test images.", 6.6)
    crows = read_or_none(read_csv, t3 / "detector_comparison.csv")
    if crows:
        table(doc, list(crows[0].keys()), [list(r.values()) for r in crows],
              "Detector comparison per image", font=8)
        para(doc, "Detailed side-by-side images:", bold=True)
        for r in crows[:3]:
            safe = r["Image"].replace("(", "").replace(")", "").replace("#", "")
            figure(doc, t3 / f"{safe}_comparison.png", f"Haar vs YOLO - {r['Image']}.", 6.2)
        para(doc, "Comparison and failure cases.", bold=True)
        disagreements = [r for r in crows if "agree" not in r["Observation"]]
        if disagreements:
            for r in disagreements:
                bullet(doc, f"{r['Image']}: {r['Observation']}.")
        else:
            para(doc, "Both detectors agreed on every image I tested.")
        h_total = sum(int(r["Haar faces"]) for r in crows)
        y_total = sum(int(r["YOLO objects"]) for r in crows)
        h_ms = sum(float(r["Haar time (ms)"]) for r in crows) / len(crows)
        y_ms = sum(float(r["YOLO time (ms)"]) for r in crows) / len(crows)
        para(doc, f"Overall the Haar cascade reported {h_total} faces and YOLO reported {y_total} objects across "
                  f"{len(crows)} images; average inference time was {h_ms:.0f} ms (Haar) and {y_ms:.0f} ms (YOLO) on my computer. "
                  "Haar is light and simple but only finds frontal faces, so side views, small faces and "
                  "partly hidden faces are missed and textured background patches can give false detections. "
                  "YOLO detects many classes, copes with different poses and sizes, and gives a real "
                  "confidence score, but it needs a larger model and more computation.")

    # ---------------- 6. Task 4 -------------------------------------------------
    doc.add_page_break()
    doc.add_heading("6. Task 4 - Video Motion Estimation (Unit 5)", 1)
    para(doc, f"Explanation. {s4.get('frames_analysed', '?')} consecutive frames starting at frame "
              f"{s4.get('start_frame', '?')} of my video were analysed after resizing to {s4.get('work_width', 640)} px "
              "width. (1) Frame differencing: absolute difference of two consecutive gray frames, blurred, "
              "thresholded at 25 and cleaned with morphology. (2) GMM background subtraction: OpenCV's MOG2 models "
              "each pixel as a mixture of Gaussians learned over earlier frames (the model was warmed up on the "
              "frames before the start frame); shadows were removed from the mask. (3) Lucas-Kanade optical flow: "
              "Shi-Tomasi corners are tracked from one frame to the next; the median motion of all points is taken "
              "as camera shake and subtracted, so points that still move (red arrows) belong to the moving object.")
    figure(doc, t4 / "motion_comparison_grid.png", "Consecutive frames with differencing masks, GMM masks and optical-flow vectors.", 6.6)
    para(doc, "Selected foreground masks and motion vectors are also saved individually in "
              "outputs/task4/masks_diff, masks_gmm and flow. A side-by-side demo video (motion_demo.mp4) is "
              "saved in outputs/task4 and uploaded to Google Drive.")
    mrows = read_or_none(read_csv, t4 / "method_comparison.csv")
    if mrows:
        table(doc, list(mrows[0].keys()), [list(r.values()) for r in mrows],
              "Comparison of frame differencing, GMM background subtraction and Lucas-Kanade optical flow",
              font=8)
    para(doc, "Comparison.", bold=True)
    bullet(doc, "Frame differencing is the fastest and needs no training, but between consecutive frames it mostly "
                "highlights only the edges of the moving object and breaks up into several small blobs; any "
                "camera shake also shows up as 'motion'.")
    bullet(doc, "GMM background subtraction gives a more complete silhouette of the mover as a single region "
                "because it compares against a learned background, but it needs warm-up frames, costs more time "
                "per frame and struggles if the camera moves.")
    bullet(doc, "Lucas-Kanade optical flow is the only one of the three that tells direction and speed of the "
                "motion, but it is sparse (only corner points) and can lose points on plain, textureless areas.")
    bullet(doc, "Best choice depends on the goal: GMM to locate the object, differencing for a quick motion "
                "trigger, optical flow to measure how it moves.")

    # ---------------- 7. conclusion -------------------------------------------------
    doc.add_heading("7. Conclusion", 1)
    para(doc, "In this assignment I implemented restoration, segmentation, feature extraction, object recognition "
              "and motion estimation on my own photographs and video. The main results were:")
    if metrics:
        bp = best_filter(metrics, "Salt & pepper noise")
        bg = best_filter(metrics, "Gaussian noise")
        if bp and bg:
            bullet(doc, f"Restoration: {bp[0]['Image']} worked best for salt-and-pepper noise "
                        f"({bp[0]['PSNR (dB)']} dB), and {bg[0]['Image']} gave the best PSNR for Gaussian noise "
                        f"({bg[0]['PSNR (dB)']} dB).")
    bullet(doc, "Segmentation: Otsu thresholding + morphological opening gave clean masks, Canny gave thin edges, "
                "and watershed with distance-transform markers separated touching objects.")
    bullet(doc, "Feature matching: SIFT kept matching correctly after rotation and resizing, confirming its "
                "scale and rotation invariance.")
    bullet(doc, "Detection: YOLO is more general and robust than the Haar cascade, while Haar is quicker and "
                "lighter but limited to frontal faces.")
    bullet(doc, "Motion: differencing, GMM and optical flow each answer a different question (where did "
                "pixels change, what is foreground, how is it moving), so they are best used together.")
    para(doc, "All source code, example outputs and the README are in the Git repository: " + cfg["git_url"])
    para(doc, "Project explanation video: " + cfg["explanation_video_url"])

    # ---------------- appendix -------------------------------------------------
    doc.add_page_break()
    doc.add_heading("Appendix A - Source code listing", 1)
    for name in ("utils.py", "task1_restoration_segmentation.py", "task2_features.py",
                 "task3_detection.py", "task4_motion.py", "run_all.py", "make_report.py"):
        doc.add_heading(f"src/{name}", 2)
        code_block(doc, (ROOT / "src" / name).read_text(encoding="utf-8"))

    out_path = ROOT / "outputs" / "Technical_Report.docx"
    doc.save(out_path)
    print(f"[report] saved {out_path}")
    for key in ("git_url", "explanation_video_url", "drive_folder_url", "prn"):
        if not cfg.get(key) or "PASTE" in cfg[key] or "YOUR" in cfg[key]:
            print(f"[report] reminder: fill in '{key}' in config.json and run again")


if __name__ == "__main__":
    main()
