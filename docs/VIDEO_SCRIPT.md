# Project explanation video - talking points (5 to 8 minutes)

Speak in your own words and keep your face/voice in the video (slides-only or silent
screen recordings are not accepted). Rough timing is given for each part.
Tip: record your screen with a voice-over (OBS, Zoom or Google Meet recording), run the code live
or show the saved outputs, and keep this page open as a checklist.

## 1. Introduction (0:00 - 0:40)
* Say your name and that this is CCA 3 covering Units 3, 4 and 5.
* Objective: restoration & segmentation, feature extraction, object recognition, motion estimation
  using Python, OpenCV, scikit-image and YOLO.
* Everything uses my own photo and my own video.

## 2. Input data (0:40 - 1:20)
* Show `data/photo.jpg`: where / what you shot, why it has good detail.
* Show the second photo (same object, different angle).
* Play a few seconds of the video: what moves, how long it is, resolution/fps, was the camera steady?
* Mention the originals are unedited and the large video is on Google Drive.

## 3. Task 1 - Restoration & segmentation (1:20 - 2:50)
* Open `src/task1_restoration_segmentation.py`: show `add_gaussian_noise`, `add_salt_pepper`, the FILTERS dictionary.
* Show `outputs/task1/restoration_*` and the PSNR/SSIM table.
  Say which filter won for each noise and why (median = outliers, Gaussian = smooth noise).
* Show thresholding, morphology (erosion/dilation/opening/closing), Canny.
* Explain watershed: distance transform -> markers -> flooding; show `watershed_steps.png`.

## 4. Task 2 - Feature extraction (2:50 - 4:00)
* HOG: gradient orientations in cells, show visualisation.
* SIFT: keypoints (scale + orientation), 128-d descriptors.
* Matching: ratio test + RANSAC (green inliers / red outliers).
* Show the table: original vs rotated vs resized - what happened to the match count and why.

## 5. Task 3 - Haar vs YOLO (4:00 - 5:10)
* Haar cascade = Viola-Jones, frontal faces only, no real confidence.
* YOLO = deep network, many classes, confidence score.
* Show `overview_all.png`; point at one miss or wrong detection (see `detector_comparison.csv`
  "Observation" column) and explain why it happened.

## 6. Task 4 - Motion estimation (5:10 - 6:30)
* Play `outputs/task4/motion_demo.mp4`.
* Frame differencing (only edges, noisy), GMM/MOG2 (full silhouette, needs warm-up),
  Lucas-Kanade (direction + speed, sparse).
* Show the comparison table.

## 7. Wrap-up (6:30 - 7:30)
* Which technique worked best for your images and why.
* Mention the Git repo with README + requirements, and the report.
* Thank the instructor.

Upload to the Google Drive location provided, then paste the link in `config.json`,
the README and the main technical report.
