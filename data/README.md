# Input data - how to organise it

Put your own files in this folder using exactly these names
(images can be .jpg / .jpeg / .png, video can be .mp4 / .mov / .avi / .mkv):

| File name    | Required? | What it is                                                                                       |
|--------------|-----------|--------------------------------------------------------------------------------------------------|
| `photo.*`    | yes       | Your original, unedited photograph (used in Tasks 1, 2 and 3)                                    |
| `photo2.*`   | yes       | A second photo of the SAME object/scene from a slightly different angle or distance (Task 2)     |
| `video.*`    | yes       | Your original 15-30 second video with a moving person/object (Tasks 3 and 4)                     |
| `objects.*`  | optional  | A photo of several small, touching objects (coins, nuts, pens...) for the watershed step. If it is missing, `photo.*` is used instead. |

Rules

* Do **not** edit, crop or filter the originals. The scripts only resize copies in memory.
* iPhone `.HEIC` photos: convert to `.jpg` first (OpenCV cannot read HEIC).
* iPhone `.MOV` (HEVC) sometimes does not open in OpenCV. If that happens keep the original file
  safe and make a converted copy for the code:
  `ffmpeg -i original.MOV -c:v libx264 -crf 18 data/video.mp4`
* Large videos are NOT committed to Git (see `.gitignore`). Upload the video to Google Drive and
  paste the link into `config.json` and into the report.

Where the original files live: Google Drive folder (link is in the main technical report).
