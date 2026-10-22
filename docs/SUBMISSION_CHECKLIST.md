# Submission checklist

## Before running
- [ ] `data/photo.*`, `data/photo2.*`, `data/video.*` (and optional `data/objects.*`) are your own, unedited files
- [ ] `config.json` has your name, PRN (links can be filled in later)
- [ ] `pip install -r requirements.txt` finished without errors

## Run
- [ ] `python src/run_all.py` finished with no task errors
- [ ] Looked through `outputs/` - all four task folders have images
- [ ] Task 3: opened `outputs/task3/detector_comparison.csv` and checked the "Observation" column;
      found at least ONE detector miss or wrong detection (if none, test a harder frame:
      side view, small or partly hidden person) and mention it in the report
- [ ] `outputs/task4/motion_demo.mp4` plays correctly

## Git
- [ ] Create a repository on GitHub (private is fine)
- [ ] `git init`, `git add .`, `git commit -m "CCA 3 assignment"`, `git remote add origin <url>`, `git push -u origin main`
- [ ] Repo contains: source files, README, requirements.txt, example outputs, data/README.md
- [ ] Big videos are NOT in the repo (`.gitignore` already blocks them)
- [ ] Repo shared (Settings -> Collaborators, or make it accessible) with:
  - rahul.inchal.inchal24@gmail.com
  - rahul.inchal.vf@mitwpu.edu.in

## Google Drive
- [ ] Upload: original photo(s), original video, explanation video, `Technical_Report.docx` (+ PDF), outputs folder
- [ ] Share the folder/files with both email addresses above (and use the Drive location your instructor gave)

## Explanation video (5-8 minutes)
- [ ] Follow `docs/VIDEO_SCRIPT.md`, your own voice, shows code + results
- [ ] Uploaded to the instructor's Drive location
- [ ] Link pasted into `config.json`

## Report
- [ ] `config.json` filled completely (Git URL, video link, Drive link)
- [ ] `python src/make_report.py` re-run
- [ ] Opened `outputs/Technical_Report.docx`, read it, added your own observations in your own words
- [ ] Git URL and video link are visible on the cover page
- [ ] Optional: export to PDF as well

## Final check
- [ ] Fresh clone of the repo + `pip install -r requirements.txt` + `python src/run_all.py` works
- [ ] Both email addresses can open the repo, Drive files and the video
