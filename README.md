# startrack — track and label stars across night-sky videos & photos

## Install
```bash
pip install numpy opencv-python scipy
# optional extras
pip install requests     # star names via astrometry.net
pip install Pillow       # read photo capture times (EXIF)
pip install rawpy        # RAW files (CR2, NEF, ARW, DNG…)
pip install astroalign   # extra-robust matching across long gaps between clips
# ffmpeg on PATH -> output is H.264 (plays on phones); otherwise MPEG-4
```

## Run
```bash
# one or more video clips, in time order
python startrack.py clip1.mp4 clip2.mp4 -o out/

# a folder of photos (ordered by name; use --sort time to order by EXIF time)
python startrack.py night_photos/ -o out/ --out-fps 8

# long 4K video: use every 5th frame at half resolution
python startrack.py timelapse.mp4 --step 5 --scale 0.5

# hide trees/buildings: white = sky, black = ignore (same size as the frames)
python startrack.py clip.mp4 --mask sky_mask.png

# real star names (free API key from nova.astrometry.net -> My Profile)
python startrack.py clip.mp4 --astrometry-key YOUR_KEY --fov 60
```

## Outputs (in `-o` folder)
| file | contents |
|---|---|
| `annotated.mp4` | every frame with circles, trails and labels (`S001`, or `Vega (S003)` when named) |
| `star_map.png` | one frame with each star's full path drawn |
| `tracks.csv` | one row per star per frame: frame, clip, source file, time, label, name, x, y, flux |
| `stars.json` | per-star summary (frames seen, brightness, stable reference position) + per-frame motion log |

Labels are ranked by brightness: `S001` is the brightest star tracked.

## How it works
1. **Detect**: removes the background (sky glow, light pollution) and applies a difference-of-Gaussians point-source filter. The threshold comes from the height of noise peaks, so it holds up on blocky, compressed phone video. Hot pixels are median-filtered out; streaks from satellites, planes and meteors are rejected by shape.
2. **Sky motion**: matches the brightest stars between frames and fits rotation + shift with RANSAC. Earth's rotation makes the whole sky turn about the celestial pole, so every star obeys one transform. A large jump (a gap between clips, a bumped tripod) falls back to phase correlation, and then to astroalign triangle matching if it's installed.
3. **Track**: predicts each star's next position from that motion and links detections nearest-first. A star can vanish for up to `--max-missed` frames (clouds, a branch) and keep its ID. Things that don't move with the sky (streetlights, hot pixels) are discarded.
4. **Name (optional)**: plate-solves the frame with the most stars on astrometry.net and copies its named stars (`bright`, then `HD` catalogue) onto the matching tracks.

## Tuning
- Missing faint stars → `--peak-k 4` (more false positives); too much junk → `--peak-k 8`
- Fast motion between frames (big `--step`, wide lens, long gaps between photos) → `--search-radius 60`
- Stars losing their ID → raise `--match-radius` (e.g. 6) and/or `--max-missed`
- Cluttered labels → `--label-top 20`; long trails → `--trail 100`

## Limits
- Assumes the camera doesn't zoom (use `--allow-zoom` if it does), and that the lens isn't extreme fisheye: distortion is absorbed frame-to-frame, but very wide fisheye edges may fragment tracks.
- With a very long gap between clips and no astroalign installed, the sky may have rotated too much to re-link IDs. Install astroalign, or let new IDs start there.
- Naming needs internet access and a sky region astrometry.net can solve (roughly 20+ stars visible, not too much blur).

## Testing on synthetic data
```bash
python tests/make_synthetic.py syn            # fake sky: 2 clips + a photo folder, with ground truth
python startrack.py syn/clip1.mp4 syn/clip2.mp4 -o out_video
python tests/evaluate.py out_video/tracks.csv syn/gt.npz
```
