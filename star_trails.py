#!/usr/bin/env python3
"""
star_trails.py - Histogram-based star trail stacker + timelapse generator.

For each frame:
  1. Build a luminance histogram, find the sky-background peak and its width.
  2. Threshold = sky_peak + k * sky_noise  -> pixels above are "stars".
  3. Clean the mask (remove lone hot pixels, feather edges).
  4. Accumulate star highlights into a trail layer with a "lighten" (max) blend.
  5. Composite: current frame lightened with the accumulated trails -> timelapse frame.

Outputs:
  out/frames/trail_00001.jpg ...  (cumulative timelapse frames, full res)
  out/star_trails_final.jpg       (final trail image, full res)
  out/star_trails_timelapse.mp4   (if ffmpeg is installed)

Usage:
  python star_trails.py /path/to/shots -o out
  python star_trails.py shots -o out --k 5 --fps 30 --no-video
  python star_trails.py shots -o out --debug   # saves star masks + histogram stats

Requires: numpy, opencv-python   (optional: rawpy for RAW files, ffmpeg for video)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}


# ----------------------------------------------------------------- loading
def load_image(path: Path):
    """Return (float32 BGR image in [0, 1], histogram bins matching bit depth)."""
    ext = path.suffix.lower()
    if ext in RAW_EXT:
        try:
            import rawpy
        except ImportError:
            sys.exit("RAW files found: pip install rawpy  (or convert to TIFF first)")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 65535.0, 4096

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    maxval = 65535.0 if img.dtype == np.uint16 else 255.0
    bins = 4096 if img.dtype == np.uint16 else 256
    return img.astype(np.float32) / maxval, bins


def luminance(img: np.ndarray) -> np.ndarray:
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


# --------------------------------------------------------- histogram logic
def star_threshold(lum: np.ndarray, k: float, bins: int = 4096):
    """
    Find the sky background peak in the histogram and its noise width.
    threshold = peak + k * sigma
    sigma is estimated from the half-width-at-half-maximum on the bright side
    of the peak (HWHM / 1.1774 = sigma for a Gaussian). The bright side is used
    because the dark side is often clipped at 0 or polluted by foreground.
    """
    hist, edges = np.histogram(lum, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth histogram so a noisy bin doesn't fake the peak
    w = 3 if bins <= 256 else 9
    smooth = np.convolve(hist, np.ones(w) / w, mode="same")

    # Ignore the very first bins (pure-black foreground / clipped shadows)
    start = max(2, bins // 256)
    peak_idx = start + int(np.argmax(smooth[start:]))
    peak_val = smooth[peak_idx]
    sky = centers[peak_idx]

    # Walk right until the histogram drops below half max
    half = peak_val / 2
    j = peak_idx
    while j < bins - 1 and smooth[j] > half:
        j += 1
    hwhm = max(centers[j] - sky, 1.5 / bins)  # never narrower than quantization
    sigma = hwhm / 1.1774

    thr = sky + k * sigma
    thr = float(np.clip(thr, sky + 2.0 / bins, 0.98))
    return thr, float(sky), float(sigma)


def star_mask(lum: np.ndarray, thr: float, min_area: int, feather: float) -> np.ndarray:
    """Binary stars -> remove tiny specks -> soft feathered float mask in [0,1]."""
    binary = (lum > thr).astype(np.uint8)

    if min_area > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False  # background
        binary = keep[labels].astype(np.uint8)

    # Grow slightly so star halos are included, then feather
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    mask = binary.astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        mask = np.clip(mask * 1.5, 0, 1)  # keep cores solid, soften only edges
    return mask


# ------------------------------------------------------------------ output
def save_jpeg(path: Path, img: np.ndarray, quality: int):
    out = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):  # 4:4:4, no chroma loss
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    cv2.imwrite(str(path), out, params)


def make_video(frames_dir: Path, out_path: Path, fps: int):
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found - skipping video (frames are in", frames_dir, ")")
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frames_dir / "trail_%05d.jpg"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    print("Video:", out_path)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Histogram-based star trail timelapse")
    ap.add_argument("input", type=Path, help="folder of shots (sorted by filename)")
    ap.add_argument("-o", "--output", type=Path, default=Path("star_trails_out"))
    ap.add_argument("--k", type=float, default=6.0,
                    help="sigmas above sky peak to count as star (lower = more faint stars)")
    ap.add_argument("--min-area", type=int, default=2,
                    help="drop star blobs smaller than this many pixels (hot pixels)")
    ap.add_argument("--feather", type=float, default=1.0, help="mask edge blur radius (px)")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--debug", action="store_true", help="save star masks + print stats")
    args = ap.parse_args()

    files = sorted(p for p in args.input.iterdir()
                   if p.suffix.lower() in IMG_EXT | RAW_EXT)
    if not files:
        sys.exit(f"No images found in {args.input}")

    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (args.output / "masks").mkdir(exist_ok=True)

    trail = None
    shape = None
    n_out = 0  # frames actually written (keeps numbering gap-free for ffmpeg)
    print(f"{len(files)} frames found")

    for i, f in enumerate(files, 1):
        img, bins = load_image(f)
        if shape is None:
            shape = img.shape
            trail = np.zeros(shape, np.float32)
        elif img.shape != shape:
            print(f"  skip {f.name}: size {img.shape[:2]} != {shape[:2]}")
            continue

        lum = luminance(img)
        thr, sky, sigma = star_threshold(lum, args.k, bins)
        mask = star_mask(lum, thr, args.min_area, args.feather)

        # Accumulate only the star highlights (lighten blend)
        np.maximum(trail, img * mask[..., None], out=trail)

        # Timelapse frame = current real frame, lightened with all trails so far
        composite = np.maximum(img, trail)
        n_out += 1
        save_jpeg(frames_dir / f"trail_{n_out:05d}.jpg", composite, args.quality)

        if args.debug:
            cv2.imwrite(str(args.output / "masks" / f"mask_{n_out:05d}.png"),
                        (mask * 255).astype(np.uint8))
        print(f"  [{i}/{len(files)}] {f.name}  sky={sky:.4f}  sigma={sigma:.4f}  "
              f"thr={thr:.4f}  star_px={(mask > 0.5).mean() * 100:.3f}%")

    if n_out == 0:
        sys.exit("No frames were processed")
    final = args.output / "star_trails_final.jpg"
    shutil.copyfile(frames_dir / f"trail_{n_out:05d}.jpg", final)
    print("Final image:", final)

    if not args.no_video:
        make_video(frames_dir, args.output / "star_trails_timelapse.mp4", args.fps)


if __name__ == "__main__":
    main()
