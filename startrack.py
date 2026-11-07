#!/usr/bin/env python3
"""
startrack.py - detect, track and label individual stars across night-sky
video clips and/or photo sequences.

Pipeline
--------
1. Read frames from any mix of videos, images, or folders (in time order).
2. Detect stars in every frame (background removal, noise-relative threshold,
   flux-weighted centroids; hot pixels, streaks and blobs rejected).
3. Estimate how the whole sky moved between frames (rotation + shift, i.e.
   Earth's rotation and any camera bump) with RANSAC on matched stars.
   Falls back to phase correlation, and to astroalign if installed, when the
   motion is large (e.g. a gap between two clips).
4. Predict every star's next position with that motion and assign detections
   to tracks, so each star keeps one ID even while it is briefly hidden
   (trees, clouds, a missed frame).
5. Label stars S001, S002, ... by brightness. Optionally plate-solve one frame
   with astrometry.net to attach real names (Vega, Betelgeuse, HD numbers).
6. Write an annotated video, a star-map image, a per-frame CSV and a JSON
   summary.

Usage
-----
    python startrack.py clip1.mp4 clip2.mp4 -o out/
    python startrack.py photos_folder/ -o out/ --out-fps 8
    python startrack.py night.mp4 --step 5 --astrometry-key YOUR_KEY

Requires: numpy, opencv-python, scipy.
Optional: requests (plate solving), Pillow (EXIF times), rawpy (RAW files),
          astroalign (extra-robust matching across big gaps), ffmpeg (H.264).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".webm", ".wmv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".fits", ".fit"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Geometry helpers (2x3 similarity transforms)
# --------------------------------------------------------------------------
def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:, :2].T + T[:, 2]


def to3(T: np.ndarray) -> np.ndarray:
    return np.vstack([T, [0.0, 0.0, 1.0]])


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A o B (apply B first, then A)."""
    return (to3(A) @ to3(B))[:2]


def invert(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(to3(T))[:2]


def describe_T(T: np.ndarray) -> tuple[float, float, float, float]:
    scale = math.hypot(T[0, 0], T[1, 0])
    rot = math.degrees(math.atan2(T[1, 0], T[0, 0]))
    return rot, scale, T[0, 2], T[1, 2]


# --------------------------------------------------------------------------
# Frame input
# --------------------------------------------------------------------------
@dataclass
class Frame:
    index: int          # running index over the whole sequence (after --step)
    clip: int           # which input clip/sequence this came from
    source: str         # file name
    time_s: float | None  # seconds since start of its clip (video) / first photo (EXIF)
    bgr: np.ndarray     # 8-bit BGR for display
    gray: np.ndarray    # float32 linear-ish intensity for detection


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def to_display(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    f = img.astype(np.float32)
    if f.ndim == 3 and f.shape[2] == 4:
        f = f[:, :, :3]
    lo, hi = np.percentile(f[::4, ::4], (0.5, 99.9))
    f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5
    u8 = (f * 255).astype(np.uint8)
    return u8 if u8.ndim == 3 else cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def read_image(path: str) -> np.ndarray | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        try:
            import rawpy
        except ImportError:
            log(f"  skipping {path}: install 'rawpy' to read RAW files")
            return None
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ext in {".fits", ".fit"}:
        try:
            from astropy.io import fits
        except ImportError:
            log(f"  skipping {path}: install 'astropy' to read FITS files")
            return None
        data = np.asarray(fits.getdata(path), dtype=np.float32)
        if data.ndim == 3:  # (channels, y, x)
            data = np.moveaxis(data, 0, -1)[:, :, ::-1]
        return data
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def exif_time(path: str) -> dt.datetime | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            val = None
            try:
                val = ex.get_ifd(0x8769).get(36867)  # DateTimeOriginal
            except Exception:
                pass
            val = val or ex.get(36867) or ex.get(306)
        if val:
            return dt.datetime.strptime(str(val).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def expand_inputs(inputs: list[str], sort: str) -> list[tuple[str, list[str]]]:
    """Turn CLI inputs into an ordered list of clips: ('video', [path]) or
    ('images', [paths...]). Consecutive images form one sequence."""
    files: list[str] = []
    for p in inputs:
        if os.path.isdir(p):
            names = sorted(os.listdir(p))
            files += [os.path.join(p, n) for n in names
                      if os.path.splitext(n)[1].lower() in VIDEO_EXT | IMAGE_EXT | RAW_EXT]
        elif os.path.isfile(p):
            files.append(p)
        else:
            log(f"warning: {p} not found, skipping")
    if sort == "time":
        def key(f):
            t = exif_time(f) if os.path.splitext(f)[1].lower() not in VIDEO_EXT else None
            return (t.timestamp() if t else os.path.getmtime(f), f)
        files.sort(key=key)

    clips: list[tuple[str, list[str]]] = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in VIDEO_EXT:
            clips.append(("video", [f]))
        else:
            if clips and clips[-1][0] == "images":
                clips[-1][1].append(f)
            else:
                clips.append(("images", [f]))
    return clips


class FrameSource:
    """Iterates frames over all clips; can be iterated repeatedly with
    identical results (needed for the second, drawing pass)."""

    def __init__(self, clips, step=1, scale=1.0, max_frames=None, image_fps=10.0):
        self.clips = clips
        self.step = max(1, int(step))
        self.scale = float(scale)
        self.max_frames = max_frames
        self.image_fps = image_fps
        self.size: tuple[int, int] | None = None  # (w, h)
        self.fps = None

    def _prep(self, img):
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.size is None:
            self.size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return img

    def __iter__(self):
        idx = 0
        for ci, (kind, paths) in enumerate(self.clips):
            if kind == "video":
                path = paths[0]
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    log(f"warning: cannot open video {path}")
                    continue
                vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if self.fps is None:
                    self.fps = vfps / self.step
                raw_i = 0
                while True:
                    ok, img = cap.read()
                    if not ok:
                        break
                    if raw_i % self.step == 0:
                        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if t <= 0 and raw_i > 0:
                            t = raw_i / vfps
                        img = self._prep(img)
                        yield Frame(idx, ci, os.path.basename(path), t, to_display(img), to_gray_float(img))
                        idx += 1
                        if self.max_frames and idx >= self.max_frames:
                            cap.release()
                            return
                    raw_i += 1
                cap.release()
            else:
                if self.fps is None:
                    self.fps = self.image_fps
                times = [exif_time(p) for p in paths]
                t0 = next((t for t in times if t), None)
                for k, (p, t) in enumerate(zip(paths, times)):
                    if k % self.step:
                        continue
                    img = read_image(p)
                    if img is None:
                        continue
                    img = self._prep(img)
                    ts = (t - t0).total_seconds() if (t and t0) else None
                    yield Frame(idx, ci, os.path.basename(p), ts, to_display(img), to_gray_float(img))
                    idx += 1
                    if self.max_frames and idx >= self.max_frames:
                        return


# --------------------------------------------------------------------------
# Star detection
# --------------------------------------------------------------------------
def estimate_background(img: np.ndarray, kernel: int) -> np.ndarray:
    h, w = img.shape
    f = 4
    small = cv2.resize(img, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    k = max(3, (kernel // f) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def mad_sigma(a: np.ndarray) -> float:
    """Robust noise level. Uses iterative 3-sigma-clipped std rather than
    plain MAD, because compressed 8-bit video is so quantised/blocky that the
    MAD often collapses to ~0 and every compression block becomes a 'star'."""
    s = a[::3, ::3].ravel().astype(np.float64)
    mu, sd = float(np.median(s)), float(s.std())
    for _ in range(5):
        keep = np.abs(s - mu) < 3 * sd
        if keep.sum() < 100:
            break
        mu, sd = float(s[keep].mean()), float(s[keep].std())
    return max(sd, 1e-6)


def detect_stars(gray: np.ndarray, args, mask: np.ndarray | None = None) -> np.ndarray:
    """Return (N, 4) array of [x, y, flux, area], brightest first."""
    img = gray
    if args.hot_pixel_filter:
        img = cv2.medianBlur(img, 3)
    resid = img - estimate_background(img, args.bg_kernel)
    # Difference-of-Gaussians: a matched filter for point sources that also
    # cancels smooth sky glow, gradients and compression blotches.
    sm = cv2.GaussianBlur(resid, (0, 0), args.smooth) - cv2.GaussianBlur(resid, (0, 0), args.smooth * 4)
    noise = mad_sigma(sm)
    # Peak threshold from the population of local maxima: almost all maxima
    # are noise, so their spread sets how high a real star must stand. This
    # copes with heavy-tailed (compressed / blocky) noise.
    lm = (sm == cv2.dilate(sm, np.ones((5, 5), np.uint8))) & (sm > 0)
    v = sm[lm]
    if v.size > 50:
        med = float(np.median(v))
        peak_thr = max(args.threshold * noise, med + args.peak_k * 1.4826 * float(np.median(np.abs(v - med))))
    else:
        peak_thr = args.threshold * noise
    det = sm > min(args.threshold * noise, 0.5 * peak_thr)  # extent of each source
    if mask is not None:
        det &= mask
    m = args.border
    if m > 0:
        det[:m, :] = det[-m:, :] = False
        det[:, :m] = det[:, -m:] = False

    n, labels, stats, _ = cv2.connectedComponentsWithStats(det.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((0, 4))
    ys, xs = np.nonzero(det)
    lab = labels[ys, xs]
    w = np.clip(resid[ys, xs], 0, None).astype(np.float64) + 1e-9
    W = np.bincount(lab, weights=w, minlength=n)
    W[W == 0] = 1e-9  # label 0 is the background
    cx = np.bincount(lab, weights=w * xs, minlength=n) / W
    cy = np.bincount(lab, weights=w * ys, minlength=n) / W
    # second moments -> elongation (reject satellite / plane / meteor streaks)
    dx, dy = xs - cx[lab], ys - cy[lab]
    mxx = np.bincount(lab, weights=w * dx * dx, minlength=n) / W
    myy = np.bincount(lab, weights=w * dy * dy, minlength=n) / W
    mxy = np.bincount(lab, weights=w * dx * dy, minlength=n) / W
    tr, det2 = mxx + myy, mxx * myy - mxy ** 2
    disc = np.sqrt(np.clip(tr ** 2 / 4 - det2, 0, None))
    l1, l2 = tr / 2 + disc, np.clip(tr / 2 - disc, 1e-6, None)
    elong = np.sqrt(l1 / l2)

    peak = np.zeros(n)
    np.maximum.at(peak, lab, sm[ys, xs])

    area = stats[:, cv2.CC_STAT_AREA]
    keep = np.arange(n) > 0
    keep &= peak >= peak_thr
    keep &= (area >= args.min_area) & (area <= args.max_area)
    keep &= ~((elong > args.max_elongation) & (area > 6))
    out = np.column_stack([cx, cy, W, area])[keep]
    out = out[np.argsort(-out[:, 2])]
    return out[: args.max_stars]


# --------------------------------------------------------------------------
# Sky-motion estimation between consecutive frames
# --------------------------------------------------------------------------
def _match_and_fit(prev, curr, G, radius, allow_zoom):
    if len(prev) < 3 or len(curr) < 3:
        return None, 0
    T = G
    n_in = 0
    tree_c = cKDTree(curr)
    for r in (radius, max(2.5, radius / 4)):
        p = apply_T(T, prev)
        d, j = tree_c.query(p, distance_upper_bound=r)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            return None, 0
        i_ok, j_ok = np.nonzero(ok)[0], j[ok]
        _, back = cKDTree(p).query(curr[j_ok])
        mutual = back == i_ok
        src, dst = prev[i_ok[mutual]], curr[j_ok[mutual]]
        if len(src) < 3:
            return None, 0
        T_new, inl = cv2.estimateAffinePartial2D(
            src.astype(np.float32), dst.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000, confidence=0.995)
        if T_new is None:
            return None, 0
        _, scale, _, _ = describe_T(T_new)
        if not allow_zoom and abs(scale - 1) > 0.05:
            return None, 0
        T, n_in = T_new, int(inl.sum())
    return T, n_in


def _render_points(pts, flux, size, f=4):
    w, h = size
    im = np.zeros((h // f + 1, w // f + 1), np.float32)
    xi = np.clip((pts[:, 0] / f).astype(int), 0, im.shape[1] - 1)
    yi = np.clip((pts[:, 1] / f).astype(int), 0, im.shape[0] - 1)
    np.add.at(im, (yi, xi), np.sqrt(np.clip(flux, 0, None)))
    im = cv2.GaussianBlur(im, (0, 0), 1.5)
    return im * cv2.createHanningWindow(im.shape[::-1], cv2.CV_32F)


def estimate_motion(prev, curr, guess, args, size, force_wide=False):
    """prev/curr: (N,4) detection arrays. Returns (T, inliers, method)."""
    P, C = prev[: args.match_stars, :2], curr[: args.match_stars, :2]
    need = max(6, int(0.2 * min(len(P), len(C))))
    best = (None, 0, "none")

    cands = [("predict", guess)] if not force_wide else []
    cands.append(("identity", IDENTITY))
    for name, G in cands:
        T, n = _match_and_fit(P, C, G, args.search_radius, args.allow_zoom)
        if n > best[1]:
            best = (T, n, name)
        if best[1] >= need and not force_wide:
            return best

    # Large jump: global shift from phase correlation of rendered star maps
    f = 4
    a = _render_points(P, prev[: args.match_stars, 2], size, f)
    b = _render_points(C, curr[: args.match_stars, 2], size, f)
    (sx, sy), _ = cv2.phaseCorrelate(a, b)
    for sgn in (1, -1):
        G = IDENTITY.copy()
        G[:, 2] = (sgn * sx * f, sgn * sy * f)
        T, n = _match_and_fit(P, C, G, max(args.search_radius, 3 * f), args.allow_zoom)
        if n > best[1]:
            best = (T, n, "phasecorr")
    if best[1] >= need:
        return best

    # Last resort: asterism (triangle) matching, robust to any rotation
    try:
        import astroalign as aa
        _, (s, d) = aa.find_transform(P[:60], C[:60])
        T, _ = cv2.estimateAffinePartial2D(s.astype(np.float32), d.astype(np.float32))
        if T is not None:
            T, n = _match_and_fit(P, C, T, args.search_radius, args.allow_zoom)
            if n > best[1]:
                best = (T, n, "astroalign")
    except Exception:
        pass
    return best


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    pred: np.ndarray
    missed: int = 0
    frames: list = field(default_factory=list)
    xy: list = field(default_factory=list)
    flux: list = field(default_factory=list)

    def add(self, k, x, y, fl):
        self.frames.append(k)
        self.xy.append((x, y))
        self.flux.append(fl)
        self.pred = np.array([x, y])
        self.missed = 0


def assign(tracks: list[Track], dets: np.ndarray, gate: float):
    """Greedy nearest-first assignment with per-track gates. Returns
    (pairs [(track_idx, det_idx)], unmatched det indices)."""
    if not tracks or len(dets) == 0:
        return [], list(range(len(dets)))
    pred = np.array([t.pred for t in tracks])
    gates = np.array([gate * min(3.0, 1 + 0.15 * t.missed) for t in tracks])
    tree = cKDTree(dets[:, :2])
    cand = tree.query_ball_point(pred, r=gates)
    pairs = []
    for ti, js in enumerate(cand):
        for j in js:
            d = math.hypot(*(dets[j, :2] - pred[ti]))
            pairs.append((d, ti, j))
    pairs.sort()
    used_t, used_d, out = set(), set(), []
    for d, ti, j in pairs:
        if ti in used_t or j in used_d:
            continue
        used_t.add(ti)
        used_d.add(j)
        out.append((ti, j))
    return out, [j for j in range(len(dets)) if j not in used_d]


def run_tracking(source: FrameSource, args, mask):
    tracks: list[Track] = []
    cum: list[np.ndarray] = []       # cum[k] maps frame-0 coords -> frame-k coords
    frame_log = []
    next_id = 0
    prev = None
    prev_clip = None
    last_T = IDENTITY.copy()
    best_ref = None                  # (n_dets, index, bgr) frame used for plate solving
    t_start = time.time()

    for fr in source:
        dets = detect_stars(fr.gray, args, mask)
        method, n_in = "start", 0
        if prev is None:
            T = IDENTITY.copy()
            cum.append(IDENTITY.copy())
        else:
            new_clip = fr.clip != prev_clip
            if len(dets) >= 3 and len(prev) >= 3:
                T, n_in, method = estimate_motion(prev, dets, last_T, args, source.size, force_wide=new_clip)
            else:
                T = None
            if T is None or n_in < 4:
                T, method = (IDENTITY.copy() if new_clip else last_T.copy()), "coast"
            else:
                last_T = T
            cum.append(compose(T, cum[-1]))

        active = [t for t in tracks if t.missed <= args.max_missed]
        for t in active:
            t.pred = apply_T(T, t.pred[None])[0]
        pairs, unmatched = assign(active, dets, args.match_radius)
        hit = set()
        for ti, j in pairs:
            active[ti].add(fr.index, dets[j, 0], dets[j, 1], dets[j, 2])
            hit.add(ti)
        for ti, t in enumerate(active):
            if ti not in hit:
                t.missed += 1
        for j in unmatched:
            t = Track(next_id, dets[j, :2].copy())
            t.add(fr.index, dets[j, 0], dets[j, 1], dets[j, 2])
            tracks.append(t)
            next_id += 1

        rot, scale, tx, ty = describe_T(T)
        frame_log.append(dict(frame=fr.index, clip=fr.clip, source=fr.source, time_s=fr.time_s,
                              detections=len(dets), matched=len(pairs), motion=method,
                              inliers=n_in, rot_deg=rot, scale=scale, dx=tx, dy=ty))
        if best_ref is None or len(dets) > best_ref[0]:
            best_ref = (len(dets), fr.index, fr.bgr.copy())
        if fr.index % 25 == 0:
            log(f"  frame {fr.index:5d}  stars {len(dets):4d}  matched {len(pairs):4d}  "
                f"motion={method:<9s} rot={rot:+.4f}deg shift=({tx:+.1f},{ty:+.1f})  "
                f"[{time.time() - t_start:.0f}s]")
        prev, prev_clip = dets, fr.clip

    return tracks, cum, frame_log, best_ref


def finalize_tracks(tracks, cum, args):
    """Drop short / static tracks, compute stable sky positions, assign labels."""
    inv = [invert(c) for c in cum]
    stars = []
    n_static = 0
    for t in tracks:
        if len(t.frames) < args.min_track_len:
            continue
        xy = np.array(t.xy)
        f0, f1 = t.frames[0], t.frames[-1]
        # Static artefact check: field moved but this "star" did not.
        expected = apply_T(compose(cum[f1], inv[f0]), xy[:1])[0]
        if np.hypot(*(expected - xy[0])) > 5 and np.hypot(*(xy[-1] - xy[0])) < 1.5:
            n_static += 1
            continue
        # position in frame-0 sky coordinates (should be ~constant for a real star)
        sky = np.array([apply_T(inv[f], p[None])[0] for f, p in zip(t.frames, xy)])
        stars.append(dict(track=t, frames=np.array(t.frames), xy=xy, flux=np.array(t.flux),
                          sky=sky.mean(0), sky_scatter=float(np.linalg.norm(sky.std(0)))))
    stars.sort(key=lambda s: -np.median(s["flux"]))
    for i, s in enumerate(stars):
        s["label"] = f"S{i + 1:03d}"
        s["name"] = None
    if n_static:
        log(f"  removed {n_static} static artefacts (hot pixels / lights / foreground)")
    return stars


# --------------------------------------------------------------------------
# Optional naming via astrometry.net plate solving
# --------------------------------------------------------------------------
def plate_solve_astrometry(img_path, api_key, fov_hint=None, timeout=900):
    import requests
    base = "https://nova.astrometry.net/api/"
    r = requests.post(base + "login", data={"request-json": json.dumps({"apikey": api_key})}, timeout=60)
    sess = r.json().get("session")
    if not sess:
        raise RuntimeError(f"astrometry.net login failed: {r.text}")
    req = {"session": sess, "allow_commercial_use": "n", "allow_modifications": "n", "publicly_visible": "n"}
    if fov_hint:
        req.update(scale_units="degwidth", scale_type="ul",
                   scale_lower=fov_hint * 0.7, scale_upper=fov_hint * 1.4)
    with open(img_path, "rb") as fh:
        r = requests.post(base + "upload", data={"request-json": json.dumps(req)},
                          files={"file": (os.path.basename(img_path), fh, "image/jpeg")}, timeout=300)
    sub = r.json().get("subid")
    if not sub:
        raise RuntimeError(f"upload failed: {r.text}")
    log(f"  uploaded to astrometry.net (submission {sub}); waiting for solve...")
    t0, job = time.time(), None
    while time.time() - t0 < timeout:
        time.sleep(5)
        if job is None:
            jobs = requests.get(f"{base}submissions/{sub}", timeout=60).json().get("jobs") or []
            job = next((j for j in jobs if j), None)
            continue
        st = requests.get(f"{base}jobs/{job}", timeout=60).json().get("status")
        if st == "success":
            ann = requests.get(f"{base}jobs/{job}/annotations/", timeout=60).json()
            return job, ann.get("annotations", [])
        if st == "failure":
            raise RuntimeError("astrometry.net could not solve the field")
    raise RuntimeError("timed out waiting for astrometry.net")


def attach_names(stars, cum, ref_index, annotations, types, tol):
    if not stars:
        return 0
    pos = apply_T(cum[ref_index], np.array([s["sky"] for s in stars]))
    tree = cKDTree(pos)
    cands = []
    for a in annotations:
        if a.get("type") not in types or not a.get("names"):
            continue
        p = np.array([a["pixelx"], a["pixely"]], float)
        d, i = tree.query(p)
        if d <= tol:
            pri = types.index(a["type"])
            cands.append((pri, d, i, a["names"][0]))
    cands.sort()
    used, n = set(), 0
    for _, _, i, name in cands:
        if i in used:
            continue
        used.add(i)
        stars[i]["name"] = name
        n += 1
    return n


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def color_for(i):
    h = (i * 0.61803398875) % 1.0
    r, g, b = [int(255 * c) for c in __import__("colorsys").hsv_to_rgb(h, 0.65, 1.0)]
    return (b, g, r)


def draw_label(img, text, org, color, fs):
    th = max(1, int(round(fs * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def trail_segments(frames, xy, clip_of):
    """Split a path into pieces that do not jump across clip boundaries."""
    if len(frames) < 2:
        return []
    clips = np.array([clip_of[int(f)] for f in frames])
    cuts = np.nonzero(np.diff(clips))[0] + 1
    return [seg.round().astype(np.int32) for seg in np.split(xy, cuts) if len(seg) > 1]


def star_text(s):
    return f"{s['name']} ({s['label']})" if s["name"] else s["label"]


def render_outputs(source, stars, frame_log, args, outdir):
    w, h = source.size
    fs = max(0.35, w / 2600)
    fps = args.out_fps or source.fps or 10
    tmp = os.path.join(outdir, "_annotated_tmp.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    show_text = {i for i, s in enumerate(stars) if i < args.label_top or s["name"]}

    clip_of = {d["frame"]: d["clip"] for d in frame_log}
    by_frame: dict[int, list[tuple[int, int]]] = {}
    for si, s in enumerate(stars):
        for k, f in enumerate(s["frames"]):
            by_frame.setdefault(int(f), []).append((si, k))

    frames_dir = None
    if args.save_frames:
        frames_dir = os.path.join(outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

    for fr in source:
        img = fr.bgr.copy()
        for si, k in by_frame.get(fr.index, []):
            s = stars[si]
            col = color_for(si)
            lo = max(0, k - args.trail)
            segs = trail_segments(s["frames"][lo:k + 1], s["xy"][lo:k + 1], clip_of)
            if segs:
                cv2.polylines(img, segs, False, col, 1, cv2.LINE_AA)
            x, y = s["xy"][k]
            rad = int(5 + 1.2 * math.log10(max(s["flux"][k], 10)))
            cv2.circle(img, (int(round(x)), int(round(y))), rad, col, 1, cv2.LINE_AA)
            if si in show_text:
                draw_label(img, star_text(s), (int(x) + rad + 2, int(y) - rad), col, fs)
        info = frame_log[fr.index]
        hud = f"{fr.source}  frame {fr.index}  stars tracked {len(by_frame.get(fr.index, []))}"
        if fr.time_s is not None:
            hud += f"  t={fr.time_s:.1f}s"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, 2)
        cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        draw_label(img, hud, (10, th + 10), (230, 230, 230), fs * 1.1)
        writer.write(img)
        if frames_dir:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{fr.index:05d}.jpg"), img)
    writer.release()

    final = os.path.join(outdir, "annotated.mp4")
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", final])
        if r.returncode == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, final)
    else:
        os.replace(tmp, final)
    return final


def write_star_map(ref_bgr, stars, cum, ref_index, frame_log, args, path):
    img = (ref_bgr.astype(np.float32) * 0.8).astype(np.uint8)
    clip_of = {d["frame"]: d["clip"] for d in frame_log}
    w = img.shape[1]
    fs = max(0.35, w / 2600)
    for si, s in enumerate(stars):
        col = color_for(si)
        segs = trail_segments(s["frames"], s["xy"], clip_of)
        if segs:
            cv2.polylines(img, segs, False, col, 1, cv2.LINE_AA)
        x, y = apply_T(cum[ref_index], s["sky"][None])[0]
        cv2.circle(img, (int(x), int(y)), 6, col, 1, cv2.LINE_AA)
        if si < args.label_top or s["name"]:
            draw_label(img, star_text(s), (int(x) + 8, int(y) - 6), col, fs)
    cv2.imwrite(path, img)


def write_tables(stars, frame_log, outdir):
    fl = {d["frame"]: d for d in frame_log}
    with open(os.path.join(outdir, "tracks.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "clip", "source", "time_s", "label", "name", "x", "y", "flux"])
        rows = []
        for s in stars:
            for f, (x, y), flx in zip(s["frames"], s["xy"], s["flux"]):
                d = fl[int(f)]
                rows.append((int(f), d["clip"], d["source"],
                             "" if d["time_s"] is None else round(d["time_s"], 3),
                             s["label"], s["name"] or "", round(x, 2), round(y, 2), round(flx, 1)))
        rows.sort(key=lambda r: (r[0], r[4]))
        wr.writerows(rows)

    summary = []
    for s in stars:
        summary.append(dict(label=s["label"], name=s["name"], n_frames=int(len(s["frames"])),
                            first_frame=int(s["frames"][0]), last_frame=int(s["frames"][-1]),
                            median_flux=float(np.median(s["flux"])),
                            ref_x=round(float(s["sky"][0]), 2), ref_y=round(float(s["sky"][1]), 2),
                            ref_scatter_px=round(s["sky_scatter"], 3)))
    with open(os.path.join(outdir, "stars.json"), "w") as fh:
        json.dump(dict(stars=summary, frames=frame_log), fh, indent=1, default=float)


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Track and label stars across night-sky videos/photos.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+", help="video files, image files and/or folders, in time order")
    p.add_argument("-o", "--out", default="startrack_out", help="output folder")
    g = p.add_argument_group("input")
    g.add_argument("--step", type=int, default=1, help="use every Nth frame (speeds up long videos)")
    g.add_argument("--scale", type=float, default=1.0, help="resize frames by this factor (e.g. 0.5 for 4K)")
    g.add_argument("--max-frames", type=int, default=None)
    g.add_argument("--sort", choices=["name", "time"], default="name",
                   help="order files by name, or by EXIF/modification time")
    g.add_argument("--mask", help="image where white = sky, black = ignore (trees, buildings)")
    g = p.add_argument_group("detection")
    g.add_argument("--threshold", type=float, default=5.0, help="minimum detection threshold in noise sigmas")
    g.add_argument("--peak-k", type=float, default=6.0,
                   help="star peak must exceed typical noise peaks by this many spreads (lower = fainter stars, more false)")
    g.add_argument("--smooth", type=float, default=1.2, help="Gaussian smoothing sigma (px) before thresholding")
    g.add_argument("--bg-kernel", type=int, default=31, help="background-removal kernel size (px)")
    g.add_argument("--min-area", type=int, default=2)
    g.add_argument("--max-area", type=int, default=1500)
    g.add_argument("--max-elongation", type=float, default=3.0, help="reject streaks longer than this ratio")
    g.add_argument("--max-stars", type=int, default=600, help="keep at most this many per frame")
    g.add_argument("--border", type=int, default=4, help="ignore this many px at the edges")
    g.add_argument("--no-hot-pixel-filter", dest="hot_pixel_filter", action="store_false")
    g = p.add_argument_group("tracking")
    g.add_argument("--match-stars", type=int, default=200, help="brightest N stars used for motion estimate")
    g.add_argument("--search-radius", type=float, default=25.0, help="px, for frame-to-frame motion search")
    g.add_argument("--match-radius", type=float, default=4.0, help="px, max error when linking a star to its track")
    g.add_argument("--max-missed", type=int, default=30, help="frames a star may vanish and keep its ID")
    g.add_argument("--min-track-len", type=int, default=5, help="drop stars seen in fewer frames")
    g.add_argument("--allow-zoom", action="store_true", help="allow scale changes between frames")
    g = p.add_argument_group("naming (optional)")
    g.add_argument("--astrometry-key", default=os.environ.get("ASTROMETRY_API_KEY"),
                   help="nova.astrometry.net API key (or set ASTROMETRY_API_KEY)")
    g.add_argument("--fov", type=float, default=None, help="approx. horizontal field of view in degrees (helps solving)")
    g.add_argument("--name-types", default="bright,hd", help="annotation types to use, in priority order")
    g.add_argument("--name-tol", type=float, default=8.0, help="px tolerance when matching names to stars")
    g = p.add_argument_group("output")
    g.add_argument("--out-fps", type=float, default=None, help="annotated video fps (default: source fps / step)")
    g.add_argument("--label-top", type=int, default=40, help="write text labels for the N brightest (+ all named)")
    g.add_argument("--trail", type=int, default=40, help="trail length in frames")
    g.add_argument("--save-frames", action="store_true", help="also save every annotated frame as JPEG")
    g.add_argument("--no-video", action="store_true", help="skip the annotated video")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    clips = expand_inputs(args.inputs, args.sort)
    if not clips:
        sys.exit("no usable inputs found")
    log(f"{len(clips)} clip(s): " + ", ".join(f"{k}:{len(p)}" for k, p in clips))
    source = FrameSource(clips, step=args.step, scale=args.scale, max_frames=args.max_frames,
                         image_fps=args.out_fps or 10.0)

    mask = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            sys.exit(f"cannot read mask {args.mask}")
        if args.scale != 1.0:
            m = cv2.resize(m, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        mask = m > 127

    log("pass 1: detecting and tracking")
    tracks, cum, frame_log, best_ref = run_tracking(source, args, mask)
    if mask is not None and mask.shape[::-1] != source.size:
        log("warning: mask size differed from frames")
    stars = finalize_tracks(tracks, cum, args)
    log(f"  {len(frame_log)} frames, {len(tracks)} raw tracks -> {len(stars)} stars")
    if not stars:
        sys.exit("no stars tracked - try a lower --threshold or check the input")

    _, ref_index, ref_bgr = best_ref
    if args.astrometry_key:
        ref_path = os.path.join(args.out, "solve_frame.jpg")
        cv2.imwrite(ref_path, ref_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            job, ann = plate_solve_astrometry(ref_path, args.astrometry_key, args.fov)
            n = attach_names(stars, cum, ref_index, ann, args.name_types.split(","), args.name_tol)
            log(f"  plate solved (job {job}); named {n} stars")
        except Exception as e:
            log(f"  naming skipped: {e}")

    write_tables(stars, frame_log, args.out)
    write_star_map(ref_bgr, stars, cum, ref_index, frame_log, args, os.path.join(args.out, "star_map.png"))
    if not args.no_video:
        log("pass 2: drawing annotated video")
        path = render_outputs(source, stars, frame_log, args, args.out)
        log(f"  wrote {path}")
    log(f"done -> {args.out}/ (annotated.mp4, star_map.png, tracks.csv, stars.json)")
    return stars


if __name__ == "__main__":
    main()
