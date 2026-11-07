"""Synthetic night sky: stars rotating about an off-frame celestial pole,
a tree silhouette that hides stars, hot pixels, sky gradient, noise, and
two clips separated by a time gap. Writes ground truth to gt.npz."""
import os, sys
import numpy as np, cv2

rng = np.random.default_rng(1)
W, H = 1280, 720
N = 350
out = sys.argv[1] if len(sys.argv) > 1 else "syn"
os.makedirs(out, exist_ok=True)

pole = np.array([1700.0, -900.0])              # celestial pole off-frame (top right)
base = np.column_stack([rng.uniform(-300, W + 300, N), rng.uniform(-300, H + 300, N)])
flux = np.exp(rng.normal(6.0, 1.0, N))         # lognormal brightness
hot = np.column_stack([rng.integers(0, W, 40), rng.integers(0, H, 40)])

yy, xx = np.mgrid[0:H, 0:W]
tree = (yy > H - 160 - 120 * np.exp(-((xx - 300) / 120.0) ** 2)) | ((xx < 140) & (yy > 250))

def render(angle):
    c, s = np.cos(angle), np.sin(angle)
    p = (base - pole) @ np.array([[c, s], [-s, c]]) + pole
    img = 20 + 15 * (yy / H) + rng.normal(0, 2.5, (H, W))
    vis = (p[:, 0] > -10) & (p[:, 0] < W + 10) & (p[:, 1] > -10) & (p[:, 1] < H + 10)
    for (x, y), f in zip(p[vis], flux[vis]):
        x0, y0 = int(x) - 6, int(y) - 6
        ys, xs = np.mgrid[y0:y0 + 13, x0:x0 + 13]
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        g = f / (2 * np.pi * 1.4 ** 2) * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * 1.4 ** 2))
        img[ys[ok], xs[ok]] += g[ok]
    img[tree] = 8 + rng.normal(0, 1.5, tree.sum())
    img[hot[:, 1], hot[:, 0]] = 255
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    visible = vis.copy()
    pi = np.clip(p.round().astype(int), [0, 0], [W - 1, H - 1])
    visible &= ~tree[pi[:, 1], pi[:, 0]]
    visible &= (p[:, 0] > 8) & (p[:, 0] < W - 8) & (p[:, 1] > 8) & (p[:, 1] < H - 8)
    return cv2.merge([u8, u8, u8]), p, visible

rate = np.radians(0.012)   # rad per frame (sped-up sky rotation)
gt = []
for clip, (start, n) in enumerate([(0, 120), (220, 80)]):   # 100-frame gap between clips
    vw = cv2.VideoWriter(f"{out}/clip{clip + 1}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for k in range(n):
        img, p, vis = render((start + k) * rate)
        vw.write(img)
        gt.append((p, vis))
    vw.release()
np.savez(f"{out}/gt.npz", pos=np.array([g[0] for g in gt]), vis=np.array([g[1] for g in gt]), flux=flux)

# photo sequence: bigger jumps between shots
os.makedirs(f"{out}/photos", exist_ok=True)
pgt = []
for k in range(25):
    img, p, vis = render(k * rate * 8)
    cv2.imwrite(f"{out}/photos/IMG_{k:04d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    pgt.append((p, vis))
np.savez(f"{out}/photos_gt.npz", pos=np.array([g[0] for g in pgt]), vis=np.array([g[1] for g in pgt]), flux=flux)
print("ok")
