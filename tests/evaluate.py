"""Compare startrack's tracks.csv to ground truth."""
import sys, csv, collections
import numpy as np
from scipy.spatial import cKDTree

csv_path, gt_path = sys.argv[1], sys.argv[2]
gt = np.load(gt_path)
pos, vis = gt["pos"], gt["vis"]

rows = list(csv.DictReader(open(csv_path)))
label_to_gt = collections.defaultdict(list)
n_det, n_hit = 0, 0
for r in rows:
    f = int(r["frame"]); x, y = float(r["x"]), float(r["y"])
    d, i = cKDTree(pos[f]).query([x, y])
    n_det += 1
    if d < 2.0:
        n_hit += 1
        label_to_gt[r["label"]].append(i)
    else:
        label_to_gt[r["label"]].append(-1)

purity, frag = [], collections.defaultdict(set)
for lab, ids in label_to_gt.items():
    c = collections.Counter(ids)
    gid, cnt = c.most_common(1)[0]
    purity.append(cnt / len(ids))
    if gid >= 0:
        frag[gid].add(lab)

visible_long = np.nonzero(vis.sum(0) >= 10)[0]
covered = [g for g in visible_long if g in frag]
pos_err = []
for r in rows:
    f = int(r["frame"])
    d, _ = cKDTree(pos[f]).query([float(r["x"]), float(r["y"])])
    if d < 2: pos_err.append(d)

print(f"labels: {len(label_to_gt)}   detections in csv: {n_det}   matched to a true star: {n_hit / n_det:.1%}")
print(f"mean centroid error: {np.mean(pos_err):.3f}px")
print(f"track purity (all labels): mean {np.mean(purity):.3f}, min {np.min(purity):.3f}, "
      f"labels <95% pure: {sum(p < .95 for p in purity)}")
print(f"true stars visible >=10 frames: {len(visible_long)}, tracked: {len(covered)} "
      f"({len(covered) / len(visible_long):.1%})")
nl = [len(frag[g]) for g in covered]
print(f"labels per true star: mean {np.mean(nl):.3f}, stars split into >1 label: {sum(n > 1 for n in nl)}")
