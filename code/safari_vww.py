import csv
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from safari_tools import SaFariAnnotation

EXCLUDE = {"sa_fari_005269", "sa_fari_006464"}   # camera-setup footage, no wildlife

# MLPerf Tiny VWW contract (mlcommons/tiny benchmark/training/visual_wake_words,
# dataset built by SiliconLabs eembc/Person_detection/buildPersonDetectionDatabase.py)
IMAGE_SIZE = 96           # default; caches/diagnosis accept image_size= for other resolutions
NUM_CLASSES = 2          # 0 = empty, 1 = animal
MLPERF_AREA_RATIO = 0.025  # bbox area / image area must be > 2.5%
POLICIES = ("drop", "relabel")  # drop = MLPerf (small-only frames skipped); relabel = 2019 VWW



def _valid_loc(x):
    """True if a location_id is usable (not None / nan / empty)."""
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    if isinstance(x, str) and x.strip().lower() in ("", "nan", "none", "null"):
        return False
    return True


def _read_manifest(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header != ["path", "label"]:
            raise ValueError(f"unexpected manifest header: {header}")
        return [(p, int(l)) for p, l in r]


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def _split_stats(rows):
    n = len(rows)
    present = sum(l for _, l in rows)
    return {
        "videos": len({p.split("/")[0] for p, _ in rows}),
        "frames": n,
        "present": present,
        "empty": n - present,
        "present_pct": round(100 * present / n, 2) if n else 0.0,
    }


def _mlperf_resize(job):
    """Resize ONE frame to size x size exactly as MLPerf's dataset builder does
    (SiliconLabs buildPersonDetectionDatabase.py): skimage resize with
    anti-aliasing, no aspect-ratio preservation, gray->RGB, saved as uint8.
    job = (src_path, dst_path, overwrite, size). Returns (status, dst_path)."""
    src, dst, overwrite, size = job
    dst = Path(dst)
    if not overwrite and dst.exists() and dst.stat().st_size > 0:
        return "skipped", str(dst)
    import numpy as np
    import skimage.io as io
    from skimage.color import gray2rgb
    from skimage.transform import resize

    I = io.imread(src)
    if I.ndim == 2:                                   # same as their builder
        I = gray2rgb(I)
    I = resize(I, (size, size), anti_aliasing=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(dst), (255 * I).astype(np.uint8), check_contrast=False)
    return "written", str(dst)


def _mlperf_resize_safe(job):
    try:
        return _mlperf_resize(job)
    except Exception as e:                      # report, don't kill the pool
        return "error", f"{type(e).__name__}: {e}"


class SaFariVWW:
    """MLPerf Tiny Visual Wake Words rules applied to SA-FARI.

    Size rule (as MLPerf's dataset builder): a frame is "animal" if its largest
    animal bbox covers MORE than `threshold` of the frame. Frames whose animals
    are all smaller are handled by `policy`:
      "drop"    -> removed from the dataset (MLPerf, 2.5%)  [default]
      "relabel" -> become label 0           (original 2019 VWW)
    Frames with no animal at all are label 0.
    """

    def __init__(self, annotation: SaFariAnnotation,
                 threshold=MLPERF_AREA_RATIO, policy="drop"):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.ann = annotation
        self.threshold = threshold
        self.policy = policy

    # ---------- size rule ----------

    def _frame_fracs(self, manifest_path):
        """Yield (path, old_label, largest_animal_frac) for every manifest row."""
        idx_cache = {}
        for path, old in _read_manifest(manifest_path):
            video = path.split("/")[0]
            if video not in idx_cache:
                fns = self.ann.video_info(video)["file_names"]
                idx_cache[video] = {p: i for i, p in enumerate(fns)}
            yield path, old, self.ann.frame_max_area_frac(video, idx_cache[video][path])

    def _crowd_annotations(self, videos):
        """MLPerf's builder ignores iscrowd annotations; count any we would include."""
        return sum(1 for v in videos for a in self.ann.annotations_info(v)
                   if a.get("iscrowd", 0))

    def area_stats(self, manifest_path):
        """Measure how the size rule would change a manifest. Read-only."""
        edges = [0.001, 0.005, 0.01, 0.025, 0.05, 0.20]
        names = ["<0.1%", "0.1-0.5%", "0.5-1%", "1-2.5%", "2.5-5%", "5-20%", ">=20%"]

        def bucket(x):
            for e, n in zip(edges, names):
                if x < e:
                    return n
            return names[-1]

        buckets = dict.fromkeys(names, 0)
        n_present = n_empty = small = 0
        mismatch, videos = [], set()
        for path, old, frac in self._frame_fracs(manifest_path):
            videos.add(path.split("/")[0])
            if old == 1:
                n_present += 1
                if frac == 0.0:
                    mismatch.append(path)
                buckets[bucket(frac)] += 1
                if frac <= self.threshold:          # not strictly above -> small
                    small += 1
            else:
                n_empty += 1
                if frac > 0.0:
                    mismatch.append(path)

        total = n_present + n_empty
        if self.policy == "drop":
            after = {"present": n_present - small, "empty": n_empty, "dropped": small}
        else:
            after = {"present": n_present - small, "empty": n_empty + small, "dropped": 0}
        kept = after["present"] + after["empty"]
        after["frames"] = kept
        after["present_pct"] = round(100 * after["present"] / kept, 2) if kept else 0.0
        return {
            "threshold": self.threshold,
            "policy": self.policy,
            "frames": total,
            "present": n_present,
            "empty": n_empty,
            "small_animal_frames": small,
            "small_pct_of_present": round(100 * small / n_present, 2) if n_present else 0.0,
            "size_buckets": buckets,
            "label_mismatches": len(mismatch),
            "mismatch_examples": mismatch[:5],
            "crowd_annotations": self._crowd_annotations(videos),
            "after_rule": after,
        }

    def print_area_stats(self, manifest_path):
        s = self.area_stats(manifest_path)
        action = "dropped" if self.policy == "drop" else "relabeled 0"
        print("=" * 60)
        print(f"SIZE RULE — animal bbox > {self.threshold*100:.2f}% of frame, "
              f"policy '{self.policy}'")
        print("=" * 60)
        print(f"manifest:   {manifest_path}")
        print(f"frames:     {s['frames']}   present {s['present']}   empty {s['empty']}  "
              f"({100*s['present']/s['frames']:.2f}% present)")
        print("largest-animal size, present frames:")
        for k, v in s["size_buckets"].items():
            pct = 100 * v / s["present"] if s["present"] else 0
            print(f"    {k:>9}: {v:>7}  ({pct:5.2f}%)")
        print(f"small-animal frames -> {action}: {s['small_animal_frames']} "
              f"({s['small_pct_of_present']}% of present)")
        a = s["after_rule"]
        print(f"after rule: {a['frames']} frames  present {a['present']}  empty {a['empty']}  "
              f"({a['present_pct']}% present)  dropped {a['dropped']}")
        print(f"label/annotation mismatches: {s['label_mismatches']}"
              + (f"  e.g. {s['mismatch_examples']}" if s["label_mismatches"] else ""))
        print(f"crowd annotations (MLPerf ignores these): {s['crowd_annotations']}")
        return s

    def relabel_manifest(self, src_path, dst_path, exclude=EXCLUDE, verbose=True):
        """Apply the size rule to an existing manifest and write a new one.

        animal (1) if the largest animal covers > threshold of the frame;
        small-only frames are dropped ("drop") or set to 0 ("relabel");
        empty frames stay 0; `exclude` videos are removed. Index file only.
        Writes dst_path (path,label) and dst_path.meta.json.
        """
        src_rows, out_rows = [], []
        excluded = dropped = flipped = 0
        mismatch = []

        for path, old, frac in self._frame_fracs(src_path):
            src_rows.append((path, old))
            if path.split("/")[0] in exclude:
                excluded += 1
                continue
            if (old == 1) != (frac > 0.0):
                mismatch.append(path)
            if frac > self.threshold:
                out_rows.append((path, 1))
            elif frac > 0.0:                      # small animals only
                if self.policy == "drop":
                    dropped += 1
                else:
                    flipped += 1
                    out_rows.append((path, 0))
            else:
                out_rows.append((path, 0))

        # ---- integrity checks (raise on any violation, before writing) ----
        src_paths = {p for p, _ in src_rows}
        out_paths = [p for p, _ in out_rows]
        old_label = dict(src_rows)
        checks = {
            "manifest labels agree with annotation": not mismatch,
            "no duplicate paths": len(out_paths) == len(set(out_paths)),
            "every path came from the source manifest": set(out_paths) <= src_paths,
            "rows conserved (written + excluded + dropped == source)":
                len(out_rows) + excluded + dropped == len(src_rows),
            "excluded videos absent": not any(p.split("/")[0] in exclude for p in out_paths),
            "labels only 0/1": all(l in (0, 1) for _, l in out_rows),
            "no empty frame became animal": all(
                not (old_label[p] == 0 and l == 1) for p, l in out_rows),
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"relabel integrity failed: {failed} "
                                 f"(mismatch e.g. {mismatch[:3]})")

        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _write_rows(dst, out_rows)

        present = sum(l for _, l in out_rows)
        meta = {
            "source": str(src_path),
            "output": str(dst),
            "rule": f"animal bbox area > {self.threshold} of frame; small-only frames: {self.policy}",
            "threshold": self.threshold,
            "policy": self.policy,
            "excluded_videos": sorted(exclude),
            "source_rows": len(src_rows),
            "excluded_rows": excluded,
            "dropped_small": dropped,
            "flipped_1_to_0": flipped,
            "rows": len(out_rows),
            "present": present,
            "empty": len(out_rows) - present,
            "present_pct": round(100 * present / len(out_rows), 2) if out_rows else 0.0,
            "checks": checks,
        }
        with open(dst.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if verbose:
            print("=" * 60)
            print(f"RELABEL  {src_path}  ->  {dst}")
            print("=" * 60)
            print(f"rule:        bbox > {self.threshold*100:.2f}%   policy '{self.policy}'")
            print(f"source rows: {len(src_rows)}   excluded: {excluded}   "
                  f"dropped: {dropped}   flipped 1->0: {flipped}   written: {len(out_rows)}")
            print(f"result:      present {present}  empty {len(out_rows)-present}  "
                  f"({meta['present_pct']}% present)")
            for k, ok in checks.items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
            print(f"meta:        {dst.with_suffix('.meta.json')}")
        return meta

    # ---------- resized cache (MLPerf's pre-resize step) ----------

    @staticmethod
    def build_cache(manifest_paths, src_root, cache_root, workers=8,
                    overwrite=False, verbose=True, image_size=IMAGE_SIZE):
        """Resize every frame listed in the manifest(s) to image_size x image_size
        exactly as MLPerf's builder does, writing cache_root/<video>/<frame>.jpg.
        Use one cache_root per resolution (e.g. data/vww96, data/vww128).

        Reads ONLY paths listed in the manifests (never globs src_root).
        Resumable: existing non-empty outputs are skipped. Verifies every
        output decodes to image_size x image_size x 3 at the end; raises if any failed.
        """
        from PIL import Image

        src_root, cache_root = Path(src_root), Path(cache_root)
        if isinstance(manifest_paths, (str, Path)):
            manifest_paths = [manifest_paths]
        rels = sorted({p for m in manifest_paths for p, _ in _read_manifest(m)})
        jobs = [(str(src_root / r), str(cache_root / r), overwrite, image_size) for r in rels]

        written = skipped = 0
        errors = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (job, res) in enumerate(zip(jobs, pool.map(_mlperf_resize_safe, jobs,
                                                              chunksize=64)), 1):
                status, info = res
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors.append((job[0], info))
                if verbose and i % 5000 == 0:
                    print(f"  {i}/{len(jobs)}  written {written}  skipped {skipped}  "
                          f"errors {len(errors)}")

        bad = []
        for r in rels:
            out = cache_root / r
            try:
                with Image.open(out) as im:
                    if im.size != (image_size, image_size) or im.mode != "RGB":
                        bad.append((r, f"{im.size} {im.mode}"))
            except Exception as e:
                bad.append((r, str(e)))

        summary = {"frames": len(rels), "written": written, "skipped": skipped,
                   "errors": len(errors), "invalid_outputs": len(bad),
                   "image_size": image_size, "cache_root": str(cache_root)}
        if verbose:
            print("=" * 60)
            print(f"CACHE  {len(rels)} frames {image_size}x{image_size} -> {cache_root}")
            print("=" * 60)
            print(f"written {written}   skipped {skipped}   errors {len(errors)}   "
                  f"invalid {len(bad)}")
            for e in (errors + bad)[:5]:
                print("  ", e)
        if errors or bad:
            raise RuntimeError(f"cache incomplete: {len(errors)} errors, {len(bad)} invalid")
        return summary

    # ---------- train/val split (index files only) ----------

    def split_manifest(self, manifest_path, out_dir, val_frac=0.15, seed=0, verbose=True):
        """Partition a manifest into train/val by camera location.

        group = location_id when valid (same camera => same background, so it
                must not span both sides); otherwise the video itself.
        A video never splits. Writes split_train.csv, split_val.csv and
        split_meta.json to out_dir. Index files only: no image is touched.
        """
        rows = _read_manifest(manifest_path)

        by_video = defaultdict(list)
        for p, l in rows:
            by_video[p.split("/")[0]].append((p, l))

        # group each video by its camera location (fallback: the video)
        group_of, with_loc = {}, 0
        for v in by_video:
            loc = self.ann.video_info(v).get("location_id")
            if _valid_loc(loc):
                group_of[v] = f"loc:{loc}"
                with_loc += 1
            else:
                group_of[v] = f"vid:{v}"

        groups = defaultdict(list)
        for v, g in group_of.items():
            groups[g].append(v)
        group_frames = {g: sum(len(by_video[v]) for v in vs) for g, vs in groups.items()}

        # deterministic shuffle of groups; fill val up to val_frac of frames
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        target = val_frac * sum(group_frames.values())
        val_groups, acc = set(), 0
        for g in keys:
            if acc >= target:
                break
            val_groups.add(g)
            acc += group_frames[g]

        train_rows, val_rows = [], []
        for v, vrows in by_video.items():
            (val_rows if group_of[v] in val_groups else train_rows).extend(vrows)
        train_rows.sort()
        val_rows.sort()

        # ---- integrity checks (raise on any violation) ----
        tr_v = {p.split("/")[0] for p, _ in train_rows}
        va_v = {p.split("/")[0] for p, _ in val_rows}
        all_paths = [p for p, _ in train_rows + val_rows]
        checks = {
            "no video in both splits": not (tr_v & va_v),
            "no location group in both splits":
                not ({group_of[v] for v in tr_v} & {group_of[v] for v in va_v}),
            "excluded videos absent": not ((tr_v | va_v) & EXCLUDE),
            "no duplicate paths": len(all_paths) == len(set(all_paths)),
            "labels only 0/1": all(l in (0, 1) for _, l in train_rows + val_rows),
            "rows conserved (train + val == manifest)":
                len(train_rows) + len(val_rows) == len(rows),
            "every split path came from the manifest":
                set(all_paths) <= {p for p, _ in rows},
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"split integrity failed: {failed}")

        n_loc = sum(1 for g in groups if g.startswith("loc:"))
        meta = {
            "manifest": str(manifest_path),
            "seed": seed,
            "val_frac_target": val_frac,
            "val_frac_actual": round(len(val_rows) / len(rows), 4) if rows else 0.0,
            "videos": len(by_video),
            "videos_with_location": with_loc,
            "videos_without_location": len(by_video) - with_loc,
            "location_groups": n_loc,
            "fallback_video_groups": len(groups) - n_loc,
            "largest_group_frames": max(group_frames.values()) if group_frames else 0,
            "train": _split_stats(train_rows),
            "val": _split_stats(val_rows),
            "checks": checks,
        }

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_rows(out / "split_train.csv", train_rows)
        _write_rows(out / "split_val.csv", val_rows)
        with open(out / "split_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if verbose:
            print("=" * 60)
            print(f"SPLIT  {manifest_path}")
            print("=" * 60)
            print(f"seed / val target:  {seed} / {val_frac}  (actual {meta['val_frac_actual']})")
            print(f"videos:             {meta['videos']}  (with location {with_loc}, "
                  f"without {meta['videos_without_location']})")
            print(f"groups:             {n_loc} locations + "
                  f"{meta['fallback_video_groups']} video-fallback")
            print(f"largest group:      {meta['largest_group_frames']} frames")
            for name in ("train", "val"):
                s = meta[name]
                print(f"{name:5}: {s['videos']:>6} videos  {s['frames']:>7} frames  "
                      f"present {s['present']:>6} ({s['present_pct']}%)  empty {s['empty']:>6}")
            for k, ok in checks.items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
            print(f"wrote: split_train.csv, split_val.csv, split_meta.json -> {out}")
        return train_rows, val_rows, meta

    def location_overlap(self, other_ann):
        """Valid camera locations shared between this annotation and another
        (e.g. official train vs test). Informative only."""
        def locs(a):
            return {a.video_info(v).get("location_id") for v in a.video_names()
                    if _valid_loc(a.video_info(v).get("location_id"))}
        mine, theirs = locs(self.ann), locs(other_ann)
        return {"this": len(mine), "other": len(theirs), "shared": len(mine & theirs)}

    # ---------- evaluation / diagnosis (use after every training run) ----------

    SIZE_BUCKETS = [(0.025, 0.05, "2.5-5%"), (0.05, 0.10, "5-10%"),
                    (0.10, 0.20, "10-20%"), (0.20, 1.01, ">=20%")]

    @staticmethod
    def _binary_metrics(y, p, threshold=0.5):
        """y: 0/1 truth, p: P(animal). Returns accuracy, balanced accuracy,
        per-class recall, precision/F1 for animal, ROC AUC, confusion."""
        import numpy as np
        from scipy.stats import rankdata

        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        n_pos, n_neg = tp + fn, tn + fp
        rec_a = tp / n_pos if n_pos else float("nan")
        rec_e = tn / n_neg if n_neg else float("nan")
        prec_a = tp / (tp + fp) if tp + fp else float("nan")
        f1 = 2 * prec_a * rec_a / (prec_a + rec_a) if tp else 0.0
        if n_pos and n_neg:                       # rank-based ROC AUC (ties averaged)
            r = rankdata(p)
            auc = (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        else:
            auc = float("nan")
        bal = (rec_a + rec_e) / 2 if n_pos and n_neg else float("nan")
        return {"n": int(len(y)), "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
                "balanced_accuracy": bal, "recall_animal": rec_a, "recall_empty": rec_e,
                "precision_animal": prec_a, "f1_animal": f1, "auc": auc,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @staticmethod
    def _predict(model, images):
        """Works with a Keras model or any callable: (N,S,S,3) in [0,1] -> (N,2)."""
        import numpy as np
        if hasattr(model, "predict"):
            out = model.predict(images, verbose=0)
        else:
            out = model(images)
        out = np.asarray(out)
        return out[:, 1] if out.ndim == 2 else out.reshape(-1)

    def diagnose(self, model, manifest_path, image_root, name="eval", threshold=0.5,
                 batch_size=512, ir_tolerance=3.0, min_camera_frames=20,
                 out_csv=None, verbose=True, image_size=None):
        """Evaluate a model on a manifest and break performance down.

        Inputs are read exactly as in training (cached JPEG at the model's
        input size, RGB, /255,
        no augmentation), only from manifest paths. Reports:
          overall  - accuracy, balanced accuracy, recall/precision, F1, ROC AUC
          size     - animal recall by size of the largest animal
          light    - color vs. grayscale/IR (night) frames  [heuristic]
          cameras  - worst cameras by error rate (background memorization)
          threshold- decision threshold that maximizes balanced accuracy on
                     THIS set (choose it on val, never on test)
        `model`: Keras model or callable returning (N,2) softmax or (N,) P(animal).
        Optionally writes per-frame results to out_csv. Returns a dict.
        """
        import numpy as np
        from PIL import Image

        root = Path(image_root)
        if image_size is None:                    # take it from the model when possible
            shp = getattr(model, "input_shape", None)
            image_size = shp[1] if shp and shp[1] else IMAGE_SIZE
        paths, labels, fracs = [], [], []
        for path, label, frac in self._frame_fracs(manifest_path):
            paths.append(path); labels.append(label); fracs.append(frac)
        labels = np.asarray(labels); fracs = np.asarray(fracs)

        probs = np.empty(len(paths), dtype=float)
        is_ir = np.empty(len(paths), dtype=bool)
        for s in range(0, len(paths), batch_size):
            chunk = paths[s:s + batch_size]
            arr = np.empty((len(chunk), image_size, image_size, 3), dtype=np.uint8)
            for i, p in enumerate(chunk):
                with Image.open(root / p) as im:
                    im = im.convert("RGB")
                    if im.size != (image_size, image_size):
                        raise ValueError(f"{p}: expected {image_size}x{image_size} cache image, got {im.size}")
                    arr[i] = np.asarray(im)
            a = arr.astype(np.int16)
            chroma = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean(axis=(1, 2)) / 2
            is_ir[s:s + len(chunk)] = chroma < ir_tolerance
            probs[s:s + len(chunk)] = self._predict(model, arr.astype(np.float32) / 255.0)

        overall = self._binary_metrics(labels, probs, threshold)

        # animal recall by size (positives only)
        by_size = {}
        for lo, hi, nm in self.SIZE_BUCKETS:
            m = (labels == 1) & (fracs > lo) & (fracs <= hi)
            n = int(m.sum())
            by_size[nm] = {"n": n, "recall_animal":
                           float((probs[m] >= threshold).mean()) if n else float("nan")}
        small_neg = (labels == 0) & (fracs > 0)
        if small_neg.any():                       # only with policy="relabel"
            by_size["label0_small_animal"] = {"n": int(small_neg.sum()), "recall_empty":
                                              float((probs[small_neg] < threshold).mean())}

        # color vs IR
        by_light = {k: self._binary_metrics(labels[m], probs[m], threshold)
                    for k, m in (("color", ~is_ir), ("ir_gray", is_ir)) if m.any()}

        # cameras
        cam = []
        for p in paths:
            loc = self.ann.video_info(p.split("/")[0]).get("location_id")
            cam.append(f"loc:{loc}" if _valid_loc(loc) else f"vid:{p.split('/')[0]}")
        cam = np.asarray(cam)
        wrong = (probs >= threshold).astype(int) != labels
        cams = []
        for c in np.unique(cam):
            m = cam == c
            if m.sum() >= min_camera_frames:
                cams.append({"camera": str(c), "n": int(m.sum()),
                             "error_rate": float(wrong[m].mean()),
                             "animal_pct": float(labels[m].mean())})
        cams.sort(key=lambda d: -d["error_rate"])
        err_rates = np.array([c["error_rate"] for c in cams]) if cams else np.array([])

        # threshold sweep (balanced accuracy)
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        bals = [self._binary_metrics(labels, probs, t)["balanced_accuracy"] for t in grid]
        best_t = float(grid[int(np.nanargmax(bals))])
        best = self._binary_metrics(labels, probs, best_t)

        result = {
            "name": name, "manifest": str(manifest_path), "threshold": threshold,
            "image_size": image_size,
            "overall": overall, "by_size": by_size, "by_light": by_light,
            "cameras": {"count": len(cams),
                        "error_rate_median": float(np.median(err_rates)) if cams else float("nan"),
                        "worst": cams[:10]},
            "best_threshold": {"threshold": best_t, **best},
            "pred_animal_pct": float((probs >= threshold).mean()),
            "true_animal_pct": float(labels.mean()),
        }

        if out_csv:
            out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "label", "prob_animal", "pred", "animal_frac", "ir_gray", "camera"])
                for i, p in enumerate(paths):
                    w.writerow([p, int(labels[i]), f"{probs[i]:.6f}", int(probs[i] >= threshold),
                                f"{fracs[i]:.6f}", int(is_ir[i]), cam[i]])
            result["per_frame_csv"] = str(out)

        if verbose:
            self.print_diagnosis(result)
        return result

    @staticmethod
    def print_diagnosis(r):
        o = r["overall"]
        print("=" * 66)
        print(f"DIAGNOSIS [{r['name']}]  {r['manifest']}  (threshold {r['threshold']})")
        print("=" * 66)
        print(f"frames {o['n']}   true animal {100*r['true_animal_pct']:.1f}%   "
              f"predicted animal {100*r['pred_animal_pct']:.1f}%")
        print(f"accuracy {o['accuracy']:.4f}   BALANCED {o['balanced_accuracy']:.4f}   AUC {o['auc']:.4f}")
        print(f"recall animal {o['recall_animal']:.4f}   recall empty {o['recall_empty']:.4f}   "
              f"precision animal {o['precision_animal']:.4f}   F1 {o['f1_animal']:.4f}")
        print(f"confusion  true\\pred   empty  animal")
        print(f"           empty   {o['tn']:>7} {o['fp']:>7}")
        print(f"           animal  {o['fn']:>7} {o['tp']:>7}")
        print("-- animal recall by size of largest animal --")
        for k, v in r["by_size"].items():
            key = "recall_animal" if "recall_animal" in v else "recall_empty"
            print(f"   {k:>20}: n={v['n']:>6}  {key} {v[key]:.4f}")
        print("-- color vs grayscale/IR (heuristic) --")
        for k, v in r["by_light"].items():
            print(f"   {k:>8}: n={v['n']:>6}  acc {v['accuracy']:.4f}  bal {v['balanced_accuracy']:.4f}  "
                  f"rec animal {v['recall_animal']:.4f}  rec empty {v['recall_empty']:.4f}")
        c = r["cameras"]
        print(f"-- cameras (>= min frames): {c['count']}, median error rate {c['error_rate_median']:.3f}; worst:")
        for w in c["worst"][:5]:
            print(f"   {w['camera']:>14}: n={w['n']:>5}  error {w['error_rate']:.3f}  "
                  f"animal {100*w['animal_pct']:.0f}%")
        b = r["best_threshold"]
        print(f"-- best threshold on this set: {b['threshold']:.2f} -> balanced {b['balanced_accuracy']:.4f} "
              f"(rec animal {b['recall_animal']:.4f}, rec empty {b['recall_empty']:.4f})")
        if "per_frame_csv" in r:
            print(f"per-frame results: {r['per_frame_csv']}")