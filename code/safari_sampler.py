from pathlib import Path
from math import ceil
import json
import csv

from safari_tools import SaFariAnnotation


def even_spaced_positions(n, k):
    """
    Independent, standalone even-spacing: return k distinct positions in [0, n-1].
    This is the REFERENCE implementation used by validate() so the check does not
    run the same code it is checking. Kept module-level and dependency-free.

    Guarantees exactly min(k, n) distinct positions, ascending.
    """
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [round((n - 1) / 2)]
    out = []
    seen = set()
    for i in range(k):
        p = round(i * (n - 1) / (k - 1))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class SaFariSampler:
    """
    Builds a download plan + label manifest for whole-image binary
    classification (animal present / empty), on top of SaFariAnnotation.

    Relies on the annotation providing: video_names(),
    animal_presence_frame_label(name), frame_path(name, idx).

    One uniform rule per stratum. A stratum = one class (present or empty)
    within one video. For each stratum of size n, select k = ceil(pct * n)
    indices by deterministic even spacing. No randomness in selection.
    """

    def __init__(self, annotation: SaFariAnnotation, pct=0.10, seed=0):
        if not (0.0 < pct <= 1.0):
            raise ValueError(f"pct must be in (0, 1], got {pct}")
        self.ann = annotation
        self.pct = pct
        self.seed = seed  # recorded in metadata only; NOT used for selection

    # ---------- selection core ----------

    def _even_spaced(self, indices, k):
        """Select k of the given ascending stratum indices by even spacing.

        For 2 <= k < n the positions round(i*(n-1)/(k-1)) are strictly
        increasing (per-step gap (n-1)/(k-1) > 1 whenever k < n), so they are
        always distinct and this returns exactly k indices. The assertion makes
        that guarantee explicit rather than relying on a silent dedupe.
        """
        n = len(indices)
        if k >= n:
            return list(indices)
        if k == 1:
            return [indices[round((n - 1) / 2)]]
        out = [indices[round(i * (n - 1) / (k - 1))] for i in range(k)]
        assert len(out) == len(set(out)) == k, (
            f"even-spacing produced {len(out)} ({len(set(out))} distinct) for n={n} k={k}"
        )
        return out

    def _k_for(self, n):
        return 0 if n == 0 else ceil(self.pct * n)

    def _sample_stratum(self, indices):
        n = len(indices)
        if n == 0:
            return []
        return self._even_spaced(indices, self._k_for(n))

    def _resolve_videos(self, videos):
        names = self.ann.video_names()
        if videos is None:
            return names
        if isinstance(videos, slice):
            return names[videos]
        if isinstance(videos, range):
            return [names[i] for i in videos]
        if isinstance(videos, (list, tuple)):
            return list(videos)
        raise TypeError(
            f"videos must be None, a slice, a range, or a list of names; got {type(videos).__name__}"
        )

    def _strata(self, video_name):
        """Return (present_indices, empty_indices) for a video — the raw split."""
        labels = self.ann.animal_presence_frame_label(video_name)
        present = [i for i, lab in enumerate(labels) if lab == 1]
        empty = [i for i, lab in enumerate(labels) if lab == 0]
        return present, empty, labels

    def _sample_video(self, video_name):
        present, empty, _ = self._strata(video_name)
        return (
            self._sample_stratum(present),
            self._sample_stratum(empty),
            len(present),
            len(empty),
        )

    # ---------- build / outputs ----------

    def build(self, videos=None):
        plan = {}
        manifest = []
        total_present = 0
        total_empty = 0

        for video_name in self._resolve_videos(videos):
            sel_present, sel_empty, _, _ = self._sample_video(video_name)
            selected = sorted(sel_present + sel_empty)
            if not selected:
                continue
            plan[video_name] = selected
            for idx in sel_present:
                manifest.append((self.ann.frame_path(video_name, idx), 1))
            for idx in sel_empty:
                manifest.append((self.ann.frame_path(video_name, idx), 0))
            total_present += len(sel_present)
            total_empty += len(sel_empty)

        metadata = {
            "pct": self.pct,
            "seed": self.seed,
            "total_selected": total_present + total_empty,
            "present": total_present,
            "empty": total_empty,
            "videos": len(plan),
        }
        return plan, manifest, metadata

    def print_plan(self, videos=None):
        tot_present = tot_empty = shown = 0
        for video_name in self._resolve_videos(videos):
            sel_present, sel_empty, n_present, n_empty = self._sample_video(video_name)
            if not sel_present and not sel_empty:
                continue
            shown += 1
            total = len(sel_present) + len(sel_empty)
            print(f"{video_name}:  present {len(sel_present)}/{n_present}  "
                  f"empty {len(sel_empty)}/{n_empty}  -> {total} frames")
            print(f"    present idx: {sel_present}")
            print(f"    empty   idx: {sel_empty}")
            tot_present += len(sel_present)
            tot_empty += len(sel_empty)
        print(f"TOTAL: {shown} videos, {tot_present + tot_empty} frames "
              f"(present {tot_present}, empty {tot_empty})  [pct={self.pct}, seed={self.seed}]")

    def save(self, out_dir="data/plans", tag=None, videos=None):
        plan, manifest, metadata = self.build(videos=videos)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = tag if tag else f"pct{self.pct}_seed{self.seed}"
        plan_path = out / f"plan_{stem}.json"
        manifest_path = out / f"manifest_{stem}.csv"
        meta_path = out / f"meta_{stem}.json"
        with open(plan_path, "w") as f:
            json.dump(plan, f)
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for path, label in manifest:
                w.writerow([path, label])
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"plan:     {plan_path}  ({metadata['videos']} videos)")
        print(f"manifest: {manifest_path}  ({metadata['total_selected']} frames)")
        print(f"meta:     {meta_path}")
        print(f"  present={metadata['present']}  empty={metadata['empty']}")
        return plan, manifest, metadata

    # ---------- independent reference (for validate) ----------

    def _reference_selection(self, video_name):
        """
        Compute the expected selection for a video by an INDEPENDENT route:
        - split labels into strata (same source data)
        - use the module-level even_spaced_positions() — a SEPARATE implementation
          from _even_spaced — to choose positions, then map to indices.
        Returns (present_sel, empty_sel). Used only by validate() so the check
        does not run the code it is verifying.
        """
        present, empty, _ = self._strata(video_name)

        def pick(stratum):
            n = len(stratum)
            if n == 0:
                return []
            k = ceil(self.pct * n)           # formula, recomputed here
            pos = even_spaced_positions(n, k)  # independent implementation
            return [stratum[p] for p in pos]

        return pick(present), pick(empty)

    # ---------- validation ----------

    def validate(self, plan=None, manifest_path=None, images_root=None,
                 verify_decode=False, videos=None):
        """
        Full self-check. The expected selection is computed by an INDEPENDENT
        implementation (_reference_selection / even_spaced_positions), so the
        which-frames checks genuinely verify the sampler rather than re-running it.

        verify_decode: if True and images_root given, each planned frame is
        decoded (cv2 if available, else JPEG EOI marker check) to catch
        truncated-but-nonzero files, not just missing/zero-byte ones.
        """
        problems = []

        def check(name, ok, detail=""):
            print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
            if not ok:
                problems.append(name)

        # ---- independent ground truth ----
        ref_plan, ref_present, ref_empty = {}, {}, {}
        k_mismatch = []
        for v in self._resolve_videos(videos):
            present, empty, _ = self._strata(v)
            sp, se = self._reference_selection(v)
            # independent count expectation
            exp_kp = ceil(self.pct * len(present)) if present else 0
            exp_ke = ceil(self.pct * len(empty)) if empty else 0
            if len(sp) != exp_kp:
                k_mismatch.append(f"{v} present got {len(sp)} exp {exp_kp}")
            if len(se) != exp_ke:
                k_mismatch.append(f"{v} empty got {len(se)} exp {exp_ke}")
            merged = sorted(sp + se)
            if merged:
                ref_plan[v] = merged
                ref_present[v] = sp
                ref_empty[v] = se
        check("k = ceil(pct*n) holds (independent formula)", not k_mismatch,
              f"{len(k_mismatch)} mismatches e.g. {k_mismatch[:3]}" if k_mismatch else "all strata")

        # ---- plan under test ----
        if plan is None:
            plan, _, _ = self.build(videos=videos)

        check("plan has exactly the expected videos", set(plan) == set(ref_plan),
              f"plan={len(plan)} ref={len(ref_plan)}")
        mism = [v for v in plan if v in ref_plan and plan[v] != ref_plan[v]]
        check("plan indices match INDEPENDENT reference", not mism,
              f"{len(mism)} videos differ e.g. {mism[:3]}" if mism else "")

        bad = []
        for v, idxs in plan.items():
            if not idxs: bad.append((v, "empty"))
            elif idxs != sorted(idxs): bad.append((v, "unsorted"))
            elif len(set(idxs)) != len(idxs): bad.append((v, "dupes"))
            elif any(not isinstance(i, int) for i in idxs): bad.append((v, "nonint"))
        check("plan lists non-empty, sorted, unique, int", not bad, str(bad[:3]) if bad else "")

        oob = []
        for v, idxs in plan.items():
            n = len(self.ann.video_info(v)["file_names"])
            if idxs and (idxs[-1] >= n or idxs[0] < 0):
                oob.append((v, idxs[-1], n))
        check("all indices within video frame count", not oob, str(oob[:3]) if oob else "")

        total_plan = sum(len(x) for x in plan.values())

        # ---- manifest ----
        if manifest_path:
            rows = list(csv.reader(open(manifest_path)))
            header, rows = rows[0], rows[1:]
            check("manifest header is path,label", header == ["path", "label"], str(header))
            labelset = set(r[1] for r in rows)
            check("manifest labels are only 0/1", labelset <= {"0", "1"}, str(labelset))
            exp_pairs = set()
            for v in ref_present:
                for i in ref_present[v]:
                    exp_pairs.add((self.ann.frame_path(v, i), "1"))
                for i in ref_empty[v]:
                    exp_pairs.add((self.ann.frame_path(v, i), "0"))
            got_pairs = set((r[0], r[1]) for r in rows)
            check("manifest row count matches plan", len(rows) == total_plan,
                  f"manifest={len(rows)} plan={total_plan}")
            check("no duplicate manifest rows", len(got_pairs) == len(rows),
                  f"{len(rows) - len(got_pairs)} dupes")
            check("manifest == INDEPENDENT (path,label) set", got_pairs == exp_pairs,
                  f"missing={len(exp_pairs - got_pairs)} extra={len(got_pairs - exp_pairs)}")
            seen = {}
            conflict = sum(1 for p, l in rows if p in seen and seen[p] != l or seen.update({p: l}))
            # (the walrus-free conflict count)
            seen = {}
            conflict = 0
            for p, l in rows:
                if p in seen and seen[p] != l:
                    conflict += 1
                seen[p] = l
            check("no path with conflicting labels", conflict == 0, f"{conflict} conflicts")

        # ---- disk ----
        if images_root:
            root = Path(images_root)
            missing, undecodable = [], []
            checked = 0
            for v, idxs in plan.items():
                for i in idxs:
                    rel = self.ann.frame_path(v, i)
                    fp = root / rel
                    checked += 1
                    if not fp.exists() or fp.stat().st_size == 0:
                        missing.append(rel)
                    elif verify_decode and not _is_decodable_jpeg(fp):
                        undecodable.append(rel)
            check("all planned frames present & non-empty on disk", not missing,
                  f"{len(missing)}/{checked} missing/zero e.g. {missing[:3]}" if missing else "")
            if verify_decode:
                check("all planned frames decodable (not truncated)", not undecodable,
                      f"{len(undecodable)}/{checked} undecodable e.g. {undecodable[:3]}"
                      if undecodable else "")

        print(f"\n{'ALL CHECKS PASSED' if not problems else f'{len(problems)} PROBLEM(S)'} "
              f"— plan: {len(plan)} videos, {total_plan} frames")
        return {"ok": not problems, "problems": problems,
                "videos": len(plan), "frames": total_plan}

    # ---------- deep trace (bulletproof per-video debug) ----------

    def trace_video(self, video_name, verbose=True):
        """
        Deep, self-checking trace of ONE video's extraction. Recomputes the
        selection by BOTH the production path (_sample_video) and the
        INDEPENDENT reference (_reference_selection) and asserts they agree,
        re-derives k from the formula, and verifies every invariant inline.
        Returns a dict of everything computed; raises AssertionError on any
        internal disagreement. Use to prove a single video end-to-end.
        """
        present, empty, labels = self._strata(video_name)
        n_total = len(labels)
        n_present, n_empty = len(present), len(empty)

        # production selection
        sp, se, _, _ = self._sample_video(video_name)
        # independent selection
        rsp, rse = self._reference_selection(video_name)

        # formula k
        kp = self._k_for(n_present)
        ke = self._k_for(n_empty)

        checks = {}
        checks["labels_len == present+empty"] = (n_present + n_empty == n_total)
        checks["present_k == ceil(pct*n)"] = (len(sp) == kp)
        checks["empty_k == ceil(pct*n)"] = (len(se) == ke)
        checks["present matches independent ref"] = (sp == rsp)
        checks["empty matches independent ref"] = (se == rse)
        checks["present within range"] = all(0 <= i < n_total for i in sp)
        checks["empty within range"] = all(0 <= i < n_total for i in se)
        checks["present ascending unique"] = (sp == sorted(set(sp)))
        checks["empty ascending unique"] = (se == sorted(set(se)))
        checks["present subset of present-stratum"] = set(sp) <= set(present)
        checks["empty subset of empty-stratum"] = set(se) <= set(empty)
        checks["present/empty disjoint"] = not (set(sp) & set(se))
        merged = sorted(sp + se)
        checks["merged unique"] = (len(merged) == len(set(merged)))
        # frame_path resolves for every selected index
        try:
            for i in merged:
                self.ann.frame_path(video_name, i)
            checks["all selected resolve to a frame_path"] = True
        except Exception:
            checks["all selected resolve to a frame_path"] = False

        if verbose:
            print("=" * 68)
            print(f"TRACE {video_name}  pct={self.pct}  frames={n_total}")
            print(f"  present stratum n={n_present} -> k={kp} -> {sp}")
            print(f"  empty   stratum n={n_empty} -> k={ke} -> {se}")
            print(f"  independent ref: present {rsp}  empty {rse}")
            print(f"  merged ({len(merged)}): {merged}")
            print("  checks:")
            for name, ok in checks.items():
                print(f"    [{'OK' if ok else 'XX'}] {name}")

        failed = [n for n, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"{video_name}: trace checks failed: {failed}")

        return {
            "video": video_name, "frames": n_total,
            "present_n": n_present, "empty_n": n_empty,
            "present_k": kp, "empty_k": ke,
            "present": sp, "empty": se, "merged": merged,
            "checks": checks, "ok": True,
        }

    def explain_video(self, video_name):
        """Readable step-by-step narration of ONE video's extraction.
        Uses the real _sample_video, so what it shows is exactly what build()
        selects. For understanding/presenting (see trace_video for self-checking)."""
        present, empty, labels = self._strata(video_name)
        n_total = len(labels)
        sel_present, sel_empty, n_present, n_empty = self._sample_video(video_name)

        print("=" * 64)
        print(f"VIDEO: {video_name}   pct={self.pct}")
        print("=" * 64)

        print(f"\nSTEP 1 — labels (1=animal 0=empty), {n_total} frames")
        preview = "".join(str(l) for l in labels[:60])
        print(f"  {preview}{' ...' if n_total > 60 else ''}")

        print("\nSTEP 2 — split into two strata")
        print(f"  PRESENT (1): n={n_present}  {present}")
        print(f"  EMPTY   (0): n={n_empty}  {empty}")

        print("\nSTEP 3/4 — k = ceil(pct*n), then even-spaced pick")
        self._explain_pick("PRESENT", n_present, sel_present)
        self._explain_pick("EMPTY", n_empty, sel_empty)

        merged = sorted(sel_present + sel_empty)
        print("\nSTEP 5 — merge + sort")
        print(f"  present picks: {sel_present}")
        print(f"  empty   picks: {sel_empty}")
        print(f"  -> SELECTED {len(merged)}/{n_total} frames: {merged}")

        print("\nMANIFEST rows this video contributes:")
        for i in sel_present:
            print(f"  {self.ann.frame_path(video_name, i)},1")
        for i in sel_empty:
            print(f"  {self.ann.frame_path(video_name, i)},0")
        print("=" * 64)

    def _explain_pick(self, name, n, picks):
        if n == 0:
            print(f"  {name}: empty stratum -> nothing")
            return
        k = ceil(self.pct * n)
        note = " (k>=n -> all)" if k >= n else (" (k==1 -> middle)" if k == 1 else "")
        print(f"  {name}: k=ceil({self.pct}*{n})=ceil({self.pct*n:.2f})={k}{note}")
        print(f"       -> {picks}")

# ---------- module-level helper: decode check ----------

def _is_decodable_jpeg(path):
    """True if the file looks like a complete, decodable JPEG.
    Uses cv2 when available (authoritative); otherwise checks the SOI/EOI
    markers (FFD8 ... FFD9), which catches truncated downloads."""
    try:
        import cv2
        img = cv2.imread(str(path))
        return img is not None
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            f.seek(-2, 2)
            tail = f.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except Exception:
        return False