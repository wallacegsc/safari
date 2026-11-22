from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from google.cloud import storage

BUCKET = "cxl-public-camera-trap"

class SaFariDownloader:
    def __init__(self, annotation, split, output_dir, max_workers=32, logs_dir="logs"):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.ann = annotation
        self.split = split
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers

        # local + bucket both follow: sa_fari/sa_fari_<split>/JPEGImages_6fps/<video>/<frame>.jpg
        self.subpath = f"sa_fari/sa_fari_{split}/JPEGImages_6fps"

        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(BUCKET)

        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(f"safari_download_{split}")
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            fh = logging.FileHandler(Path(logs_dir) / f"download_{split}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.log.addHandler(fh)

    def _blob_path(self, rel):
        return f"{self.subpath}/{rel}"

    def _local_path(self, rel):
        return self.output_dir / self.subpath / rel

    def _download_one(self, rel, force):
        local = self._local_path(rel)
        if local.exists() and not force:
            return "skipped"
        local.parent.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.blob(self._blob_path(rel))
        blob.download_to_filename(str(local))
        return "downloaded"

    def download_video(self, video_name, frames=None, force=False):
        v = self.ann.video_info(video_name)
        file_names = v["file_names"]

        if frames is None:
            indices = range(len(file_names))
        else:
            indices = frames

        # resolve indices -> relative paths (raises IndexError on bad index)
        rels = [self.ann.frame_path(video_name, i) for i in indices]

        downloaded = 0
        skipped = 0
        failed = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._download_one, rel, force): rel for rel in rels}
            for fut in as_completed(futures):
                rel = futures[fut]
                try:
                    result = fut.result()
                    if result == "downloaded":
                        downloaded += 1
                    else:
                        skipped += 1
                except Exception as e:
                    failed.append(rel)
                    self.log.info(f"FAILED {rel}: {e}")

        summary = {"downloaded": downloaded, "skipped": skipped, "failed": len(failed)}
        line = f"{video_name}: {downloaded} downloaded, {skipped} skipped, {len(failed)} failed"
        print(line)
        self.log.info(line)
        return summary
    
    def download_videos(self, plan, force=False):
        totals = {"downloaded": 0, "skipped": 0, "failed": 0}
        per_video = {}

        for video_name, frames in plan.items():
            try:
                s = self.download_video(video_name, frames=frames, force=force)
            except Exception as e:
                line = f"{video_name}: ERROR {e}"
                print(line)
                self.log.info(line)
                per_video[video_name] = {"downloaded": 0, "skipped": 0, "failed": 0, "error": str(e)}
                continue
            per_video[video_name] = s
            for k in totals:
                totals[k] += s[k]

        line = (f"TOTAL {len(plan)} videos: "
                f"{totals['downloaded']} downloaded, "
                f"{totals['skipped']} skipped, "
                f"{totals['failed']} failed")
        print(line)
        self.log.info(line)
        return {"totals": totals, "per_video": per_video}
