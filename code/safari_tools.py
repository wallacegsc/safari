from pathlib import Path
import json

class SaFariAnnotation:
    def __init__(self, d):
        self.d = d
        self._cat_name = {c["id"]: c["name"] for c in d["categories"]}
        self._video_by_name = {v["video_name"]: v for v in d["videos"]}
        self._video_by_id = {v["id"]: v for v in d["videos"]}

        self._anns_by_video = {}
        for a in d["annotations"]:
            self._anns_by_video.setdefault(a["video_id"], []).append(a)

    def dataset_info(self):
        info = self.d['info']
        print(info)

    def video_info(self, video_name):
        video_name = video_name.strip()
        try:
            return self._video_by_name[video_name]
        except KeyError:
            raise KeyError(f"video_name not found: {video_name!r}")

    def video_info_id(self, video_id):
        try:
            return self._video_by_id[video_id]
        except KeyError:
            raise KeyError(f"video id not found: {video_id!r}")

    def annotations_info(self, video_name, video_id=-1):
        if video_id == -1:
            v_id = self._video_by_name[video_name]["id"]  # KeyError if bad name
        else:
            v_id = video_id
        return self._anns_by_video.get(v_id, [])

    def video_names(self):
        return list(self._video_by_name.keys())

    def category_id_to_name(self, cat_id) -> str:
        return self._cat_name.get(cat_id)

    def animal_presence_frame_label(self, video_name, video_id = -1):
        presence = []
        a_info = self.annotations_info(video_name, video_id)
        if not a_info:
            v_info = self.video_info_id(video_id) if video_id != -1 else self.video_info(video_name)

            for i in range(v_info['length']):
                presence.append(0)
            return presence

        an_count = len(a_info)
        frame_count = len(a_info[0]["bboxes"])

        for i in range(frame_count):
            is_present = 0
            for animal in range(an_count):
                if a_info[animal]["bboxes"][i] is not None:
                    is_present = 1
                    break
            presence.append(is_present)
        
        return presence

    def presence_vs_empty_counts(self):
        present = 0
        empty = 0
        for v in self.d["videos"]:
            labels = self.animal_presence_frame_label(v["video_name"])
            for x in labels:
                if x == 1:
                    present += 1
                else:
                    empty += 1
        return {"present": present, "empty": empty}

    def annotations_at_frame(self, video_name, frame_idx, video_id = -1):
        a_info = self.annotations_info(video_name, video_id)
        if not a_info:
            return []

        frames_count = len(a_info[0]["bboxes"])

        if frame_idx >= frames_count:
            raise IndexError(f"frame {frame_idx} out of range (max {frames_count - 1})")

        an_count = len(a_info)

        frame_items = []
        for animal in range(an_count):
            bbox = a_info[animal]["bboxes"][frame_idx]
            if bbox is not None:
                frame_items.append({
                    "bbox": bbox,
                    "category_id": a_info[animal]["category_id"],
                })
        return frame_items

    def annotations_at_all_frames(self, video_name, video_id = -1):
        a_info = self.annotations_info(video_name, video_id)
        if not a_info:
            return []

        frame_count = len(a_info[0]["bboxes"])

        frames = []
        for frame_idx in range(frame_count):
            frame_items = []
            for a in a_info:
                bbox = a["bboxes"][frame_idx]
                if bbox is not None:
                    frame_items.append({
                        "bbox": bbox,
                        "category_id": a["category_id"],
                    })
            frames.append(frame_items)
        return frames

    def frame_path(self, video_name, frame_idx):
        v_info = self.video_info(video_name)
        frames_count = len(v_info["file_names"])
        if frame_idx >= frames_count:
            raise IndexError(f"frame {frame_idx} out of range (max {frames_count - 1})")
        return v_info["file_names"][frame_idx]

    def summary(self):
        total_videos = len(self.d["videos"])
        total_annotations = len(self.d["annotations"])
        vocab_size = len(self.d["categories"])

        annotated_ids = set()
        used_categories = set()
        for a in self.d["annotations"]:
            annotated_ids.add(a["video_id"])
            used_categories.add(a["category_id"])

        zero_animal = 0
        resolutions = {}
        for v in self.d["videos"]:
            if v["id"] not in annotated_ids:
                zero_animal += 1
            res = f"{v['width']}x{v['height']}"
            resolutions[res] = resolutions.get(res, 0) + 1

        print(f"videos:          {total_videos}")
        print(f"annotations:     {total_annotations}")
        print(f"category vocab:  {vocab_size}")
        print(f"used categories: {len(used_categories)}")
        print(f"zero-animal:     {zero_animal}")
        print(f"resolutions:     {resolutions}")
    
    def length_stats(self):
        distribution = {}
        total = 0
        count = 0
        min_len = None
        max_len = None

        for v in self.d["videos"]:
            n = v["length"]
            distribution[n] = distribution.get(n, 0) + 1
            total += n
            count += 1
            if min_len is None or n < min_len:
                min_len = n
            if max_len is None or n > max_len:
                max_len = n

        mean = total / count if count else 0

        return {
            "min": min_len,
            "max": max_len,
            "mean": mean,
            "total": total,
            "videos": count,
            "distribution": distribution,
        }

    def category_counts(self):
        counts = {}
        for a in self.d["annotations"]:
            name = self.category_id_to_name(a["category_id"])
            counts[name] = counts.get(name, 0) + 1
        return counts

    def videos_with_category(self, name_or_id):
        if isinstance(name_or_id, str):
            cat_id = None
            for c in self.d["categories"]:
                if c["name"] == name_or_id:
                    cat_id = c["id"]
                    break
            if cat_id is None:
                raise KeyError(f"category not found: {name_or_id!r}")
        else:
            cat_id = name_or_id

        video_ids = set()
        for a in self.d["annotations"]:
            if a["category_id"] == cat_id:
                video_ids.add(a["video_id"])

        names = []
        for v in self.d["videos"]:
            if v["id"] in video_ids:
                names.append(v["video_name"])
        return names

    def empty_videos(self):
        names = []
        for v in self.d["videos"]:
            if v["id"] not in self._anns_by_video:
                names.append(v["video_name"])
        return names

    def print_video_names(self):
        list_names = self.video_names()
        print(list_names)

    def print_video_info(self, video_name):
        info = self.video_info(video_name)
        print(f"ID:    {info['id']}")
        print(f"FN:    {info['file_names'][0]} ... {info['file_names'][-1]}")
        print(f"W x H: {info['width']}x{info['height']}")
        print(f"len:   {info['length']}")

    def print_annotations_info(self, video_name):
        a_info = self.annotations_info(video_name)
        if not a_info:
            print("no annotations for this video")
            return

        print(f"vID:   {a_info[0]['video_id']}")
        print(f"W x H: {a_info[0]['width']}x{a_info[0]['height']}")
        print(f"Animals: {len(a_info)}")

        groups = {}
        for a in a_info:
            name = self.category_id_to_name(a['category_id'])
            groups.setdefault(name, []).append(a)

        i = 1
        for name, animals in groups.items():
            n = len(animals)
            print(f"Animal {i}-{i+n-1}: {name}")
            for a in animals:
                print(f"    id={a['id']} iscrowd={a['iscrowd']} segs={len(a['segmentations'])} bboxes={len(a['bboxes'])}")
            i += n

    def print_length_stats(self):
        stats = self.length_stats()
        print(f"videos: {stats['videos']}")
        print(f"min:    {stats['min']}")
        print(f"max:    {stats['max']}")
        print(f"mean:   {stats['mean']:.1f}")
        print(f"total:  {stats['total']}")

        buckets = {}
        for length, cnt in stats["distribution"].items():
            bucket = (length // 50) * 50
            buckets[bucket] = buckets.get(bucket, 0) + cnt

        print("distribution (length range: count):")
        for bucket in sorted(buckets):
            print(f"    {bucket}-{bucket + 49}: {buckets[bucket]}")

    def frame_max_area_frac(self, video_name, frame_idx):
        """Largest animal bbox area / frame area at one frame. 0.0 if no animal.
        bbox is [x, y, w, h]; frame area uses the video's width x height."""
        v = self.video_info(video_name)
        frame_area = v["width"] * v["height"]
        best = 0.0
        for a in self._anns_by_video.get(v["id"], []):
            b = a["bboxes"][frame_idx]
            if b is not None:
                best = max(best, (b[2] * b[3]) / frame_area)
        return best

    def __len__(self):
        return len(self.d["videos"])
        
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

