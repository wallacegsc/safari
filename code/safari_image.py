from safari_tools import SaFariAnnotation
from pathlib import Path 
import cv2

class SaFariImage:
    def __init__(self, s:SaFariAnnotation, jpeg_6fps_dir:str):
        self.s = s
        self.dir = Path(jpeg_6fps_dir)

    def _frame_local_path(self, video_name, frame_idx):
        return self.dir / self.s.frame_path(video_name, frame_idx)

    def load_frame(self, video_name, frame_idx):
        frame_path = self._frame_local_path(video_name, frame_idx)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Frame not found: {frame_path}")
        return frame

    def draw_boxes(self, video_name, frame_idx):
        frame = self.load_frame(video_name, frame_idx)
        for obj in self.s.annotations_at_frame(video_name, frame_idx):
            x, y, w, h = map(int, obj["bbox"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        return frame
