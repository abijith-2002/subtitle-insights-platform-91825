anaimport os
import cv2
import numpy as np
import pysrt
import re
from rapidocr_onnxruntime import RapidOCR
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
log = logging.getLogger()

# Shared frame counter for visibility in logs
a = 0  # frame counter

# Global RapidOCR instance (initialized once, shared across all threads)
_GLOBAL_RAPIDOCR = None
_GLOBAL_RAPIDOCR_INIT_LOCK = threading.Lock()

# ------------------------------------------------------------
# Placement abstractions
# ------------------------------------------------------------

class PlacementKind(Enum):
    """Defines the type of subtitle placement to be applied."""
    ANCHOR = "anchor"       # Fixed pre-defined anchors (corners, edges, center)
    GRID = "grid"           # Grid cell based placement (rows x cols)
    ABSOLUTE = "absolute"   # Absolute (pixel or normalized) position with optional alignment
    REGION = "region"       # Place inside a rectangular region

class Anchor(Enum):
    """Standard 3x3 anchor positions matching ASS alignment semantics."""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    MIDDLE_CENTER = "middle_center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"

@dataclass
class Region:
    """Rectangle region where subtitles can be placed.

    The coordinates can be absolute pixels or normalized [0..1] based on the 'relative' flag.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    relative: bool = True  # True means values are normalized [0..1]

@dataclass
class Placement:
    """Unified placement descriptor used by all formats.

    kind:
      - ANCHOR: use 'anchor'
      - GRID: use (grid_rows, grid_cols, grid_r, grid_c)
      - ABSOLUTE: use (x, y) with relative flag
      - REGION: use Region; optionally combine with anchor to pick a spot in the region

    The 'anchor' can also serve as alignment preference for ABSOLUTE/REGION placements.
    """
    kind: PlacementKind
    anchor: Optional[Anchor] = None

    # ABSOLUTE
    x: Optional[float] = None
    y: Optional[float] = None
    relative_xy: bool = True  # Only relevant for ABSOLUTE

    # GRID
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None
    grid_r: Optional[int] = None
    grid_c: Optional[int] = None

    # REGION
    region: Optional[Region] = None

    # Optional: estimated lines and size hints (for bbox estimation and overlap scoring)
    text_lines: int = 2
    width_fraction_hint: Optional[float] = None  # If None, defaults used
    height_fraction_hint: Optional[float] = None

# ------------------------------------------------------------
# Precompiled regex patterns
# ------------------------------------------------------------
ASS_TIME_RE = re.compile(r"Dialogue: \d+,(.*?),(.*?),")
ASS_TEXT_RE = re.compile(
    r"Dialogue: \d+,[0-9]:[0-9]{2}:[0-9]{2}\.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}\.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,(?:.*)?,(?:\{\\an\d\})?(.*)"
)
SSA_TIME_RE = re.compile(r"Dialogue: Marked=\d+,(.*?),(.*?),")
SSA_TEXT_RE = re.compile(
    r"Dialogue: Marked=\d+,[0-9]:[0-9]{2}:[0-9]{2}\.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}\.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,,(?:\{\\an\d\})?(.*)"
)
AN_TAG_RE = re.compile(r"\{\\an\d\}")
POS_TAG_RE = re.compile(r"\{\\pos\([^)]+\)\}")
MOVE_TAG_RE = re.compile(r"\{\\move\([^)]+\)\}")

VTT_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})")
VTT_LINE_POS_RE = re.compile(r"line:\d+%?")
VTT_POSITION_RE = re.compile(r"position:\d+%")
VTT_ALIGN_RE = re.compile(r"align:(start|center|end|left|right)")

# SRT styling tags -> ASS overrides (compiled with IGNORECASE)
SRT_TAG_REPLACEMENTS = [
    (re.compile(r"<i>", re.IGNORECASE), r"{\\\\i1}"),
    (re.compile(r"</i>", re.IGNORECASE), r"{\\\\i0}"),
    (re.compile(r"<b>", re.IGNORECASE), r"{\\\\b1}"),
    (re.compile(r"</b>", re.IGNORECASE), r"{\\\\b0}"),
    (re.compile(r"<u>", re.IGNORECASE), r"{\\\\u1}"),
    (re.compile(r"</u>", re.IGNORECASE), r"{\\\\u0}"),
    (re.compile(r"<s>", re.IGNORECASE), r"{\\\\s1}"),
    (re.compile(r"</s>", re.IGNORECASE), r"{\\\\s0}"),
]

# ------------------------------------------------------------
# OCR Engine
# ------------------------------------------------------------

def _get_global_ocr_engine() -> RapidOCR:
    """Return the single global RapidOCR instance, initializing it once lazily.

    This ensures only one model is loaded in memory and reused by all threads,
    avoiding per-thread/per-process instantiation and memory duplication.
    """
    global _GLOBAL_RAPIDOCR
    if _GLOBAL_RAPIDOCR is None:
        with _GLOBAL_RAPIDOCR_INIT_LOCK:
            if _GLOBAL_RAPIDOCR is None:
                log.debug("Initializing global RapidOCR instance (once)")
                try:
                    _GLOBAL_RAPIDOCR = RapidOCR()
                except Exception as e:
                    log.error("RapidOCR initialization failed: %s", e, exc_info=True)
                    raise
    return _GLOBAL_RAPIDOCR


class VideoReader:
    """Thread-safe video reader that opens a single VideoCapture and reuses it.

    This class wraps cv2.VideoCapture with a lock, allowing safe random access
    to frames by index from multiple threads (seek + read guarded by a lock).
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._lock = threading.Lock()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Cache properties for reuse
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1e-6 else 25.0  # reasonable fallback
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame_at(self, frame_idx: int):
        """Random-access read: seek to frame_idx and read a frame under lock."""
        with self._lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = self.cap.read()
            return ret, frame

    def release(self):
        """Release underlying VideoCapture."""
        with self._lock:
            try:
                self.cap.release()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def detect_using_rapidocr(img):
    """Run OCR using the shared global RapidOCR instance and structure detections."""
    global a
    if log.isEnabledFor(logging.DEBUG):
        log.debug("frame_counter=%s", a)

    engine = _get_global_ocr_engine()
    results, _ = engine(img)  # results = [(box, text, score), ...]

    detections = []
    if results:
        log.info("ocr_detections_found")
        log.info(f"results : {results}")
        for box, text, score in results:
            if score>0.7:
                detections.append({"box": box, "text": text, "score": float(score)})

    a += 1
    return detections

# ------------------------------------------------------------
# Format helpers and conversion
# ------------------------------------------------------------

def to_ass_timestamp(srt_time):
    """Convert pysrt.SubRipTime to ASS H:MM:SS.CC format."""
    total_ms = (
        srt_time.hours * 3600 * 1000
        + srt_time.minutes * 60 * 1000
        + srt_time.seconds * 1000
        + srt_time.milliseconds
    )
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1_000
    centiseconds = (total_ms % 1_000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

def anchor_to_ass_alignment(anchor: Anchor) -> int:
    """Map Anchor to ASS alignment code (an1..an9)."""
    mapping = {
        Anchor.TOP_LEFT: 7, Anchor.TOP_CENTER: 8, Anchor.TOP_RIGHT: 9,
        Anchor.MIDDLE_LEFT: 4, Anchor.MIDDLE_CENTER: 5, Anchor.MIDDLE_RIGHT: 6,
        Anchor.BOTTOM_LEFT: 1, Anchor.BOTTOM_CENTER: 2, Anchor.BOTTOM_RIGHT: 3,
    }
    return mapping[anchor]

def anchor_to_vtt_settings(anchor: Anchor) -> Tuple[int, int, str]:
    """Map Anchor to VTT settings: (line_percent, position_percent, align)."""
    # Define percent grid near edges/center
    row_line = {
        Anchor.TOP_LEFT: 5, Anchor.TOP_CENTER: 5, Anchor.TOP_RIGHT: 5,
        Anchor.MIDDLE_LEFT: 50, Anchor.MIDDLE_CENTER: 50, Anchor.MIDDLE_RIGHT: 50,
        Anchor.BOTTOM_LEFT: 95, Anchor.BOTTOM_CENTER: 95, Anchor.BOTTOM_RIGHT: 95,
    }
    col_pos = {
        Anchor.TOP_LEFT: 5, Anchor.MIDDLE_LEFT: 5, Anchor.BOTTOM_LEFT: 5,
        Anchor.TOP_CENTER: 50, Anchor.MIDDLE_CENTER: 50, Anchor.BOTTOM_CENTER: 50,
        Anchor.TOP_RIGHT: 95, Anchor.MIDDLE_RIGHT: 95, Anchor.BOTTOM_RIGHT: 95,
    }
    align = {
        Anchor.TOP_LEFT: "start", Anchor.MIDDLE_LEFT: "start", Anchor.BOTTOM_LEFT: "start",
        Anchor.TOP_CENTER: "center", Anchor.MIDDLE_CENTER: "center", Anchor.BOTTOM_CENTER: "center",
        Anchor.TOP_RIGHT: "end", Anchor.MIDDLE_RIGHT: "end", Anchor.BOTTOM_RIGHT: "end",
    }
    return row_line[anchor], col_pos[anchor], align[anchor]

def _safe_percent(val: float) -> int:
    """Clamp a percentage to 0..100 and cast to int."""
    return int(max(0, min(100, round(val))))

def ass_inline_for_placement(placement: Placement, frame_w: int, frame_h: int) -> str:
    """Create ASS inline override for placement.

    Returns strings like:
      - "{\\anN}" for anchor/grid placement
      - "{\\anN}{\\pos(x,y)}" for absolute/region placement (with integer pixel coordinates)
    Curly braces are doubled to emit literal braces inside f-strings.
    """
    # Helper to get alignment code safely
    def _align_code(p: Placement) -> int:
        return anchor_to_ass_alignment(p.anchor or Anchor.MIDDLE_CENTER)

    if placement.kind in (PlacementKind.ANCHOR, PlacementKind.GRID):
        code = _align_code(placement)
        return f"{{\\an{code}}}"

    elif placement.kind == PlacementKind.ABSOLUTE:
        code = _align_code(placement)

        # Fallback to alignment only if coordinates are missing
        if placement.x is None or placement.y is None:
            return f"{{\\an{code}}}"

        # Compute pixel coordinates
        if placement.relative_xy:
            x = int(placement.x * frame_w)
            y = int(placement.y * frame_h)
        else:
            x = int(placement.x)
            y = int(placement.y)

        # Include both alignment and position
        return f"{{\\an{code}}}{{\\pos({x},{y})}}"

    elif placement.kind == PlacementKind.REGION and placement.region is not None:
        # Use region center as absolute position
        reg = placement.region
        x0 = reg.x0 * frame_w if reg.relative else reg.x0
        y0 = reg.y0 * frame_h if reg.relative else reg.y0
        x1 = reg.x1 * frame_w if reg.relative else reg.x1
        y1 = reg.y1 * frame_h if reg.relative else reg.y1
        cx = int((x0 + x1) / 2.0)
        cy = int((y0 + y1) / 2.0)

        code = _align_code(placement)
        return f"{{\\an{code}}}{{\\pos({cx},{cy})}}"

    else:
        code = _align_code(placement)
        return f"{{\\an{code}}}"

def vtt_settings_for_placement(placement: Placement, frame_w: int, frame_h: int) -> Dict[str, str]:
    """Create VTT cue settings (line, position, align) for a placement."""
    if placement.kind in (PlacementKind.ANCHOR, PlacementKind.GRID):
        anchor = placement.anchor or Anchor.MIDDLE_CENTER
        line_pct, pos_pct, align = anchor_to_vtt_settings(anchor)
        return {
            "line": f"line:{line_pct}%",
            "position": f"position:{pos_pct}%",
            "align": f"align:{align}",
        }
    elif placement.kind == PlacementKind.ABSOLUTE:
        if placement.x is None or placement.y is None:
            # Fall back to center
            line_pct, pos_pct, align = anchor_to_vtt_settings(placement.anchor or Anchor.MIDDLE_CENTER)
            return {
                "line": f"line:{line_pct}%",
                "position": f"position:{pos_pct}%",
                "align": f"align:{align}",
            }
        if placement.relative_xy:
            x_pct = _safe_percent(placement.x * 100.0)
            y_pct = _safe_percent(placement.y * 100.0)
        else:
            x_pct = _safe_percent(placement.x / frame_w * 100.0)
            y_pct = _safe_percent(placement.y / frame_h * 100.0)
        # In WebVTT, 'line' controls vertical, 'position' controls horizontal
        align = placement.anchor or Anchor.MIDDLE_CENTER
        _, _, align_val = anchor_to_vtt_settings(align)
        return {
            "line": f"line:{y_pct}%",
            "position": f"position:{x_pct}%",
            "align": f"align:{align_val}",
        }
    elif placement.kind == PlacementKind.REGION and placement.region is not None:
        reg = placement.region
        # Use region center for VTT line/position
        cx = ((reg.x0 + reg.x1) / 2.0) if reg.relative else ((reg.x0 + reg.x1) / 2.0 / frame_w)
        cy = ((reg.y0 + reg.y1) / 2.0) if reg.relative else ((reg.y0 + reg.y1) / 2.0 / frame_h)
        x_pct = _safe_percent(cx * 100.0)
        y_pct = _safe_percent(cy * 100.0)
        align_anchor = placement.anchor or Anchor.MIDDLE_CENTER
        _, _, align_val = anchor_to_vtt_settings(align_anchor)
        return {
            "line": f"line:{y_pct}%",
            "position": f"position:{x_pct}%",
            "align": f"align:{align_val}",
        }
    else:
        line_pct, pos_pct, align = anchor_to_vtt_settings(Anchor.MIDDLE_CENTER)
        return {
            "line": f"line:{line_pct}%",
            "position": f"position:{pos_pct}%",
            "align": f"align:{align}",
        }

# ------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------

def preprocess_threshold(image):
    """Prepare image for OCR using simple threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def preprocess_using_blackout(image, block_size=50, contrast_thresh=0.3, blur_thresh=100.0):
    """Black out low-contrast or blurry regions."""
    output = image.copy()
    h, w = image.shape[:2]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y: y + block_size, x: x + block_size]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            I_max, I_min = np.max(gray), np.min(gray)
            contrast = (I_max - I_min) / (I_max + I_min + 1e-5)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if contrast < contrast_thresh or variance < blur_thresh:
                output[y: y + block_size, x: x + block_size] = (0, 0, 0)

    return output

def preprocess(
    image,
    block_size=50,
    contrast_thresh=0.3,
    blur_thresh=100.0,
    adaptive_block=15,
    adaptive_C=5,
):
    """Blackout low-quality regions and apply adaptive threshold."""
    output = image.copy()
    h, w = image.shape[:2]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y: y + block_size, x: x + block_size]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            I_max, I_min = np.max(gray), np.min(gray)
            contrast = (I_max - I_min) / (I_max + I_min + 1e-5)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if contrast < contrast_thresh or variance < blur_thresh:
                output[y: y + block_size, x: x + block_size] = (0, 0, 0)

    gray_full = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block, adaptive_C
    )
    return binary

def preprocess_adaptive_threshold(image):
    """Prepare image for OCR using adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=5
    )
    return thresh

# ------------------------------------------------------------
# Geometry and scoring helpers
# ------------------------------------------------------------

Rect = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

def poly_to_aabb(poly: List[List[float]]) -> Rect:
    """Convert a 4-point polygon [[x,y],...] to axis-aligned bounding box."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

def rect_area(r: Rect) -> float:
    x1, y1, x2, y2 = r
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersection_area(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return rect_area((x1, y1, x2, y2))

def ocr_detections_to_rects(dets: List[Dict[str, Any]]) -> List[Rect]:
    rects = []
    for d in dets or []:
        try:
            rects.append(poly_to_aabb(d["box"]))
        except Exception:
            continue
    return rects

def estimate_subtitle_rect_for_anchor(
    frame_w: int, frame_h: int, anchor: Anchor, text_lines: int = 2,
    width_fraction_hint: Optional[float] = None, height_fraction_hint: Optional[float] = None
) -> Rect:
    """Estimate a plausible subtitle rectangle for a given anchor.

    Heuristics: center anchors use ~80% width; corners ~50% width. Height depends on line count.
    """
    # Heuristics
    if width_fraction_hint is not None:
        width_frac = width_fraction_hint
    else:
        if anchor in (Anchor.TOP_CENTER, Anchor.MIDDLE_CENTER, Anchor.BOTTOM_CENTER):
            width_frac = 0.80
        else:
            width_frac = 0.50

    if height_fraction_hint is not None:
        height_frac = height_fraction_hint
    else:
        # ~5% height per line + small padding
        height_frac = min(0.25, 0.05 * text_lines + 0.02)

    w = frame_w * width_frac
    h = frame_h * height_frac

    # Margins from edges
    margin_x = frame_w * 0.05
    margin_y = frame_h * 0.05

    if anchor in (Anchor.TOP_LEFT, Anchor.TOP_CENTER, Anchor.TOP_RIGHT):
        top = margin_y
    elif anchor in (Anchor.MIDDLE_LEFT, Anchor.MIDDLE_CENTER, Anchor.MIDDLE_RIGHT):
        top = (frame_h - h) / 2.0
    else:
        top = frame_h - h - margin_y

    if anchor in (Anchor.TOP_LEFT, Anchor.MIDDLE_LEFT, Anchor.BOTTOM_LEFT):
        left = margin_x
    elif anchor in (Anchor.TOP_CENTER, Anchor.MIDDLE_CENTER, Anchor.BOTTOM_CENTER):
        left = (frame_w - w) / 2.0
    else:
        left = frame_w - w - margin_x

    return (left, top, left + w, top + h)

def estimate_subtitle_rect_for_placement(
    placement: Placement, frame_w: int, frame_h: int
) -> Rect:
    """Estimate a subtitle rectangle (axis-aligned) for a placement."""
    if placement.kind in (PlacementKind.ANCHOR, PlacementKind.GRID) and placement.anchor:
        return estimate_subtitle_rect_for_anchor(
            frame_w, frame_h, placement.anchor, placement.text_lines,
            placement.width_fraction_hint, placement.height_fraction_hint
        )
    elif placement.kind == PlacementKind.ABSOLUTE:
        # Use a small box around the (x,y) position
        if placement.x is None or placement.y is None:
            return estimate_subtitle_rect_for_anchor(frame_w, frame_h, Anchor.MIDDLE_CENTER, placement.text_lines)
        x = placement.x * frame_w if placement.relative_xy else placement.x
        y = placement.y * frame_h if placement.relative_xy else placement.y
        width_frac = placement.width_fraction_hint or 0.60
        height_frac = placement.height_fraction_hint or min(0.20, 0.05 * placement.text_lines + 0.02)
        w = frame_w * width_frac
        h = frame_h * height_frac
        left = max(0.0, x - w / 2.0)
        top = max(0.0, y - h / 2.0)
        return (left, top, min(frame_w, left + w), min(frame_h, top + h))
    elif placement.kind == PlacementKind.REGION and placement.region is not None:
        reg = placement.region
        x0 = reg.x0 * frame_w if reg.relative else reg.x0
        y0 = reg.y0 * frame_h if reg.relative else reg.y0
        x1 = reg.x1 * frame_w if reg.relative else reg.x1
        y1 = reg.y1 * frame_h if reg.relative else reg.y1
        # Use a box that fits well within the region (90% of region area centered)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rw = (x1 - x0) * 0.9
        rh = (y1 - y0) * 0.9
        return (cx - rw / 2.0, cy - rh / 2.0, cx + rw / 2.0, cy + rh / 2.0)
    else:
        return estimate_subtitle_rect_for_anchor(frame_w, frame_h, Anchor.MIDDLE_CENTER, placement.text_lines)

def occupancy_score_for_rect(candidate: Rect, ocr_rects: List[Rect]) -> float:
    """Compute normalized overlap of candidate rect with OCR rects (0=free, high=busy)."""
    if rect_area(candidate) <= 0.0:
        return 1.0
    inter = 0.0
    for r in ocr_rects:
        inter += intersection_area(candidate, r)
    return inter / rect_area(candidate)

def default_anchor_preference() -> List[Anchor]:
    """Preferred anchors order if multiple are similarly free."""
    # Favor bottom center first (default), then other bottom corners, then top side, then middle
    return [
        Anchor.BOTTOM_CENTER, Anchor.TOP_CENTER,
        Anchor.BOTTOM_RIGHT, Anchor.BOTTOM_LEFT, Anchor.TOP_RIGHT, Anchor.TOP_LEFT,
        Anchor.MIDDLE_CENTER, Anchor.MIDDLE_RIGHT, Anchor.MIDDLE_LEFT
    ]

def decide_placement_from_detections(
    filtered_detections_list: List[List[Dict[str, Any]]],
    frame_width: int,
    frame_height: int,
    allowed_anchors: Optional[List[Anchor]] = None,
    text_lines_estimate: int = 2
) -> Placement:
    """Decide a subtitle Placement based on OCR detections occupancy.

    Strategy:
      - For each allowed anchor, estimate a subtitle rect and compute average occupancy across frames.
      - Choose the anchor with minimal occupancy. Break ties by preference order.
    """
    if not allowed_anchors:
        allowed_anchors = list(Anchor)  # all

    ocr_rects_per_frame: List[List[Rect]] = [ocr_detections_to_rects(dets) for dets in filtered_detections_list]
    avg_scores: Dict[Anchor, float] = {}

    for anchor in allowed_anchors:
        cand = estimate_subtitle_rect_for_anchor(frame_width, frame_height, anchor, text_lines_estimate)
        scores = []
        for frame_rects in ocr_rects_per_frame:
            s = occupancy_score_for_rect(cand, frame_rects)
            scores.append(s)
        avg_scores[anchor] = (sum(scores) / max(1, len(scores))) if scores else 0.0

    # sort by score then preference
    pref = default_anchor_preference()
    pref_rank = {a: i for i, a in enumerate(pref)}
    best_anchor = min(avg_scores.keys(), key=lambda a: (avg_scores[a], pref_rank.get(a, 999)))
    log.debug("anchor_scores=%s best_anchor=%s", avg_scores, best_anchor)

    return Placement(kind=PlacementKind.ANCHOR, anchor=best_anchor, text_lines=text_lines_estimate)

def decide_subtitle_position(filtered_detections_list, frame_height, bottom_threshold_ratio=0.75):
    """Return 'top' if burnt-in text detected in bottom region; else 'bottom'.

    Backward-compatible simple decision used when only top/bottom is needed.
    """
    for frame_detections in filtered_detections_list:
        if frame_detections:
            for det in frame_detections:
                y_coords = [p[1] for p in det["box"]]
                avg_y = sum(y_coords) / len(y_coords)
                if avg_y > frame_height * bottom_threshold_ratio:
                    return "top"
    return "bottom"

# ------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------

# PUBLIC_INTERFACE
def get_position_for_segment(video_reader: VideoReader, start_sec, end_sec, min_frames=3):
    """Decide top/bottom for a segment using a shared VideoReader.

    Parameters:
    - video_reader: Shared VideoReader wrapping a single cv2.VideoCapture.
    - start_sec: Segment start time in seconds.
    - end_sec: Segment end time in seconds.
    - min_frames: Minimum frames to sample within the segment.

    Returns:
    - 'top' or 'bottom' based on OCR detections in the bottom region.
    """
    log.info("segment_position_eval start=%s end=%s", start_sec, end_sec)
    fps = video_reader.fps
    frame_height = int(video_reader.frame_height)
    frame_width = int(video_reader.frame_width)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    sub_time = end_sec - start_sec
    required_min_frames = int(sub_time) // 2
    min_frames = max(min_frames, required_min_frames)
    log.info("min_frames_required=%s selected_min_frames=%s", required_min_frames, min_frames)

    frame_indices = np.linspace(
        start_frame, end_frame, min(min_frames, abs(end_frame - start_frame + 1)), dtype=int
    )

    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        ret, frame = video_reader.read_frame_at(frame_idx)
        if not ret:
            log.warning("frame_read_failed index=%s", frame_idx)
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)

    # Decide using bottom-first heuristic: default 'bottom' unless bottom region has burnt-in text
    rec_pos = decide_subtitle_position(filtered_detections_per_frame, frame_height)
    return rec_pos

# PUBLIC_INTERFACE
def get_detections(video_reader: VideoReader, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide multi-position placement and collect analysis.

    Parameters:
    - video_reader: Shared VideoReader instance
    - start_sec: Segment start time (seconds)
    - end_sec: Segment end time (seconds)
    - min_frames: Minimum number of frames to sample

    Returns:
    - dict with keys:
      - analysis: list of {frame_index, timestamp, detections}
      - placement: Placement (as dict)
      - recommended_position: 'top' or 'bottom' for backward compatibility
      - frame_width: int
      - frame_height: int
    """
    log.info("detections_segment start=%s end=%s", start_sec, end_sec)
    fps = video_reader.fps
    frame_height = int(video_reader.frame_height)
    frame_width = int(video_reader.frame_width)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    sub_time = end_sec - start_sec
    required_min_frames = int(sub_time) // 2
    min_frames = max(min_frames, required_min_frames)

    frame_indices = np.linspace(
        start_frame, end_frame, min(min_frames, abs(end_frame - start_frame + 1)), dtype=int
    )
    analysis = []
    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        result = {}
        ret, frame = video_reader.read_frame_at(frame_idx)
        if not ret:
            log.warning("frame_read_failed index=%s", frame_idx)
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
        result["frame_index"] = int(frame_idx)
        result["timestamp"] = float(frame_idx / fps)
        result["detections"] = detections
        analysis.append(result)

    log.debug("detections_per_frame=%s", filtered_detections_per_frame)
    # Decide rich placement
    # Determine top/bottom with a bottom-first rule: move to 'top' only if bottom region has burnt-in text
    rec_pos = decide_subtitle_position(filtered_detections_per_frame, frame_height)
    # Constrain placement search to chosen side to keep default bottom unless overlap forces top
    if rec_pos == "top":
        allowed = [Anchor.TOP_CENTER, Anchor.TOP_RIGHT, Anchor.TOP_LEFT]
    else:
        allowed = [Anchor.BOTTOM_CENTER, Anchor.BOTTOM_RIGHT, Anchor.BOTTOM_LEFT]
    placement_obj = decide_placement_from_detections(
        filtered_detections_per_frame, frame_width, frame_height, allowed_anchors=allowed
    )

    d = {
        "analysis": analysis,
        "placement": placement_to_dict(placement_obj),
        "recommended_position": rec_pos,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }
    return d

def placement_to_dict(p: Placement) -> Dict[str, Any]:
    """Serialize Placement to JSON-friendly dict."""
    d = {
        "kind": p.kind.value,
        "anchor": p.anchor.value if p.anchor else None,
        "x": p.x, "y": p.y, "relative_xy": p.relative_xy,
        "grid_rows": p.grid_rows, "grid_cols": p.grid_cols, "grid_r": p.grid_r, "grid_c": p.grid_c,
        "text_lines": p.text_lines, "width_fraction_hint": p.width_fraction_hint,
        "height_fraction_hint": p.height_fraction_hint,
    }
    if p.region:
        d["region"] = {
            "x0": p.region.x0, "y0": p.region.y0, "x1": p.region.x1, "y1": p.region.y1, "relative": p.region.relative
        }
    else:
        d["region"] = None
    return d

def dict_to_placement(d: Dict[str, Any]) -> Placement:
    """Deserialize Placement from dict."""
    kind = PlacementKind(d.get("kind", PlacementKind.ANCHOR.value))
    anchor_val = d.get("anchor")
    anchor = Anchor(anchor_val) if anchor_val else None
    region_dict = d.get("region")
    region = None
    if region_dict:
        region = Region(
            x0=region_dict["x0"], y0=region_dict["y0"], x1=region_dict["x1"], y1=region_dict["y1"],
            relative=region_dict.get("relative", True)
        )
    return Placement(
        kind=kind, anchor=anchor,
        x=d.get("x"), y=d.get("y"), relative_xy=d.get("relative_xy", True),
        grid_rows=d.get("grid_rows"), grid_cols=d.get("grid_cols"),
        grid_r=d.get("grid_r"), grid_c=d.get("grid_c"),
        region=region, text_lines=d.get("text_lines", 2),
        width_fraction_hint=d.get("width_fraction_hint"),
        height_fraction_hint=d.get("height_fraction_hint"),
    )

# ------------------------------------------------------------
# Detection wrappers for different formats
# ------------------------------------------------------------

# PUBLIC_INTERFACE
def detect_text_srt(video_reader: VideoReader, srt_path, min_frames=3, max_workers=5):
    """Analyze SRT lines and return detections and recommended positions.

    Parameters:
    - video_reader: Shared VideoReader instance
    - srt_path: Path to the .srt file
    - min_frames: Minimum frames sampled per subtitle segment
    - max_workers: Thread workers for parallel processing

    Returns:
    - dict keyed by processing order index with detection payloads including
      'subtitle_index', 'placement' (dict) and 'recommended_position'.
    """
    subs = pysrt.open(srt_path)

    def process_sub(sub, sub_index):
        start_sec = (
            sub.start.hours * 3600
            + sub.start.minutes * 60
            + sub.start.seconds
            + sub.start.milliseconds / 1000
        )
        end_sec = (
            sub.end.hours * 3600
            + sub.end.minutes * 60
            + sub.end.seconds
            + sub.end.milliseconds / 1000
        )
        log.debug("srt_detect_text index=%s text=%s", sub_index, sub.text)
        detections = get_detections(video_reader, start_sec, end_sec, min_frames)
        detections["subtitle_index"] = sub_index
        log.debug("srt_detections index=%s payload_len=%s", sub_index, len(str(detections)))
        return detections

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, sub, sub.index): i for i, sub in enumerate(subs)}
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                log.info("srt_detection_result_received index=%s", idx)
            except Exception as e:
                log.error("srt_detection_error idx=%s err=%s", idx, e, exc_info=True)
                results[idx] = None
    return results

# PUBLIC_INTERFACE
def detect_text_ass(video_reader: VideoReader, ass_path, max_workers=5):
    """Analyze ASS lines and return detections and recommended positions."""
    def process_sub(line: str, sub_index):
        if line.startswith("Dialogue:"):
            m = ASS_TIME_RE.match(line)
            if m:
                start_str, end_str = m.groups()

                def ass_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                matches = ASS_TEXT_RE.search(line)
                sub_text = matches.groups()[0] if matches else ""
                log.debug("ass_detect_text index=%s text=%s", sub_index, sub_text)
                detections = get_detections(video_reader, start_sec, end_sec)
                detections["subtitle_index"] = sub_index
                return detections
        return None

    with open(ass_path, "r") as f:
        input_ass_file = f.read()
        log.debug("ass_file_loaded size=%s", len(input_ass_file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_ass_file.splitlines())}
        results = {}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                result = future.result()
                log.debug("ass_original[%s]=%s", i, input_ass_file.splitlines()[i])
                log.debug("ass_result[%s]=%s", i, result)
                if result:
                    results[i] = result
                log.info("ass_detection_completed index=%s", i)
            except Exception as e:
                log.error("ass_detection_error line='%s' err=%s", input_ass_file.splitlines()[i], e)
                traceback.print_exc()
    return results

# PUBLIC_INTERFACE
def detect_text_ssa(video_reader: VideoReader, ssa_path, max_workers=5):
    """Analyze SSA lines and return detections and recommended positions."""
    def process_sub(line: str, sub_index):
        if line.startswith("Dialogue:"):
            m = SSA_TIME_RE.match(line)
            if m:
                start_str, end_str = m.groups()

                def ssa_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ssa_time_to_sec(start_str)
                end_sec = ssa_time_to_sec(end_str)

                matches = SSA_TEXT_RE.search(line)
                if matches:
                    sub_text = matches.groups()[0]
                    log.debug("ssa_detect_text index=%s text=%s", sub_index, sub_text)

                    detections = get_detections(video_reader, start_sec, end_sec)
                    detections["subtitle_index"] = sub_index
                    return detections
        return None

    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_file = f.read()
        log.debug("ssa_file_loaded size=%s", len(input_ssa_file))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_ssa_file.splitlines())}
        results = {}
        k = 0
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.debug("ssa_original[%s]=%s", i, input_ssa_file.splitlines()[i])
                log.debug("ssa_result[%s]=%s", i, result)
                if result:
                    results[k] = result
                    k += 1
            except Exception:
                log.error("ssa_detection_error line='%s'", input_ssa_file.splitlines()[i])
    return results

# PUBLIC_INTERFACE
def detect_text_vtt(video_reader: VideoReader, vtt_path, max_workers=5):
    """Analyze VTT timing lines and return detections and recommended positions."""
    log.info("vtt_detection_start")
    min_frames = 3

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:
            h, m, s_ms = hms
        else:
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line):
        matches = VTT_TIME_RE.search(line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str, end_str

    def process_sub(line: str, sub_index):
        if "-->" in line:
            start_str, end_str = extract_timestamps_string(line)
            log.debug("vtt_times start=%s end=%s", start_str, end_str)
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            detections = get_detections(video_reader, start_sec, end_sec, min_frames)
            detections["subtitle_index"] = sub_index
            return detections
        return None

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()
        log.debug("vtt_file_loaded size=%s", len(input_vtt_file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_vtt_file.splitlines())}
        results = {}
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.debug("vtt_original[%s]=%s", i, input_vtt_file.splitlines()[i])
                log.debug("vtt_result[%s]=%s", i, result)
                if result:
                    results[i] = result
            except Exception as e:
                log.error("vtt_detection_error line='%s' err=%s", input_vtt_file.splitlines()[i], e, exc_info=True)
    return results

# ------------------------------------------------------------
# Writers / Repositioning
# ------------------------------------------------------------

def _first_result_frame_dims(results: Dict[Any, Dict[str, Any]], default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """Extract frame (w,h) from the first result payload that contains them."""
    try:
        for _, payload in results.items():
            if not payload:
                continue
            w = payload.get("frame_width")
            h = payload.get("frame_height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return default

def _parse_ass_playres(ass_text: str) -> Tuple[int, int]:
    """Parse PlayResX/PlayResY from an ASS file; fallback to 1920x1080."""
    w, h = 1920, 1080
    try:
        for line in ass_text.splitlines():
            if line.startswith("PlayResX:"):
                w = int(line.split(":")[1].strip())
            elif line.startswith("PlayResY:"):
                h = int(line.split(":")[1].strip())
    except Exception:
        pass
    return w, h

def _placement_from_results_payload(payload: Dict[str, Any]) -> Optional[Placement]:
    """Get Placement from payload with backward compatibility."""
    if not payload:
        return None
    if "placement" in payload and payload["placement"]:
        try:
            return dict_to_placement(payload["placement"])
        except Exception:
            pass
    # Fallback to simple top/bottom anchor
    pos = payload.get("recommended_position")
    if pos == "top":
        return Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.TOP_CENTER)
    elif pos == "bottom":
        return Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.BOTTOM_CENTER)
    return None

# PUBLIC_INTERFACE
def reposition_srt(video_path, srt_path, output_ass_path, results, min_frames=3, max_workers=5):
    """Read SRT, use provided placements, and output ASS with appropriate overrides."""
    subs = pysrt.open(srt_path)
    if not subs:
        return
    output_ass_file = [None] * len(subs)

    # O(1) subtitle lookup by index (perf)
    subs_by_index = {s.index: s for s in subs}

    # Determine frame dims (prefer results' dims)
    frame_w, frame_h = _first_result_frame_dims(results)
    if (frame_w, frame_h) == (1920, 1080) and video_path:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if fw > 0 and fh > 0:
                    frame_w, frame_h = fw, fh
            cap.release()
        except Exception:
            pass

    def process_sub(sub_index, payload):
        sub = subs_by_index.get(sub_index)
        log.info("srt_line_loaded index=%s", sub_index)
        log.debug("srt_text=%s", getattr(sub, "text", None))

        placement = _placement_from_results_payload(payload) or Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.BOTTOM_CENTER)
        alignment_tag = ass_inline_for_placement(placement, frame_w, frame_h)

        raw_text = sub.text

        # Escape braces then preserve line breaks for ASS
        safe_text = raw_text.replace("{", r"\\{").replace("}", r"\\}")
        safe_text = safe_text.replace("\\n", r"\\N")

        # Map common styling tags using precompiled regex
        for pat, repl in SRT_TAG_REPLACEMENTS:
            safe_text = pat.sub(repl, safe_text)

        formatted_text = safe_text
        line = (
            f"Dialogue: 0,{to_ass_timestamp(sub.start)},{to_ass_timestamp(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return line, sub_index

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result],
            ): result
            for result in results
        }

        for future in as_completed(future_to_idx):
            result_key = future_to_idx[future]
            try:
                line, index = future.result()
                output_ass_file[index - 1] = line
            except Exception as e:
                sub_index = results[result_key].get("subtitle_index")
                sub = subs_by_index.get(sub_index)
                log.error(
                    "srt_processing_error text=%s index=%s subs_total=%s err=%s out_len=%s",
                    getattr(sub, "text", None),
                    sub_index,
                    len(subs),
                    e,
                    len(output_ass_file),
                    exc_info=True,
                )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {frame_w}\n"
            f"PlayResY: {frame_h}\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        for i in range(len(subs)):
            if output_ass_file[i]:
                f.write(output_ass_file[i])

    log.info("ass_written path=%s", output_ass_path)
    return output_ass_path

def _apply_placement_to_ass_line(line: str, placement: Placement, frame_w: int, frame_h: int) -> str:
    """Apply placement to a single ASS dialogue line by updating/adding overrides."""
    # Remove existing pos/move/an tags to avoid conflicts
    line = POS_TAG_RE.sub("", line)
    line = MOVE_TAG_RE.sub("", line)
    if placement.kind in (PlacementKind.ABSOLUTE, PlacementKind.REGION):
        # For absolute/region, include \pos and keep a reasonable alignment for baseline
        tag = ass_inline_for_placement(placement, frame_w, frame_h)
        # If an {\\anX} exists and our tag also includes \\an, replace it; else just append
        if AN_TAG_RE.search(line):
            # Replace existing {\\anX} with our alignment part, but retain any \pos we add
            # Extract anX from tag if present
            m = re.search(r"\{\\an(\d)\}", tag)
            if m:
                line = AN_TAG_RE.sub("{\\an" + m.group(1) + "}", line)
                # Append \pos part if exists
                pos_part = re.search(r"(\{\\pos\([^)]+\)\})", tag)
                if pos_part:
                    line = line.rstrip("\n") + pos_part.group(1)
            else:
                line = line.rstrip("\n") + tag
        else:
            line = line.rstrip("\n") + tag
    else:
        # Anchor/grid only - deal with {\\anX}
        if AN_TAG_RE.search(line):
            line = AN_TAG_RE.sub("{\\an" + str(anchor_to_ass_alignment(placement.anchor or Anchor.MIDDLE_CENTER)) + "}", line)
        else:
            line = line.rstrip("\n") + "{\\an" + str(anchor_to_ass_alignment(placement.anchor or Anchor.MIDDLE_CENTER)) + "}"
    return line

# PUBLIC_INTERFACE
def reposition_ass(ass_path, results, output_ass_path, max_workers=5):
    """Modify alignment/position tags in ASS dialogue lines using provided placements."""
    with open(ass_path, "r", encoding="utf-8") as f:
        input_ass_text = f.read()
        input_ass_file = input_ass_text.splitlines()
        log.debug("ass_loaded_lines=%s", len(input_ass_file))
    output_ass_file = input_ass_file[:]

    frame_w, frame_h = _parse_ass_playres(input_ass_text)

    def process_sub(sub_index, payload):
        line = input_ass_file[sub_index]
        if not line.startswith("Dialogue:"):
            return line
        m = ASS_TIME_RE.match(line)
        if not m:
            return line

        placement = _placement_from_results_payload(payload) or Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.BOTTOM_CENTER)
        line_updated = _apply_placement_to_ass_line(line, placement, frame_w, frame_h)
        log.info("ass_line_repositioned index=%s", sub_index)
        return line_updated

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result],
            ): results[result].get("subtitle_index")
            for result in results
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                result_line = future.result()
                log.debug("ass_original[%s]=%s", i, input_ass_file[i])
                log.debug("ass_result[%s]=%s", i, result_line)
                output_ass_file[int(i)] = result_line
                log.info("ass_completed[%s]", i)
            except Exception as e:
                log.error("ass_process_error line='%s' err=%s", input_ass_file[i], e)
                traceback.print_exc()

    new_ass_file = "\n".join(output_ass_file) if output_ass_file else ""
    log.debug("ass_new_file_len=%s", len(new_ass_file))
    open(output_ass_path, "w", encoding="utf-8").write(new_ass_file)

def _apply_placement_to_ssa_line(line: str, placement: Placement, frame_w: int, frame_h: int) -> str:
    """Apply placement to a single SSA dialogue line by updating alignment/position tags."""
    # SSA supports {\\anX} overrides similar to ASS. Use same strategy.
    line = POS_TAG_RE.sub("", line)
    line = MOVE_TAG_RE.sub("", line)

    if placement.kind in (PlacementKind.ABSOLUTE, PlacementKind.REGION):
        tag = ass_inline_for_placement(placement, frame_w, frame_h)
        if AN_TAG_RE.search(line):
            m = re.search(r"\{\\an(\d)\}", tag)
            if m:
                line = AN_TAG_RE.sub("{\\an" + m.group(1) + "}", line)
                pos_part = re.search(r"(\{\\pos\([^)]+\)\})", tag)
                if pos_part:
                    line = line.rstrip("\n") + pos_part.group(1)
            else:
                line = line.rstrip("\n") + tag
        else:
            line = line.rstrip("\n") + tag
    else:
        if AN_TAG_RE.search(line):
            line = AN_TAG_RE.sub("{\\an" + str(anchor_to_ass_alignment(placement.anchor or Anchor.MIDDLE_CENTER)) + "}", line)
        else:
            line = line.rstrip("\n") + "{\\an" + str(anchor_to_ass_alignment(placement.anchor or Anchor.MIDDLE_CENTER)) + "}"
    return line

# PUBLIC_INTERFACE
def reposition_ssa(ssa_path, results, output_ssa_path, max_workers=5):
    """Modify alignment/position tags in SSA dialogue lines using provided placements."""
    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_text = f.read()
        input_ssa_file = input_ssa_text.splitlines()
        log.debug("ssa_loaded_lines=%s", len(input_ssa_file))
    output_ssa_file = input_ssa_file[:]

    # SSA may not always have PlayRes; fallback to typical FHD
    frame_w, frame_h = _parse_ass_playres(input_ssa_text)

    def process_sub(sub_index, payload):
        line = input_ssa_file[sub_index]
        if not line.startswith("Dialogue:"):
            return line
        m = SSA_TIME_RE.match(line)
        if not m:
            return line

        placement = _placement_from_results_payload(payload) or Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.BOTTOM_CENTER)
        line_updated = _apply_placement_to_ssa_line(line, placement, frame_w, frame_h)
        log.info("ssa_line_repositioned index=%s", sub_index)
        return line_updated

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result],
            ): results[result].get("subtitle_index")
            for result in results
        }
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result_line = future.result()
                log.debug("ssa_original[%s]=%s", i, input_ssa_file[i])
                log.debug("ssa_result[%s]=%s", i, result_line)
                output_ssa_file[i] = result_line
            except Exception as e:
                log.error("ssa_process_error line='%s' err=%s", input_ssa_file[i], e)

    new_ssa_file = "\n".join(output_ssa_file) if results else ""
    log.debug("ssa_new_file_len=%s", len(new_ssa_file))

    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)

# PUBLIC_INTERFACE
def reposition_vtt(vtt_path, results, output_vtt_path, max_workers=5):
    """For VTT: modify 'line:' / 'position:' / 'align:' cue settings using Placement."""
    log.info("vtt_reposition_start")
    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()
        input_vtt_file = input_vtt_file.splitlines()
        log.debug("vtt_loaded_lines=%s", len(input_vtt_file))
    output_vtt_file = input_vtt_file[:]

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:
            h, m, s_ms = hms
        else:
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line):
        matches = VTT_TIME_RE.search(line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str, end_str

    def process_sub(sub_index, payload):
        line = input_vtt_file[sub_index]
        if "-->" not in line:
            return line

        # Frame dims for percentage conversion (from payload)
        frame_w = payload.get("frame_width", 1920)
        frame_h = payload.get("frame_height", 1080)
        placement = _placement_from_results_payload(payload) or Placement(kind=PlacementKind.ANCHOR, anchor=Anchor.BOTTOM_CENTER)

        # Prepare settings
        settings = vtt_settings_for_placement(placement, frame_w, frame_h)

        # Apply or append settings
        if "line:" in line:
            line = VTT_LINE_POS_RE.sub(settings["line"], line)
        else:
            line = line.strip() + " " + settings["line"]

        if "position:" in line:
            line = VTT_POSITION_RE.sub(settings["position"], line)
        else:
            line = line.strip() + " " + settings["position"]

        if "align:" in line:
            line = VTT_ALIGN_RE.sub(settings["align"], line)
        else:
            line = line.strip() + " " + settings["align"]

        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result],
            ): results[result].get("subtitle_index")
            for result in results
        }

        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result_line = future.result()
                log.debug("vtt_original[%s]=%s", i, input_vtt_file[i])
                log.debug("vtt_result[%s]=%s", i, result_line)
                output_vtt_file[i] = result_line
            except Exception as e:
                log.error("vtt_process_error line='%s' err=%s", input_vtt_file[i], e, exc_info=True)

    new_vtt_file = "\n".join(output_vtt_file) if output_vtt_file else ""
    log.debug("vtt_new_file_len=%s", len(new_vtt_file))

    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_vtt_file)

# ------------------------------------------------------------
# Display and Orchestrator
# ------------------------------------------------------------

# PUBLIC_INTERFACE
def display_results(results):
    """Log collected detection results in chronological order."""
    results = dict(sorted(results.items()))
    for index, result in results.items():
        log.info("result_index=%s", index)
        log.debug("result_payload=%s", result)

# PUBLIC_INTERFACE
def process_subtitle(video_path, subtitle_path, max_workers=12):
    """Process a subtitle file, detect positions, and write repositioned outputs.

    Parameters:
    - video_path: Path to the video file
    - subtitle_path: Path to the subtitle file (.srt, .ass, .ssa, .vtt)
    - max_workers: ThreadPool workers for parallel analysis

    Returns:
    - Path to the output repositioned subtitle file (.ass/.ssa/.vtt)
    """
    start = time.time()
    ext = os.path.splitext(subtitle_path)[1].lower()
    print("in process subtitle")

    with VideoReader(video_path) as video_reader:
        if ext == ".srt":
            results = detect_text_srt(video_reader, subtitle_path, max_workers=max_workers)
            log.info("srt_results_collected")
            if results:
                display_results(results)
            else:
                print("No results")
            output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
            reposition_srt(video_path, subtitle_path, output_file, results=results, max_workers=max_workers)
        elif ext == ".ass":
            results = detect_text_ass(video_reader, subtitle_path, max_workers=max_workers)
            print("displaying results")
            if results:
                display_results(results)
            else:
                print("No results")
            output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
            reposition_ass(ass_path=subtitle_path, output_ass_path=output_file, results=results, max_workers=max_workers)
        elif ext == ".ssa":
            results = detect_text_ssa(video_reader, subtitle_path, max_workers=max_workers)
            log.info("ssa_results_collected type=%s", type(results))
            if results:
                display_results(results)
            else:
                print("No results")
            output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
            reposition_ssa(ssa_path=subtitle_path, output_ssa_path=output_file, results=results, max_workers=max_workers)
        elif ext == ".vtt":
            results = detect_text_vtt(video_reader, subtitle_path, max_workers=max_workers)
            log.info("vtt_results_collected")
            if results:
                display_results(results)
            else:
                print("No results")
            output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
            reposition_vtt(vtt_path=subtitle_path, output_vtt_path=output_file, results=results, max_workers=max_workers)
        else:
            raise ValueError(f"Unsupported subtitle format: {ext}")

    log.info("output_written path=%s", output_file)
    print(f"Repositioned subtitle saved: {output_file}")
    end = time.time()
    print("total time taken", end - start)
    return output_file

if __name__ == "__main__":
    # Example invocation retained from original script footprint
    video_path = r"C:\Users\42785\Desktop\sample_videos\ABC_news.mp4"
    sub_path = r"C:\Users\42785\Desktop\sample_videos\generated_subtitles\ABC_news_generated.srt"
    max_workers = 8
    process_subtitle(video_path, sub_path, max_workers)
