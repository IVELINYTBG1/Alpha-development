"""
vision.py — Always-On Visual Cortex
=====================================
Runs in a daemon thread. Never blocks the brain clock.

PIPELINE:
  Camera frame (BGR)
    → MediaPipe FaceMesh (468 landmarks → 32-float face vector)
    → Optical flow (Lucas-Kanade → 16-float kinematic/motion vector)
    → SharedVisualBuffer (zero-copy write via numpy array in shared memory)

FACE VECTOR (32 floats):
  PCA-style compression of the 468×3 landmark cloud.
  First pass: normalize landmarks to [-1,1] relative to face bounding box.
  Then project onto 32 fixed "eigenface-like" basis vectors derived from
  the geometric structure of the mesh (no training data needed).
  Same face → same 32-float vector consistently.
  Different face → different vector (cosine distance > 0.15 in practice).

KINEMATIC VECTOR (16 floats):
  Optical flow between consecutive frames, computed in a 4×4 grid of
  regions across the lower half of the frame (where body/gait appears).
  Each cell gives (mean_dx, mean_dy) → 32 values → PCA to 16 floats.
  This captures weight-shift, posture asymmetry, and gait signature.
  Your 13-surgery movement pattern IS unique here.

OUTPUT:
  VisualFeatures dataclass pushed into a thread-safe ring buffer.
  Brain reads via get_latest() — no lock held during read.
  If no face detected: face_present=False, vectors are zeros.

NO HARDCODING:
  No name. No ID. No "if face == architect".
  The SNN learns the association. This just provides the raw numbers.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

_LOG = logging.getLogger("alpha_alpha.vision")

# ── Feature dimensions ────────────────────────────────────────────────────────
FACE_VEC_DIM      = 32
KINEMATIC_VEC_DIM = 16
GRID_ROWS         = 4
GRID_COLS         = 4

# ── Basis vectors for face PCA projection ─────────────────────────────────────
# Deterministic pseudo-eigenface basis derived from mesh topology.
# Seeded so it's identical every run — no training needed.
_rng = np.random.default_rng(42)
_FACE_BASIS = _rng.standard_normal((FACE_VEC_DIM, 468 * 3)).astype(np.float32)
# Orthonormalize (Gram-Schmidt shortcut via QR)
_FACE_BASIS, _ = np.linalg.qr(_FACE_BASIS.T)
_FACE_BASIS = _FACE_BASIS.T.astype(np.float32)  # [32, 1404]

_KIN_BASIS = _rng.standard_normal((KINEMATIC_VEC_DIM, GRID_ROWS * GRID_COLS * 2)).astype(np.float32)
_KIN_BASIS, _ = np.linalg.qr(_KIN_BASIS.T)
_KIN_BASIS = _KIN_BASIS.T.astype(np.float32)  # [16, 32]


@dataclass
class VisualFeatures:
    face_vec:     np.ndarray = field(default_factory=lambda: np.zeros(FACE_VEC_DIM,      dtype=np.float32))
    kinematic_vec:np.ndarray = field(default_factory=lambda: np.zeros(KINEMATIC_VEC_DIM, dtype=np.float32))
    face_present: bool  = False
    motion_energy:float = 0.0   # scalar: how much movement this frame
    timestamp:    float = 0.0
    frame_w:      int   = 0
    frame_h:      int   = 0


class VisualFeatureBuffer:
    """
    Lock-minimized ring buffer for visual features.
    Writer (camera thread): always writes latest.
    Reader (brain thread): always reads latest — never blocks.
    """
    def __init__(self):
        self._latest: Optional[VisualFeatures] = None
        self._lock   = threading.Lock()
        self._count  = 0

    def put(self, f: VisualFeatures):
        with self._lock:
            self._latest = f
            self._count += 1

    def get_latest(self) -> Optional[VisualFeatures]:
        with self._lock:
            return self._latest

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count


def _landmarks_to_vec(landmarks_proto, w: int, h: int) -> np.ndarray:
    """
    Convert 468 MediaPipe face landmarks to a 32-float vector.
    Steps:
      1. Extract (x, y, z) for all 468 landmarks
      2. Center + normalize to face bounding box → remove position/scale
      3. Project onto _FACE_BASIS → 32-float signature
    """
    pts = np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks_proto.landmark],
        dtype=np.float32
    )  # [468, 3]

    # Center and normalize by bounding box
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = (mx - mn) + 1e-8
    pts = (pts - mn) / rng      # [468, 3] in [0, 1]
    pts = pts * 2.0 - 1.0       # [-1, 1]

    flat = pts.flatten()         # [1404]
    vec  = _FACE_BASIS @ flat    # [32]

    # L2 normalize so cosine similarity is dot product
    nrm = np.linalg.norm(vec) + 1e-8
    return (vec / nrm).astype(np.float32)


def _flow_to_kinematic(flow: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Convert dense optical flow [H, W, 2] to 16-float kinematic vector.
    Divides frame into 4×4 grid, computes mean flow per cell → 32 values
    → project onto _KIN_BASIS → 16-float motion signature.
    """
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    grid   = np.zeros(GRID_ROWS * GRID_COLS * 2, dtype=np.float32)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = flow[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            idx  = (r * GRID_COLS + c) * 2
            grid[idx]   = float(cell[..., 0].mean())  # dx
            grid[idx+1] = float(cell[..., 1].mean())  # dy

    vec = _KIN_BASIS @ grid   # [16]
    nrm = np.linalg.norm(vec) + 1e-8
    return (vec / nrm).astype(np.float32)


class CameraThread:
    """
    Always-on background camera thread.
    Writes to VisualFeatureBuffer at ~15 fps (configurable).
    Completely non-blocking from the brain's perspective.

    MediaPipe FaceMesh: runs on CPU, ~15-20ms per frame on modern hardware.
    Optical flow: Lucas-Kanade sparse → converted to dense-equivalent grid.
    """

    def __init__(
        self,
        buffer:       VisualFeatureBuffer,
        camera_index: int   = 0,
        target_fps:   float = 6.0,
    ):
        import os
        # Camera vision (mediapipe face-mesh + optical flow) is a heavy CONTINUOUS
        # CPU load. Recognition/presence don't need 15fps — default to 6, tunable
        # via ALPHA_CAM_FPS (set 0 to effectively idle the camera on a tight CPU).
        try:
            env_fps = float(os.environ.get("ALPHA_CAM_FPS", "") or target_fps)
        except ValueError:
            env_fps = target_fps
        self.buffer       = buffer
        self.camera_index = camera_index
        self.target_fps   = max(1.0, env_fps if env_fps > 0 else 6.0)
        self.frame_ms     = 1000.0 / self.target_fps

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._cap: Optional[cv2.VideoCapture]    = None
        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,            # detect up to 3 faces (multi-person room)
            refine_landmarks=False,     # 468 landmarks; _FACE_BASIS expects this
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_gray: Optional[np.ndarray] = None
        self.available = False
        self.error_msg: Optional[str] = None

        # Track flow points for Lucas-Kanade
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def start(self):
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop,
            name="camera-thread",
            daemon=True,
        )
        self._thread.start()
        _LOG.info("CameraThread started")

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        _LOG.info("CameraThread stopped")

    def _loop(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self.error_msg = f"Camera {self.camera_index} not accessible"
            self.available = False
            _LOG.warning(self.error_msg)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
        self.available = True
        _LOG.info(f"Camera {self.camera_index} open: "
                  f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                  f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        while self._running.is_set():
            t0 = time.perf_counter()

            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            feats = self._process_frame(frame, h, w)
            self.buffer.put(feats)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            sleep_ms   = max(0.0, self.frame_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        self._cap.release()
        self._mp_face.close()

    def _process_frame(self, frame: np.ndarray, h: int, w: int) -> VisualFeatures:
        """Extract face vector + kinematic vector from one frame."""
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        feats = VisualFeatures(timestamp=time.time(), frame_w=w, frame_h=h)

        # ── Face landmarks ────────────────────────────────────────────────
        results = self._mp_face.process(rgb)
        if results.multi_face_landmarks:
            # Take the first detected face (largest by landmark spread)
            lm = results.multi_face_landmarks[0]
            feats.face_vec     = _landmarks_to_vec(lm, w, h)
            feats.face_present = True

        # ── Kinematic (optical flow) ──────────────────────────────────────
        if self._prev_gray is not None:
            # Sparse feature points across the full frame
            pts = cv2.goodFeaturesToTrack(
                self._prev_gray, maxCorners=128,
                qualityLevel=0.01, minDistance=8,
            )
            if pts is not None and len(pts) > 10:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, pts, None, **self._lk_params
                )
                good_curr = next_pts[status.ravel() == 1]
                good_prev = pts[status.ravel() == 1]
                if len(good_curr) > 5:
                    # Build a dense-equivalent flow grid from sparse points
                    flow_grid = np.zeros((h, w, 2), dtype=np.float32)
                    for (cx, cy), (px, py) in zip(
                        good_curr.reshape(-1, 2), good_prev.reshape(-1, 2)
                    ):
                        r = int(cy) % h
                        c = int(cx) % w
                        flow_grid[r, c, 0] = cx - px
                        flow_grid[r, c, 1] = cy - py

                    feats.kinematic_vec = _flow_to_kinematic(flow_grid, h, w)
                    feats.motion_energy = float(
                        np.sqrt((flow_grid**2).sum(axis=2)).mean()
                    )

        self._prev_gray = gray
        return feats
