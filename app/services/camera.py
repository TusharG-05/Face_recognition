import cv2
import threading
import time
import os
from typing import Optional, Tuple
from .face import FaceDetector
from .gaze import GazeDetector

class CameraService:
    """
    Singleton class to manage the camera resource and orchestrate detectors.
    Ensures only one thread accesses the camera at a time.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CameraService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        
        self.camera: Optional[cv2.VideoCapture] = None
        self.face_detector: Optional[FaceDetector] = None
        self.gaze_detector: Optional[GazeDetector] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[bytes] = None
        self.frame_id: int = 0
        self.frame_lock = threading.Lock()

    def start(self, video_source=0):
        if self.running: return
        print(f"Lazy starting camera (Source: {video_source})...", flush=True)
        
        # Init Detectors in background to avoid blocking the first frame
        def init_detectors():
            print("Background: Initializing Detectors...", flush=True)
            known_path = "app/assets/known_person.jpg"
            if os.path.exists(known_path):
                try:
                    self.face_detector = FaceDetector(known_person_path=known_path)
                    print("Background: FaceDetector ready.", flush=True)
                except Exception as e:
                    print(f"Background: FaceDetector failed: {e}", flush=True)
            
            gaze_path = "app/assets/face_landmarker.task"
            if os.path.exists(gaze_path):
                try:
                    self.gaze_detector = GazeDetector(model_path=gaze_path, max_faces=1)
                    print("Background: GazeDetector ready.", flush=True)
                except Exception as e:
                    print(f"Background: GazeDetector failed: {e}", flush=True)

        threading.Thread(target=init_detectors, daemon=True).start()

        # Open Camera: Try DSHOW for Windows stability first
        print("Opening VideoCapture...", flush=True)
        self.camera = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(video_source)
            
        if not self.camera.isOpened():
            print("Error: Could not open camera hardware.", flush=True)
            return
            
        # Fast initialization
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Camera Thread Started. System Live.", flush=True)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()
        if self.face_detector:
            self.face_detector.close()
        if self.gaze_detector:
            self.gaze_detector.close()

    def _process_loop(self):
        last_face_status = (False, 1.0, 0, [])
        last_gaze_status = "Initializing..."
        last_face_time = 0
        
        while self.running:
            success, frame = self.camera.read()
            
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # 1. Detection
            now = time.time()
            if self.face_detector and (now - last_face_time) > 0.033: # 30 FPS (Zero-Lag)
                last_face_time = now
                # Pass BGR; worker will convert to RGB
                f_res = self.face_detector.process_frame(frame)
                if f_res and f_res[0] is not None: 
                    last_face_status = f_res
            
            if self.gaze_detector:
                # Pass BGR; worker will convert to RGB
                g_res = self.gaze_detector.process_frame(frame)
                if g_res: 
                    last_gaze_status = g_res

            # 2. Annotation
            found, dist, n_face, locs = last_face_status
            gaze_txt = last_gaze_status
            
            if n_face > 1: gaze_txt = "ERROR: Multiple Faces"
            elif n_face == 0: gaze_txt = "No Face"

            f_clr = (0, 255, 0) if (found and n_face == 1) else (0, 0, 255)
            g_clr = (0, 0, 255) if "WARNING" in str(gaze_txt) else (255, 255, 0)
            
            cv2.putText(frame, f"Auth: {found} ({dist:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, f_clr, 2)
            cv2.putText(frame, f"Gaze: {gaze_txt}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_clr, 2)
            
            for (t, r, b, l) in locs:
                cv2.rectangle(frame, (l, t), (r, b), f_clr, 2)

            # 3. Store
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with self.frame_lock:
                    self.latest_frame = buffer.tobytes()
                    self.frame_id += 1

    def update_identity(self, image_bytes: bytes) -> bool:
        """Updates the known person identity and reloads the detector."""
        filepath = "app/assets/known_person.jpg"
        
        # Save new file
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        print(f"Identity updated. Reloading FaceDetector from {filepath}...")
        
        # Stop old detector
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
            
        # Start new detector
        try:
            self.face_detector = FaceDetector(known_person_path=filepath)
            print("FaceDetector reloaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to reload FaceDetector: {e}")
            return False

    def get_frame(self) -> Tuple[Optional[bytes], int]:
        """Returns the latest MJPEG frame and its ID."""
        with self.frame_lock:
            return self.latest_frame, self.frame_id
