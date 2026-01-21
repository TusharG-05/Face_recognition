import cv2
import time
import numpy as np
import multiprocessing
import os
import threading
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def face_recognition_worker(frame_queue, result_queue, known_encoding):
    """
    Multithreaded worker: 
    - Main Loop: Fast Detection (30+ FPS)
    - Background Thread: Slow Recognition (No blocking)
    """
    state = {
        'img': None,
        'locs': [],
        'match': False,
        'conf': 1.0,
        'last_recog_time': 0,
        'lock': threading.Lock()
    }

    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path='app/assets/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=4,
        min_face_detection_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    def recognition_loop():
        # Pre-warm DeepFace model
        try:
            DeepFace.build_model("ArcFace")
        except:
            pass

        while True:
            time.sleep(0.01)
            with state['lock']:
                img_copy = state['img'].copy() if state['img'] is not None else None
                locs_copy = list(state['locs'])
            
            if img_copy is not None and locs_copy:
                try:
                    # Slow recognition happens in background thread
                    # Convert MediaPipe locs (top, right, bottom, left) to DeepFace bboxes if needed
                    # DeepFace represent can take a list of alignments or full image.
                    # For accuracy in 2026, we use ArcFace.
                    matches = []
                    for (t, r, b, l) in locs_copy:
                        # Crop face
                        face_img_rgb = img_copy[max(0, t):b, max(0, l):r]
                        if face_img_rgb.size == 0: continue
                        
                        # DeepFace expectation: BGR for numpy arrays
                        face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
                        
                        objs = DeepFace.represent(
                            img_path=face_img_bgr, 
                            model_name="ArcFace", 
                            enforce_detection=False,
                            detector_backend="skip",
                            align=True # Standard ArcFace alignment
                        )
                        
                        if objs and known_encoding is not None:
                            embedding = np.array(objs[0]["embedding"])
                            # Manual Cosine Distance: 1 - (A.B / (|A||B|))
                            dot = np.dot(embedding, known_encoding)
                            norm_a = np.linalg.norm(embedding)
                            norm_b = np.linalg.norm(known_encoding)
                            dist = 1 - (dot / (norm_a * norm_b))
                            matches.append(dist)

                    if matches:
                        min_dist = min(matches)
                        # ArcFace Cosine threshold is typically around 0.4 for high security
                        match = min_dist <= 0.45 
                        with state['lock']:
                            state['match'] = match
                            state['conf'] = float(min_dist)
                except Exception as e:
                    # print(f"Recognition Thread Error: {e}")
                    pass
            time.sleep(0.3) # Throttle recognition thread for efficiency

    recog_thread = threading.Thread(target=recognition_loop, daemon=True)
    recog_thread.start()

    while True:
        try:
            # Wait for 1 second max so we can check for exit signals
            frame_bgr = frame_queue.get(timeout=1)
        except:
            continue

        if frame_bgr is None: # Sentinel value to stop
            break

        try:
            # Convert to RGB inside the worker process
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Scale frame for speed
            h, w = frame_rgb.shape[:2]
            target_h = 540
            s = target_h / h if h > target_h else 1.0
            
            # Resize if needed
            img = cv2.resize(frame_rgb, (0,0), fx=s, fy=s) if s < 1.0 else frame_rgb
            
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            
            # Perform face detection with MediaPipe
            detection_result = detector.detect(mp_image)
            
            ih, iw = img.shape[:2]
            new_locs = []
            if detection_result.face_landmarks:
                for landmarks in detection_result.face_landmarks:
                    # MediaPipe landmarks are normalized [0,1]. Convert to pixel coordinates.
                    # Calculate bounding box from landmarks
                    xs = [lm.x for lm in landmarks]
                    ys = [lm.y for lm in landmarks]
                    
                    # Pixels (MediaPipe uses normalized 0-1)
                    t, l, b, r = int(min(ys) * ih), int(min(xs) * iw), int(max(ys) * ih), int(max(xs) * iw)
                    
                    # Add padding for better recognition accuracy
                    hp, wp = int((b-t)*0.1), int((r-l)*0.1)
                    new_locs.append((max(0, t-hp), min(iw, r+wp), min(ih, b+hp), max(0, l-wp)))

            with state['lock']:
                state['img'] = img
                state['locs'] = new_locs
                match_val, conf_val = state['match'], state['conf']
            
            # Map back coordinates accurately
            final_locs = [(int(t/s), int(r/s), int(b/s), int(l/s)) for (t,r,b,l) in new_locs]

            if not result_queue.full():
                result_queue.put((match_val, conf_val, len(final_locs), final_locs))
        except Exception as e:
            print(f"Face Worker Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Starting Zero-Lag Modernized Face Service (2026)...")
        try:
            # Generate encoding for known person using DeepFace
            objs = DeepFace.represent(
                img_path=known_person_path, 
                model_name="ArcFace", 
                enforce_detection=True,
                detector_backend="opencv"
            )
            self.known_encoding = np.array(objs[0]["embedding"]) if objs else None
        except Exception as e:
            print(f"Known Person Load Error: {e}")
            self.known_encoding = None

        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        self.worker = multiprocessing.Process(
            target=face_recognition_worker, 
            args=(self.frame_queue, self.result_queue, self.known_encoding)
        )
        self.worker.daemon = True
        self.worker.start()

    def process_frame(self, frame_bgr):
        # RESIZE IN MAIN THREAD - Crucial for zero delay
        h, w = frame_bgr.shape[:2]
        s = 360.0 / h if h > 360 else 1.0
        
        # Convert to RGB and Resize
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (0,0), fx=s, fy=s) if s < 1.0 else img_rgb

        if not self.frame_queue.full():
            self.frame_queue.put(img_small)
        
        try:
            match, conf, n_faces, locs = self.result_queue.get_nowait()
            # Scale coordinates back up
            scaled_locs = [(int(t/s), int(r/s), int(b/s), int(l/s)) for (t,r,b,l) in locs]
            return match, conf, n_faces, scaled_locs
        except:
            return None, None, 0, []

    def close(self):
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()


