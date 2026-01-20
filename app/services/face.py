import cv2
import time
import numpy as np
import multiprocessing
import os
import face_recognition

def face_recognition_worker(frame_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, known_encoding: np.ndarray):
    """
    Continuous loop for face recognition. 
    Running in a separate process to avoid blocking the main server thread.
    """
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
            
            # Scale frame for speed (720p is the sweet spot for HOG distance detection)
            h, w = frame_rgb.shape[:2]
            s = 720 / h if h > 720 else 1.0
            img = cv2.resize(frame_rgb, (0,0), fx=s, fy=s) if s < 1.0 else frame_rgb

            # Detect with upsample=1 (Faster than level 2, but 720p provides enough detail)
            locs = face_recognition.face_locations(img, model="hog", number_of_times_to_upsample=1)
            encs = face_recognition.face_encodings(img, locs)
            
            match, conf = False, 1.0
            if encs and known_encoding is not None:
                dists = face_recognition.face_distance(encs, known_encoding)
                match = any(d <= 0.5 for d in dists)
                conf = min(dists)
            
            # Map back coordinates accurately
            final_locs = [(int(t/s), int(r/s), int(b/s), int(l/s)) for (t,r,b,l) in locs]

            if not result_queue.full():
                result_queue.put((match, conf, len(encs), final_locs))
        except Exception as e:
            print(f"Face Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Initializing FaceDetector...")
        if not os.path.exists(known_person_path):
            raise FileNotFoundError(f"Error: {known_person_path} not found")
            
        # Load Reference Image safely
        try:
            known_image = face_recognition.load_image_file(known_person_path)
            encodings = face_recognition.face_encodings(known_image)
            if not encodings:
                print(f"Warning: No face found in {known_person_path}. Auth will fail.")
                self.known_encoding = None
            else:
                self.known_encoding = encodings[0]
                print("Known face loaded successfully.")
        except Exception as e:
            print(f"Error loading reference image: {e}")
            self.known_encoding = None

        # Setup Queues
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)

        # Start Worker
        self.worker = multiprocessing.Process(
            target=face_recognition_worker, 
            args=(self.frame_queue, self.result_queue, self.known_encoding)
        )
        self.worker.daemon = True
        self.worker.start()
        print("Worker process started.")

    def process_frame(self, frame_bgr):
        # 1. Send BGR directly (Fastest for main thread)
        if not self.frame_queue.full():
            self.frame_queue.put(frame_bgr)

        # 3. Retrieve Latest Result (Non-blocking)
        try:
            return self.result_queue.get_nowait()
        except:
            # Return defaults/empty if processing isn't done yet
            return None, None, 0, []

    def close(self):
        """Clean up background resources."""
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()


