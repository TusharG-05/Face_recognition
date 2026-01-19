import cv2
import time
import numpy as np
import multiprocessing
import os
import face_recognition

def face_recognition_worker(frame_queue, result_queue, known_encoding):
    while True:
        try:
            rgb_small_frame = frame_queue.get(timeout=1)
        except:
            continue

        if rgb_small_frame is None:
            break

        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame)
            
            match_found = False
            min_dist = 1.0
            
            if len(face_encodings) > 0:
                distances = face_recognition.face_distance([known_encoding], face_encodings[0])
                min_dist = distances[0]
                if min_dist <= 0.5: # Threshold
                    match_found = True
            
            # Send result back
            if not result_queue.full():
                result_queue.put((match_found, min_dist))
                
        except Exception as e:
            print(f"Worker Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Initializing FaceDetector...")
        if not os.path.exists(known_person_path):
            raise FileNotFoundError(f"Error: {known_person_path} not found")
            
        # Load Known Face
        known_image = face_recognition.load_image_file(known_person_path)
        self.known_encoding = face_recognition.face_encodings(known_image)[0]
        print("Known face loaded.")

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
        # 1. Resize & Convert to RGB
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 2. Put in Queue
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small_frame)

        # 3. Check for Result
        try:
            return self.result_queue.get_nowait()
        except:
            return None, None 

    def close(self):
        # Send Sentinel to stop the worker gracefully
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate() # Fallback

# =============================
# MOCK FRONTEND (Simulating a Server)
# =============================
if __name__ == "__main__":
    # It opens the camera locally to test the Class.
    
    detector = FaceDetector()
    try:
        from gaze_detector import GazeDetector
        gaze_detector = GazeDetector()
    except ImportError:
        print("Could not import GazeDetector. Make sure mediapipe is installed.")
        gaze_detector = None
    
    # Simulate Camera Input
    video_path = "video.mp4"
    if os.path.exists(video_path):
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(0)

    print("--- SIMULATING FRONTEND CONNECTION ---")
    start_time = time.time()
    
    last_found = False
    last_dist = 1.0
    last_gaze = "Initializing..."

    while True:
        ret, frame = video.read()
        if not ret: break

        # === THE API CALL ===
        face_result = detector.process_frame(frame)
        
        gaze_result = None
        if gaze_detector:
            gaze_result = gaze_detector.process_frame(frame)
        
        # Update State if we got a fresh result
        found, dist = face_result
        if found is not None:
            last_found = found
            last_dist = dist
            if found:
                print(f"API Result: Found! ({dist:.2f})")
        
        if gaze_result:
            last_gaze = gaze_result

        # Display (Only for testing)
        color = (0, 255, 0) if last_found else (0, 0, 255)
        cv2.putText(frame, f"Found: {last_found} ({last_dist:.2f})", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Gaze: {last_gaze}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Frontend Simulation", frame)
        if cv2.waitKey(1) == ord('q'): break

    total_time = time.time() - start_time
    print(f"Session ended. Total time: {total_time:.2f}s")

    detector.close()
    if gaze_detector:
        gaze_detector.close()
    video.release()
    cv2.destroyAllWindows()
