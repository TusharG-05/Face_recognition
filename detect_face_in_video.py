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
            # Face Recognition Logic
            # Upsample 2x to find smaller faces (High Sensitivity)
            face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            num_faces = len(face_encodings)
            
            match_found = False
            min_dist = 1.0
            
            if num_faces > 0:
                # We check the first face found against our known encoding
                distances = face_recognition.face_distance([known_encoding], face_encodings[0])
                min_dist = distances[0]
                if min_dist <= 0.5: # Strict threshold for a match
                    match_found = True
            
            # Send result back (including face count and locations)
            if not result_queue.full():
                result_queue.put((match_found, min_dist, num_faces, face_locations))
                
        except Exception as e:
            print(f"Worker Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Initializing FaceDetector...")
        if not os.path.exists(known_person_path):
            raise FileNotFoundError(f"Error: {known_person_path} not found")
            
        # Load Reference Image
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
        # 1. Resize & Convert to RGB (Increase resolution for distant faces)
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 2. Send to Worker (Non-blocking)
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small_frame)

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

# =============================
# MOCK FRONTEND (Simulating a Server)
# =============================
if __name__ == "__main__":
    
    # helper for UI interaction
    stop_button_pressed = [False]
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and hasattr(on_mouse_click, 'button_rect'):
            rx, ry, rw, rh = on_mouse_click.button_rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                stop_button_pressed[0] = True

    detector = FaceDetector()
    
    # Gaze Detection (Optional Plugin)
    gaze_detector = None
    if not os.path.exists('face_landmarker.task'):
        print("WARNING: face_landmarker.task missing. Gaze detection disabled.")
    else:
        try:
            from gaze_detector import GazeDetector
            gaze_detector = GazeDetector(max_faces=1)
        except Exception as e:
            print(f"Gaze Init Failed: {e}")
            gaze_detector = None
    
    # 2. Setup Video Source
    video_path = "video.mp4"
    video = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)

    print("--- SIMULATION STARTED ---")
    
    last_found = False
    last_dist = 1.0
    last_num_faces = 0
    last_locations = []
    last_gaze = "Initializing..." if gaze_detector else "Disabled"

    while True:
        ret, frame = video.read()
        if not ret: break

        # --- [STEP 1] CALL BACKEND APIS ---
        face_result = detector.process_frame(frame)
        
        gaze_result = None
        if gaze_detector:
            gaze_result = gaze_detector.process_frame(frame)
        
        # --- [STEP 2] UPDATE STATE ---
        found, dist, n_faces, locs = face_result
        if found is not None:
            last_found = found
            last_dist = dist
            last_num_faces = n_faces
            last_locations = locs
            if found: print(f"Found Match! Confidence: {dist:.2f}")

        # Gaze Status Update
        if gaze_result:
            last_gaze = gaze_result

        # --- [STEP 3] RENDER UI (Logic Mapping) ---
        
        # Determine Global Status Message
        display_status = last_gaze
        if last_num_faces > 1:
            display_status = "ERROR: Multiple Faces Detected!"
        elif last_num_faces == 0:
             display_status = "No Face Detected"
        elif last_num_faces == 1 and gaze_result:
            display_status = last_gaze

        # Display (Only for testing)
        color = (0, 255, 0) if (last_found and last_num_faces == 1) else (0, 0, 255)
        text_found = f"Authorized: {last_found}" if last_num_faces <= 1 else "Authorized: False"
        
        cv2.putText(frame, f"{text_found} ({last_dist:.2f})", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        gaze_color = (0, 0, 255) if "WARNING" in display_status or "ERROR" in display_status else (255, 255, 0)
        cv2.putText(frame, f"Status: {display_status}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
        
        # Draw Faces
        for i, (top, right, bottom, left) in enumerate(last_locations):
            top *= 2; right *= 2; bottom *= 2; left *= 2 # Undo scaling
            box_color = (0, 255, 0) if (i == 0 and last_found and last_num_faces == 1) else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Stop Button
        h, w = frame.shape[:2]
        bx, by, bw, bh = w - 120, h - 60, 100, 40
        on_mouse_click.button_rect = (bx, by, bw, bh)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 200), -1)
        cv2.putText(frame, "STOP", (bx + 20, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frontend Simulation", frame)
        cv2.setMouseCallback("Frontend Simulation", on_mouse_click)
        
        if cv2.waitKey(1) == ord('q') or stop_button_pressed[0]: 
            break

    detector.close()
    if gaze_detector: gaze_detector.close()
    video.release()
    cv2.destroyAllWindows()
