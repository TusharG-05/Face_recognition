import cv2
import multiprocessing
import numpy as np
import mediapipe as mp
import time

def gaze_worker(frame_queue, result_queue):
    mp_face_mesh = mp.solutions.face_mesh
    # refine_landmarks=True gives us iris landmarks (468: Left Iris, 473: Right Iris)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        try:
            frame_data = frame_queue.get(timeout=1)
        except:
            continue
        
        if frame_data is None:
            break
            
        rgb_frame = frame_data
        
        try:
            results = face_mesh.process(rgb_frame)
            
            gaze_status = "Unknown"
            
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [rgb_frame.shape[1], rgb_frame.shape[0]]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                # Indices for Left Eye
                # 33: Left Corner, 133: Right Corner, 468: Iris Center
                # Indices for Right Eye
                # 362: Left Corner, 263: Right Corner, 473: Iris Center
                
                # We can average the ratio for both eyes or just use one. Using both is more robust.
                
                def get_ratio(eye_points, iris_center):
                    # eye_points = [left_corner, right_corner]
                    lc = eye_points[0]
                    rc = eye_points[1]
                    
                    # Horizontal Ratio
                    # dist_total = rc[0] - lc[0]
                    # dist_iris = iris_center[0] - lc[0]
                    # ratio = dist_iris / dist_total
                    
                    # A better metric might be relative position
                    # Center is 0.5
                    
                    full_width = np.linalg.norm(rc - lc)
                    dist_to_left = np.linalg.norm(iris_center - lc)
                    
                    ratio_x = dist_to_left / full_width
                    return ratio_x

                # Left Eye Analysis
                p33 = mesh_points[33]
                p133 = mesh_points[133]
                p468 = mesh_points[468]
                
                ratio_left = get_ratio([p33, p133], p468)
                
                # Right Eye Analysis
                p362 = mesh_points[362]
                p263 = mesh_points[263]
                p473 = mesh_points[473]
                
                ratio_right = get_ratio([p362, p263], p473)
                
                avg_ratio = (ratio_left + ratio_right) / 2
                
                # These thresholds might need tuning based on camera and distance
                if avg_ratio < 0.40:
                    gaze_status = "Looking Right" # Mirrored logic usually: Iris closer to detector's right (subject's left) means looking right? 
                    # Actually:
                    # If iris is closer to point 33 (left corner of left eye), ratio is small.
                    # 33 is the outer corner of left eye. 
                    # If iris is close to outer corner of left eye, person is looking to their left (Detector's Right).
                    gaze_status = "Right"
                elif avg_ratio > 0.60:
                    gaze_status = "Left"
                else:
                    gaze_status = "Center"
            
            if not result_queue.full():
                result_queue.put(gaze_status)
                
        except Exception as e:
            print(f"Gaze Worker Error: {e}")

    face_mesh.close()

class GazeDetector:
    def __init__(self):
        print("Initializing GazeDetector...")
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        
        self.worker = multiprocessing.Process(
            target=gaze_worker,
            args=(self.frame_queue, self.result_queue)
        )
        self.worker.daemon = True
        self.worker.start()
        print("Gaze Worker started.")
        
    def process_frame(self, frame_bgr):
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Performance optimization for MediaPipe
        
        if not self.frame_queue.full():
            self.frame_queue.put(frame_rgb)
            
        try:
            return self.result_queue.get_nowait()
        except:
            return None
            
    def close(self):
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()
