import cv2
import multiprocessing
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def gaze_worker(frame_queue, result_queue, max_faces):
    # Initialize MediaPipe FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=max_faces,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    while True:
        try:
            # Wait for frame
            try:
                frame_data = frame_queue.get(timeout=1)
            except:
                continue
            
            if frame_data is None: # Sentinel
                break
                
            rgb_frame = frame_data
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            try:
                # Perform Inference
                results = landmarker.detect(mp_image)
                
                status_message = "No Face Detected"
                
                if results.face_landmarks:
                    num_faces = len(results.face_landmarks)
                    
                    if num_faces > max_faces:
                        status_message = f"ERROR: Too many faces ({num_faces} > {max_faces})"
                    else:
                        face_landmarks = results.face_landmarks[0]
                        
                        # Convert normalized landmarks [0,1] to pixel coordinates
                        h, w, _ = rgb_frame.shape
                        mesh_points = np.array([[int(l.x * w), int(l.y * h)] for l in face_landmarks])
                        
                        # Indices from MediaPipe Face Mesh Canonical Model
                        # P33: Left Eye Inside, P133: Left Eye Outside, P468: Left Iris Center
                        # P362: Right Eye Inside, P263: Right Eye Outside, P473: Right Iris Center
                        
                        def get_ratio(p1, p2, iris_center):
                            """Calculates where the iris is between two eye corners (0.0 to 1.0)"""
                            full_dist = np.linalg.norm(p2 - p1)
                            if full_dist == 0: return 0.5
                            dist_to_p1 = np.linalg.norm(iris_center - p1)
                            return dist_to_p1 / full_dist

                        if len(mesh_points) > 473:
                            # --- HORIZONTAL GAZE ANALYSIS ---
                            p33, p133, p468 = mesh_points[33], mesh_points[133], mesh_points[468]
                            ratio_left_h = get_ratio(p33, p133, p468)
                            p362, p263, p473 = mesh_points[362], mesh_points[263], mesh_points[473]
                            ratio_right_h = get_ratio(p362, p263, p473)
                            avg_ratio_h = (ratio_left_h + ratio_right_h) / 2
                            
                            # --- VERTICAL GAZE ANALYSIS ---
                            p159, p145, p468_v = mesh_points[159], mesh_points[145], mesh_points[468]
                            dist_eye_v = np.linalg.norm(p159 - p145)
                            
                            ratio_left_v = get_ratio(p159, p145, p468_v)
                            p386, p374, p473_v = mesh_points[386], mesh_points[374], mesh_points[473]
                            ratio_right_v = get_ratio(p386, p374, p473_v)
                            avg_ratio_v = (ratio_left_v + ratio_right_v) / 2
                            
                            # --- THRESHOLDS ---
                            # Tuned for standard webcam distance ~50cm
                            SAFE_H_MIN, SAFE_H_MAX = 0.42, 0.58
                            SAFE_V_MIN, SAFE_V_MAX = 0.38, 0.62 
                            
                            is_blinking = dist_eye_v < (h * 0.012) # Blinking threshold

                            if avg_ratio_h < SAFE_H_MIN:
                                status_message = "WARNING: Looking Away (Right)"
                            elif avg_ratio_h > SAFE_H_MAX:
                                status_message = "WARNING: Looking Away (Left)"
                            elif is_blinking:
                                status_message = "Safe: Center (Blink)"
                            elif avg_ratio_v < SAFE_V_MIN:
                                status_message = "WARNING: Looking Away (Up)"
                            elif avg_ratio_v > SAFE_V_MAX:
                                status_message = "WARNING: Looking Away (Down)"
                            else:
                                status_message = "Safe: Center"
                        else:
                            status_message = "Face Mesh Limited"

                # Helper: Clear queue if full to always provide latest status
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except:
                        pass
                result_queue.put(status_message)

            except Exception as e:
                print(f"Gaze Worker Error: {e}")
                
        except Exception as e:
            print(f"Gaze Worker Error: {e}")

    landmarker.close()

# =============================================================================
# BACKEND API: GazeDetector Class
# =============================================================================
class GazeDetector:
    def __init__(self, max_faces=1):
        print("Initializing GazeDetector...")
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        
        self.worker = multiprocessing.Process(
            target=gaze_worker,
            args=(self.frame_queue, self.result_queue, max_faces)
        )
        self.worker.daemon = True
        self.worker.start()
        print("Gaze Worker started.")
        
    def process_frame(self, frame_bgr):
            # MediaPipe expects RGB
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Non-blocking put
            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)
                
            # Non-blocking get
            try:
                return self.result_queue.get_nowait()
            except:
                return None # Return None if no NEW result available
        except Exception as e:
            print(f"Gaze API Error: {e}")
            return None
            
    def close(self):
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()
