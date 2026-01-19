# Pseudo-code for main.py
detector = FaceDetector()

@app.post("/detect")
def detect(image: UploadFile):
    frame = convert_image_to_cv2(image)
    result = detector.process_frame(frame)
    return result # Sends JSON back to React
