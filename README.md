# Face & Gaze Recognition Web App

A robust, FastAPI-based application for real-time face authentication and gaze tracking.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam

### Installation
1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
```bash
python main.py
```
The server will start at `http://localhost:8000`.

---

## 📂 Project Structure

```
├── app/
│   ├── assets/              # Models & Images
│   │   ├── face_landmarker.task
│   │   └── known_person.jpg  <-- The authorized face
│   ├── routers/             # API Endpoints
│   │   ├── video.py         # /video_feed stream
│   │   ├── settings.py      # /upload-identity
│   │   └── site.py          # / (Index HTML)
│   ├── services/            # Business Logic
│   │   ├── camera.py        # Singleton Camera Manager
│   │   ├── face.py          # Face Detection Worker
│   │   └── gaze.py          # Gaze Detection Worker
│   └── templates/           # Frontend
│       └── index.html
│   └── server.py            # FastAPI App Definition
├── main.py                  # Entry Point
└── requirements.txt
```

---

## 👨‍💻 For Frontend Developers

### 1. Video Stream (`GET /video_feed`)
The video feed is served as an **MJPEG Stream**.
- **URL**: `http://localhost:8000/video_feed`
- **Format**: `multipart/x-mixed-replace`
- **Usage**:
  ```html
  <img src="/video_feed" alt="Live Stream" />
  ```
- **Note**: The red/green bounding boxes and status text are currently drawn **server-side** onto the image frames.

### 2. Upload Identity (`POST /upload-identity`)
Endpoint to update the "Authorized Person" without restarting the server.
- **URL**: `http://localhost:8000/upload-identity`
- **Method**: `POST`
- **Body**: `FormData` with a key `file` containing the image.
- **Response**: JSON `{ "message": "Identity updated successfully", ... }`

**Example Fetch:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

await fetch('/upload-identity', {
    method: 'POST',
    body: formData
});
```

---

## 🔧 Configuration
- **Camera Source**: Defaults to Webcam (`0`).
- **Models**:
    - **Face**: Uses `face_recognition` (dlib).
    - **Gaze**: Uses MediaPipe `face_landmarker.task` located in `app/assets/`.
