import os
import signal
import threading
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ..services.camera import CameraService

router = APIRouter()
camera_service = CameraService()

@router.post("/upload-identity")
async def upload_identity(file: UploadFile = File(...)):
    """
    Uploads a new image for the 'Known Person'.
    Reloads the FaceDetector with the new identity.
    """
    try:
        content = await file.read()
        success = camera_service.update_identity(content)
        
        if success:
            return JSONResponse(content={
                "message": "Identity updated successfully", 
                "filename": file.filename
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to reload detector with new image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown")
def shutdown_server():
    """
    Gracefully shuts down the server by sending SIGINT to the process.
    """
    def kill_self():
        time.sleep(1) # Give time for response to send
        os.kill(os.getpid(), signal.SIGINT)
        
    threading.Thread(target=kill_self).start()
    return {"message": "Server shutting down..."}
