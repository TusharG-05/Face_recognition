import contextlib
import os
from fastapi import FastAPI
from .routers import video, site, settings
from .services.camera import CameraService

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting Application...")
    service = CameraService()
    # service.start() is now lazy-loaded in video.py
    # to prevent camera access until frontend is opened.
    
    yield
    
    # Shutdown
    print("Stopping CameraService...")
    service.stop()

app = FastAPI(lifespan=lifespan)

app.include_router(site.router)
app.include_router(video.router)
app.include_router(settings.router)


