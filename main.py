import uvicorn
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Ensure spawn method is used (default on Windows, but good to be explicit for stability)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    print("Starting Server in STABLE mode (Reload Disabled for camera stability)...")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
