import os
import shutil
import logging
import time
import uuid
import threading
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import mimetypes
from datetime import datetime
from Auth.auth import router as auth_router
from middleware.middleware import AuthMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form , Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from subtitle_correction import write_subtitle_file
from models import RepositionRequest, GenerationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants - These are fine as globals since they're immutable
MAX_FILE_SIZE = 1000 * 1024 * 1024  # 100MB
MAX_WORKERS = 2
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
ALLOWED_SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Directory setup - Calculate once at startup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

class AppState:
    """Thread-safe application state container."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.task_manager = None
        self.executor = None
        self.config = None
        self._initialized = False
    
    def initialize(self, config: 'AppConfig', task_manager: 'TaskManager', executor: ThreadPoolExecutor):
        """Initialize app state - called once at startup"""
        with self._lock:
            if self._initialized:
                return
            self.config = config
            self.task_manager = task_manager
            self.executor = executor
            self._initialized = True
    
    def get_task_manager(self) -> 'TaskManager':
        """Thread-safe access to task manager"""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("App state not initialized")
            return self.task_manager
    
    def get_executor(self) -> ThreadPoolExecutor:
        """Thread-safe access to executor"""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("App state not initialized")
            return self.executor
    
    def get_config(self) -> 'AppConfig':
        """Thread-safe access to config"""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("App state not initialized")
            return self.config
    
    async def shutdown(self):
        """Cleanup resources"""
        with self._lock:
            if self.executor:
                self.executor.shutdown(wait=True)
            self._initialized = False

# Global app state - properly managed singleton
app_state = AppState()

class AppConfig:
    """Application configuration."""
    def __init__(self):
        self.upload_dir = UPLOAD_DIR
        self.output_dir = OUTPUT_DIR
        self.max_file_size = MAX_FILE_SIZE
        self.max_workers = MAX_WORKERS
        self.allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000","http://<ipv4_address>:3000","http://<ipv4_address>:3000"]
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = False

        # Create directories if they do not exist.
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

# Global app config
app_config = AppConfig()

# Task management with thread safety
class TaskManager:
    """Thread-safe manager for background tasks."""
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._websocket_connections: Dict[str, WebSocket] = {}  # task_id -> websocket
        self._update_queue = []  # Queue for pending WebSocket updates
        self._queue_lock = threading.Lock()
    
    def create_task(self, task_type: str) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "type": task_type,
                "status": "pending",
                "message": "Task created",
                "progress": 0,
                "start_time": time.time(),
                "created_at": time.time()
            }
        return task_id
    
    def update_task(self, task_id: str, **kwargs) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)
                # Queue WebSocket update if connection exists
                if task_id in self._websocket_connections:
                    with self._queue_lock:
                        self._update_queue.append((task_id, dict(self._tasks[task_id])))
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._tasks.get(task_id)
    
    def add_websocket_connection(self, task_id: str, websocket: WebSocket) -> None:
        with self._lock:
            self._websocket_connections[task_id] = websocket
    
    def remove_websocket_connection(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._websocket_connections:
                del self._websocket_connections[task_id]
    
    async def send_websocket_update(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Send update via WebSocket if connection exists."""
        if task_id in self._websocket_connections:
            try:
                websocket = self._websocket_connections[task_id]
                await websocket.send_json(task_data)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update for task {task_id}: {e}")
                # Remove broken connection
                self.remove_websocket_connection(task_id)
    
    def get_pending_updates(self) -> List[tuple]:
        """Get and clear pending WebSocket updates."""
        with self._queue_lock:
            updates = list(self._update_queue)
            self._update_queue.clear()
            return updates
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up tasks older than max_age_hours."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        with self._lock:
            old_tasks = [
                task_id for task_id, task in self._tasks.items()
                if task.get("created_at", 0) < cutoff_time
            ]
            for task_id in old_tasks:
                del self._tasks[task_id]
                # Also clean up websocket connections
                if task_id in self._websocket_connections:
                    del self._websocket_connections[task_id]
            return len(old_tasks)

# Dependency functions for FastAPI
def get_app_config() -> AppConfig:
    """Dependency to get app config."""
    return app_state.get_config()

def get_task_manager() -> TaskManager:
    """Dependency to get task manager."""
    return app_state.get_task_manager()

def get_executor() -> ThreadPoolExecutor:
    """Dependency to get executor."""
    return app_state.get_executor()

# Lifespan context manager for proper resource management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle and resources."""
    logger.info("Starting Subtitle Synchronizer API")
    
    # Initialize global state
    task_manager = TaskManager()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    app_state.initialize(app_config, task_manager, executor)
    
    # Start WebSocket update handler
    websocket_task = asyncio.create_task(websocket_update_handler(task_manager))
    
    try:
        yield
    finally:
        logger.info("Shutting down Subtitle Synchronizer API")
        websocket_task.cancel()
        await app_state.shutdown()
        executor.shutdown(wait=False)  # Non-blocking shutdown

async def websocket_update_handler(task_manager: TaskManager):
    """Background task to handle WebSocket updates."""
    while True:
        try:
            # Get pending updates
            updates = task_manager.get_pending_updates()
            
            # Send each update
            for task_id, task_data in updates:
                await task_manager.send_websocket_update(task_id, task_data)
            
            # Sleep briefly to avoid busy waiting
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in WebSocket update handler: {e}")
            await asyncio.sleep(1)  # Sleep longer on error


# Dependency for file validation
async def validate_file_upload(file: UploadFile, allowed_extensions: set, max_size: int = MAX_FILE_SIZE):
    """Validate the uploaded file for extension and size."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension using mimetypes
    mime_type, _ = mimetypes.guess_type(file.filename)
    if not mime_type or Path(file.filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(allowed_extensions)}"
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size // (1024 * 1024)}MB"
        )

    # Cache file size in metadata for reuse
    file.metadata = {"size": file_size}
    return file

# Safe file operations
def safe_save_file(file: UploadFile, directory: Path) -> Path:
    """Safely save the uploaded file with proper error handling."""
    try:
        file_path = directory / file.filename
        # Prevent overwriting by adding a timestamp if the file exists.
        if file_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            timestamp = datetime.now().isoformat().replace(":", "-")  # Use ISO timestamp
            file_path = directory / f"{stem}_{timestamp}{suffix}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def safe_import_module(module_name: str, file_path: str):
    """Safely import a module with proper error handling."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {module_name} from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to import {module_name}: {str(e)}")
        raise

# Try to import required modules
try:
    from subtitle_correction import main as correct_subtitles_main
    from subtitle_generation_combined import generate_subtitles, write_subtitles

    subtitle_sync_module = safe_import_module("subtitle_sync", "subtitle_correction.py")
    SubtitleSynchronizer = subtitle_sync_module.SubtitleSynchronizer
    
    reposition_module = safe_import_module("reposition_subtitle", "reposition_subtitle.py")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    raise

# FastAPI app with proper configuration
app = FastAPI(
    title="Subtitle Synchronizer API",
    description="API for synchronizing subtitle files with video audio tracks",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(AuthMiddleware)
app.include_router(auth_router, prefix="/auth", tags=["Auth"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Subtitle Synchronizer API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "upload_files": "/upload-files/",
            "generate_subtitles": "/generate/",
            "sync_subtitles": "/correct_subtitles_async/",
            "get_status": "/sync_status/{task_id}",
            "websocket_status": "/ws/task/{task_id}",
            "download_result": "/download/{filename}",
            "list_files": "/files/",
            "docs": "/docs"
        }
    }

# WebSocket endpoint for real-time task updates
@app.websocket("/ws/task/{task_id}")
async def websocket_task_updates(
    websocket: WebSocket, 
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """WebSocket endpoint for real-time task status updates."""
    await websocket.accept()
    
    try:
        # Check if task exists
        task = task_manager.get_task(task_id)
        if not task:
            await websocket.send_json({"error": "Task not found"})
            await websocket.close()
            return
        
        # Register this WebSocket connection for the task
        task_manager.add_websocket_connection(task_id, websocket)
        
        # Send current task status immediately
        await websocket.send_json(task)
        
        # Keep the connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client (heartbeat, etc.)
                message = await websocket.receive_text()
                
                # Send current task status if requested
                if message == "get_status":
                    current_task = task_manager.get_task(task_id)
                    if current_task:
                        await websocket.send_json(current_task)
                    
                # If task is completed or failed, we can close the connection
                current_task = task_manager.get_task(task_id)
                if current_task and current_task.get("status") in ["completed", "failed"]:
                    # Send final status and close
                    await websocket.send_json(current_task)
                    break
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        # Clean up the WebSocket connection
        task_manager.remove_websocket_connection(task_id)
        try:
            await websocket.close()
        except:
            pass


# API endpoint for uploading video only (for generation)
@app.post("/upload-video/")
async def upload_video(
    request:Request,
    video_file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
):
    """Upload video file for subtitle generation."""
    try:
        user_id = request.state.user.get("id",None)
        upload_dir = config.upload_dir / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        await validate_file_upload(video_file, ALLOWED_VIDEO_EXTENSIONS)
        video_path = safe_save_file(video_file, upload_dir)
        
        return {
            "message": "Video uploaded successfully",
            "filename": video_file.filename,
            "size": video_path.stat().st_size
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

# API endpoint for uploading files
@app.post("/upload-files/")
async def upload_files(
    request : Request,
    video_file: UploadFile = File(...),
    subtitle_file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
    
):
    """Upload video and subtitle files with validation."""
    try:
        user_id = request.state.user.get("id",None)
        upload_dir = config.upload_dir / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Validate files first
        await validate_file_upload(video_file, ALLOWED_VIDEO_EXTENSIONS)
        await validate_file_upload(subtitle_file, ALLOWED_SUBTITLE_EXTENSIONS)
        
        video_path = safe_save_file(video_file, upload_dir)
        subtitle_path = safe_save_file(subtitle_file, upload_dir)
        
        return {
            "message": "Files uploaded successfully",
            "video_file": video_file.filename,
            "subtitle_file": subtitle_file.filename,
            "video_size": video_path.stat().st_size,
            "subtitle_size": subtitle_path.stat().st_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# API endpoint for subtitle generation
@app.post("/generate/") 
async def start_generation(
    user_data : Request,
    request: GenerationRequest, 
    background_tasks: BackgroundTasks,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Start subtitle generation task."""
    try:
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = UPLOAD_DIR / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / request.video_filename
        if not video_path.exists():
            raise HTTPException(404, f"Video file not found: {request.video_filename}")
        
        if request.output_filename:
            output_filename = request.output_filename
        else:
            base_name = Path(request.video_filename).stem
            output_filename = f"{base_name}_subtitles.{request.subtitle_format}"
        
        task_id = task_manager.create_task("generation")
        
        def run_generation_task():
            try:
                start_time = time.time()

                task_manager.update_task(task_id, status="processing", message="Extracting audio and detecting language...", progress=10)

                # Call the refined generate_subtitles function
                task_manager.update_task(task_id, message="Transcribing audio in chunks...", progress=30)
                
                # Pass translation_engine and translation_credentials to generate_subtitles
                final_segments = generate_subtitles(
                    str(video_path),
                    subtitle_lang=request.subtitle_lang,
                    translation_engine=request.translation_engine,
                    translation_credentials=request.translation_credentials
                )

                task_manager.update_task(task_id, progress=80, message="Writing subtitle file...")

                # Write the subtitles using the write_subtitles function
                output_path = output_dir / output_filename
                write_subtitles(final_segments, request.subtitle_format, str(output_path))
                logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

                task_manager.update_task(task_id, 
                    progress=100,
                    status="completed",
                    message="Subtitles generated successfully",
                    result_file=output_filename,
                    processing_time=time.time() - task_manager.get_task(task_id)["start_time"],
                    segments_count=len(final_segments)
                )
            except Exception as e:
                task_manager.update_task(task_id,
                    status="failed",
                    message=str(e),
                    processing_time=time.time() - task_manager.get_task(task_id)["start_time"]
                )

        threading.Thread(target=run_generation_task, daemon=True).start()
        return {
            "task_id": task_id,
            "message": "Subtitle generation task started",
            "output_filename": output_filename,
            "subtitle_format": request.subtitle_format,
            "subtitle_lang": request.subtitle_lang
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Exception occured :{e}")
        raise HTTPException(500, f"Failed to start generation: {str(e)}")

# API endpoints for checking generation status 
@app.get("/generate/status/{task_id}")
async def get_generation_status(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get generation task status."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

# API endpoint for getting generation result
@app.get("/generate/result/{task_id}")
async def get_generation_result(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get generation task result."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed")
    return {
        "success": True,
        "message": task["message"],
        "output_file": task.get("result_file"),
        "processing_time": task.get("processing_time", 0),
        "segments_count": task.get("segments_count", 0)
    }

# API endpoint for subtitle correction (async with progress tracking)
@app.post("/correct_subtitles_async/")
async def correct_subtitles_async(
    user_data: Request,
    video: UploadFile = File(...), 
    subtitle: UploadFile = File(...),
    translation_engine: str = Form("m2m100"),
    translation_credentials: Optional[str] = Form(None),
    task_manager: TaskManager = Depends(get_task_manager),
    config: AppConfig = Depends(get_app_config)
):
    """Start async subtitle correction with progress tracking."""
    try:
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = config.upload_dir / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = config.output_dir / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Validate files
        await validate_file_upload(video, ALLOWED_VIDEO_EXTENSIONS)
        await validate_file_upload(subtitle, ALLOWED_SUBTITLE_EXTENSIONS)
        
        # Generate task ID using injected TaskManager
        task_id = task_manager.create_task("correction")
        
        # Save uploaded files
        video_path = safe_save_file(video, upload_dir)
        subtitle_path = safe_save_file(subtitle, upload_dir)

        # Parse credentials if provided
        credentials = None
        if translation_credentials:
            import json
            try:
                credentials = json.loads(translation_credentials)
            except Exception:
                credentials = None
        
        # Start background task with dependency injection
        def run_sync_task():
            try:
                task_manager.update_task(task_id, status="processing", message="Parsing subtitles...", progress=10, current_step="Parsing subtitle file")
                task_manager.update_task(task_id, progress=20, current_step="Detecting subtitle language")
                task_manager.update_task(task_id, progress=30, current_step="Extracting audio from video")
                task_manager.update_task(task_id, progress=40, current_step="Loading Whisper model")
                task_manager.update_task(task_id, progress=50, current_step="Transcribing audio (this may take a while)")
                
                # Call the actual correction function
                result = correct_subtitles_main(
                    str(video_path),
                    str(subtitle_path),
                    output_dir=str(output_dir),
                    translation_engine=translation_engine,
                    translation_credentials=credentials
                )
                
                task_manager.update_task(task_id, progress=80, current_step="Aligning subtitles with transcript")
                
                if result["success"] and result["output_file"]:
                    output_filename = result["output_file"]
                    
                    # Search for the file in multiple possible locations
                    search_paths = [
                        output_filename,  # Current working directory
                        os.path.join(os.getcwd(), output_filename),
                        os.path.join(str(upload_dir), output_filename),
                        os.path.join(str(output_dir), output_filename),  # Maybe it's already there
                    ]
                    
                    source_path = None
                    for path in search_paths:
                        if os.path.exists(path):
                            source_path = path
                            logger.info(f"Found file at: {path}")
                            break
                    
                    if source_path:
                        destination_path = output_dir / output_filename
                        if Path(source_path) != destination_path:
                            if destination_path.exists():
                                destination_path.unlink()
                            shutil.move(source_path, destination_path)
                            logger.info(f"✅ Moved file: {source_path} → {destination_path}")
                        else:
                            logger.info(f"✅ File already in correct location: {destination_path}")
                    else:
                        logger.error(f"❌ ERROR: Could not find {output_filename} in any expected location")

                task_manager.update_task(task_id, progress=90, current_step="Writing corrected subtitle file")
                
                # Task completed
                start_time = task_manager.get_task(task_id)["start_time"]
                task_manager.update_task(
                    task_id, 
                    status="completed", 
                    progress=100, 
                    current_step="Synchronization completed", 
                    message=result["message"], 
                    success=result["success"], 
                    output_file=result["output_file"], 
                    processing_time=time.time() - start_time
                )
                
            except Exception as e:
                start_time = task_manager.get_task(task_id)["start_time"]
                task_manager.update_task(
                    task_id, 
                    status="failed", 
                    message=str(e), 
                    current_step=f"Error: {str(e)}", 
                    processing_time=time.time() - start_time
                )
        
        # Start the task in a separate thread
        threading.Thread(target=run_sync_task, daemon=True).start()
        
        return {
            "task_id": task_id,
            "message": "Synchronization task started",
            "status": "started"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start sync task: {str(e)}"}
        )

# API endpoints for checking sync status
@app.get("/sync_status/{task_id}")
async def get_sync_status(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get synchronization task status and progress."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

# API endpoint for getting sync result
@app.get("/sync_result/{task_id}")
async def get_sync_result(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get synchronization task result."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    return {
        "success": task.get("success", False),
        "message": task.get("message", ""),
        "output_file": task.get("output_file", ""),
        "processing_time": task.get("processing_time", 0)
    }

# API endpoint for repositioning subtitles
@app.post("/reposition/", response_model=Dict[str, str])
async def reposition_subtitles(
    user_data: Request,
    request: RepositionRequest, 
    background_tasks: BackgroundTasks,
    executor: ThreadPoolExecutor = Depends(get_executor)
):
    """Start subtitle repositioning task (after sync)."""
    try:
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = UPLOAD_DIR / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / request.video_filename
        subtitle_path_outputs = output_dir / request.subtitle_filename
        subtitle_path_uploads = upload_dir / request.subtitle_filename
        print("before checking the video path",video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")
        print(f"before checking subtitle paht :{subtitle_path_outputs}")
        if subtitle_path_outputs.exists():
            subtitle_path = subtitle_path_outputs
        elif subtitle_path_uploads.exists():
            subtitle_path = subtitle_path_uploads
        else:
            raise HTTPException(status_code=404, detail=f"Subtitle file not found: {request.subtitle_filename}")

        # Run repositioning in background
        def run_reposition():
            print("starting run_reposition")
            try:
                output_file = reposition_module.process_subtitle(str(video_path), str(subtitle_path))
                output_path = output_dir / Path(output_file).name
                if Path(output_file) != output_path:
                    # Always overwrite the output file in OUTPUT_DIR
                    if output_path.exists():
                        output_path.unlink()  # Remove old file
                    Path(output_file).rename(output_path)  # Use Path.rename for moving
                return output_path.name
            except Exception as e:
                raise e
        print("reached loop")
        loop = asyncio.get_running_loop()
        output_filename = await loop.run_in_executor(executor, run_reposition)
        print("reached output filename: ",output_filename)
        if output_filename:
            return {
                "success": "true",
                "message": "Subtitle repositioning completed",
                "output_filename": output_filename
            }
        else:
            raise HTTPException(status_code=500, detail="Subtitle repositioning failed")
    except Exception as e:
        print("Exception during repositioning: ",e)
        raise HTTPException(status_code=500, detail=f"Error during repositioning: {str(e)}")

# API endpoint for downloading result files
@app.get("/download/{filename}")
async def download_file(user_data: Request,filename: str):
    """Download a synchronized subtitle file."""
    # Validate filename to prevent path traversal.
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    user_id = user_data.state.user.get('id')
    file_path = OUTPUT_DIR / user_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

# API endpoint for listing available models
@app.get("/models/")
async def get_available_models():
    """Get available Whisper models."""
    models = [
        {
            "name": "tiny",
            "description": "Fastest, English-only (~39 MB)",
            "languages": "English only"
        },
        {
            "name": "base", 
            "description": "Good balance of speed and accuracy (~74 MB)",
            "languages": "Multilingual"
        },
        {
            "name": "small",
            "description": "Better accuracy, slower (~244 MB)", 
            "languages": "Multilingual"
        },
        {
            "name": "medium",
            "description": "High accuracy, slower (~769 MB)",
            "languages": "Multilingual" 
        },
        {
            "name": "large",
            "description": "Best accuracy, slowest (~1550 MB)",
            "languages": "Multilingual"
        }
    ]
    
    return {
        "models": models,
        "supported_formats": ["srt", "vtt"],
        "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
    }

# API endpoint for listing available files
@app.get("/files/")
async def list_files(user_data: Request):
    """List available files in upload and output directories."""
    try:
        # Use os.scandir for efficient directory traversal
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = UPLOAD_DIR / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        upload_files = [entry.name for entry in os.scandir(upload_dir) if entry.is_file()]
        output_files = [entry.name for entry in os.scandir(output_dir) if entry.is_file()]
        
        return {
            "upload_files": upload_files,
            "output_files": output_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
    
# API endpoint for updating file
@app.post("/update-file")
async def update_file(request: Request):
    data = await request.json()
    filename = data.get("filename")
    content = data.get("content")
    user_id = request.state.user.get("id",None)
    output_path = OUTPUT_DIR / user_id / filename
    format = Path(filename).suffix.lstrip(".")
    converted = [
    {
        "index": c["index"],
        "start": c["startSeconds"],
        "end": c["endSeconds"],
        "text": c["text"],
        "format": format
    }
    for c in content
    ]
    write_subtitle_file(converted, output_path)

@app.get("/get-file-content/{filename}", response_class=PlainTextResponse)
def get_file_content(filename: str, response: Request):
    user_id = response.state.user.get("id", None)
    file_path = os.path.join(OUTPUT_DIR / user_id / filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        return f.read()

# API endpoint for deleting files
@app.delete("/files/{filename}")
async def delete_file(user_data: Request ,filename: str, file_type: str = "upload"):
    """Delete a file from the upload or output directory."""
    try:
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = UPLOAD_DIR / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Validate filename to prevent path traversal.
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        if file_type == "upload":
            file_path = upload_dir / filename
        elif file_type == "output":
            file_path = output_dir / filename
        else:
            raise HTTPException(status_code=400, detail="file_type must be 'upload' or 'output'")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
        return {"message": f"File {filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# API endpoint for cleaning up files
@app.delete("/cleanup/")
async def cleanup_files(user_data: Request, task_manager: TaskManager = Depends(get_task_manager)):
    """Clean up all uploaded and output files."""
    try:
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = UPLOAD_DIR / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Batch delete files to reduce I/O overhead
        upload_files = list(upload_dir.iterdir())
        output_files = list(output_dir.iterdir())

        for file_path in upload_files + output_files:
            if file_path.is_file():
                file_path.unlink()

        # Clear task status and results
        task_manager.cleanup_old_tasks(0)  # Clear all tasks
        
        return {
            "message": "Cleanup completed",
            "upload_files_deleted": len(upload_files),
            "output_files_deleted": len(output_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

# API endpoint for health check
@app.get("/health/")
async def health_check(user_data: Request, config: AppConfig = Depends(get_app_config)):
    """Enhanced health check endpoint."""
    try:
        task_manager = app_state.get_task_manager()
        user_id = user_data.state.user.get("id",None)
        print(user_id)
        upload_dir = config.upload_dir / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir = config.output_dir / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # Optimize disk usage check
        upload_space, output_space = map(
            shutil.disk_usage, [upload_dir, output_dir]
        )
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "active_tasks": len([t for t in task_manager._tasks.values() if t.get("status") == "processing"]),
            "total_tasks": len(task_manager._tasks),
            "websocket_connections": len(task_manager._websocket_connections),
            "disk_space": {
                "upload_dir_free_gb": upload_space.free / (1024**3),
                "output_dir_free_gb": output_space.free / (1024**3)
            },
            "directories": {
                "upload_files": len(list(upload_dir.glob("*"))),
                "output_files": len(list(output_dir.glob("*")))
            },
            "features": {
                "websocket_support": True,
                "real_time_updates": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}



if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Subtitle Synchronizer API server...")
    logger.info("Available at: http://localhost:8000")
    logger.info("API docs at: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
        log_level="info"
    )