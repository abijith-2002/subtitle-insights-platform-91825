import os
import shutil
import logging
import time
import uuid
import threading
import asyncio
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import mimetypes
from datetime import datetime, timedelta, timezone

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Depends,
    Form,
    Request,
    Header,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, EmailStr

# Third-party security/auth imports
import bcrypt  # Password hashing (ensure your DB stores bcrypt hashes)
import jwt as pyjwt  # PyJWT for token encode/decode
from bson import ObjectId  # Mongo ObjectId

# Async MongoDB driver
try:
    import motor.motor_asyncio as motor_asyncio
except Exception as e:
    # If motor is not installed, FastAPI app will start but auth routes will throw 500 with a helpful message.
    motor_asyncio = None

from models import RepositionRequest, GenerationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration and constants
# -----------------------------------------------------------------------------
# Constants - These are fine as globals since they're immutable
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_WORKERS = 2
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
ALLOWED_SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Directory setup - Calculate once at startup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Environment variables used by the authentication layer.
# Note: These must be set in your environment (e.g., .env file or deployment config).
# - MONGODB_URI: MongoDB connection string (e.g., mongodb+srv://user:pass@cluster/dbname)
# - JWT_SECRET: Secret used to sign JWTs (keep secure; do not commit)
# - JWT_EXPIRES_IN: Access token lifetime in seconds (default 900)
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_IN = int(os.getenv("JWT_EXPIRES_IN", "900"))

# -----------------------------------------------------------------------------
# App state and configuration
# -----------------------------------------------------------------------------
class AppState:
    """Thread-safe application state container."""

    def __init__(self):
        self._lock = threading.RLock()
        self.task_manager = None
        self.executor = None
        self.config = None
        self._initialized = False

        # Mongo-related runtime attributes
        self.mongo_client = None
        self.mongo_db = None
        self.users_collection = None
        self.blacklist_collection = None

    def initialize(self, config: 'AppConfig', task_manager: 'TaskManager', executor: ThreadPoolExecutor):
        """Initialize app state - called once at startup"""
        with self._lock:
            if self._initialized:
                return
            self.config = config
            self.task_manager = task_manager
            self.executor = executor
            self._initialized = True

    def set_mongo(self, client, db, users_collection, blacklist_collection):
        """Set MongoDB client and useful collections after connection is established."""
        with self._lock:
            self.mongo_client = client
            self.mongo_db = db
            self.users_collection = users_collection
            self.blacklist_collection = blacklist_collection

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

    def get_users_collection(self):
        """Get Mongo users collection or raise if not configured"""
        with self._lock:
            return self.users_collection

    def get_blacklist_collection(self):
        """Get Mongo blacklist collection or raise if not configured"""
        with self._lock:
            return self.blacklist_collection

    async def shutdown(self):
        """Cleanup resources"""
        with self._lock:
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.mongo_client:
                try:
                    self.mongo_client.close()
                except Exception:
                    pass
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
        self.allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
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

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._tasks.get(task_id)

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

# -----------------------------------------------------------------------------
# Auth models and helpers
# -----------------------------------------------------------------------------
class LoginRequest(BaseModel):
    """Login payload."""
    email: EmailStr = Field(..., description="User email address (must exist in MongoDB users collection)")
    password: str = Field(..., min_length=1, description="User password (bcrypt-hashed in DB)")

class TokenResponse(BaseModel):
    """Access token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Token lifetime in seconds")

class UserPublic(BaseModel):
    """Public user representation."""
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")

# PUBLIC_INTERFACE
def create_access_token(subject: str, email: str, expires_in: Optional[int] = None) -> Tuple[str, str, datetime]:
    """Create a JWT access token.

    Args:
        subject: The subject (typically user id) as a string.
        email: Email for convenience/claims.
        expires_in: Optional seconds to live; defaults to JWT_EXPIRES_IN.

    Returns:
        tuple[token, jti, expires_at]
    """
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfiguration: JWT_SECRET not set")
    lifetime = int(expires_in or JWT_EXPIRES_IN)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=lifetime)
    jti = str(uuid.uuid4())
    payload = {
        "sub": subject,
        "email": email,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, jti, exp

# PUBLIC_INTERFACE
async def get_jwt_payload(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Extract and validate JWT from Authorization header.

    Validates:
    - Presence and Bearer scheme
    - Signature and expiration
    - Token not present in blacklist

    Returns:
        The token payload dict.

    Raises:
        HTTPException on missing/invalid/expired/revoked token.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()

    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfiguration: JWT_SECRET not set")
    if app_state.get_blacklist_collection() is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: MongoDB not configured")

    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # Check blacklist
    blacklist = app_state.get_blacklist_collection()
    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token (missing jti)")

    exists = await blacklist.find_one({"jti": jti})
    if exists:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    return payload

# PUBLIC_INTERFACE
async def get_current_user(payload: Dict[str, Any] = Depends(get_jwt_payload)) -> UserPublic:
    """Resolve the current user from a valid JWT payload."""
    users = app_state.get_users_collection()
    if users is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: MongoDB not configured")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token (missing subject)")
    try:
        doc = await users.find_one({"_id": ObjectId(sub)})
    except Exception:
        doc = None

    if not doc:
        # User was deleted or not found
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return UserPublic(id=str(doc.get("_id")), email=doc.get("email"))

# -----------------------------------------------------------------------------
# Lifespan context manager for proper resource management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle and resources.

    - Initializes task manager/executor
    - Connects to MongoDB if MONGODB_URI is set
    - Ensures indexes for users and token blacklist (including TTL on expiresAt)
    """
    logger.info("Starting Subtitle Synchronizer API")

    # Initialize global state
    task_manager = TaskManager()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    app_state.initialize(app_config, task_manager, executor)

    # Initialize MongoDB if available
    try:
        if MONGODB_URI and motor_asyncio is not None:
            client = motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
            db = client.get_default_database() or client["subtitle_auth"]
            users_collection = db["users"]
            blacklist_collection = db["token_blacklist"]

            # Helpful indexes
            await users_collection.create_index("email", unique=True)
            await blacklist_collection.create_index("jti", unique=True)
            # TTL index uses expiresAt with expireAfterSeconds=0 to expire at exact timestamp
            await blacklist_collection.create_index(
                "expiresAt",
                expireAfterSeconds=0
            )

            app_state.set_mongo(client, db, users_collection, blacklist_collection)
            logger.info("MongoDB connected and indexes ensured")
        else:
            if not MONGODB_URI:
                logger.warning("MONGODB_URI not set - auth endpoints will not function")
            if motor_asyncio is None:
                logger.warning("motor not installed - auth endpoints will not function")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {str(e)}")
        # Do not crash the app; allow non-auth endpoints to work
        app_state.set_mongo(None, None, None, None)

    try:
        yield
    finally:
        logger.info("Shutting down Subtitle Synchronizer API")
        await app_state.shutdown()
        executor.shutdown(wait=False)  # Non-blocking shutdown

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Subtitle Synchronizer API",
    description="API for synchronizing subtitle files with video audio tracks and authentication endpoints",
    version="1.0.1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Auth endpoints
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Login and obtain an access token",
    tags=["Auth"],
    responses={
        200: {"description": "Login successful; access token returned"},
        401: {"description": "Invalid credentials"},
        500: {"description": "Server misconfiguration/internals"},
    },
)
async def auth_login(payload: LoginRequest):
    """Authenticate user via email and password and issue a JWT.

    Request body:
    - email: user email stored in MongoDB (users collection)
    - password: user's password in plaintext; will be verified against stored bcrypt hash

    Returns:
    - access_token: signed JWT (HS256) with jti, sub (user id), email, iat, exp
    - token_type: "bearer"
    - expires_in: token lifetime in seconds

    Notes:
    - Ensure environment variables MONGODB_URI and JWT_SECRET are set.
    - Passwords in Mongo should be stored as bcrypt hashes.
    """
    if app_state.get_users_collection() is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured - set MONGODB_URI and install motor")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")

    users = app_state.get_users_collection()
    email_lower = payload.email.lower()
    user_doc = await users.find_one({"email": email_lower})
    if not user_doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # Accept 'password_hash' or 'password' field; require bcrypt hash
    stored_hash = user_doc.get("password_hash") or user_doc.get("password")
    if not stored_hash:
        raise HTTPException(
            status_code=500,
            detail="User record missing password hash; ensure bcrypt-hashed passwords are stored as 'password_hash'"
        )

    try:
        if isinstance(stored_hash, str):
            stored_hash_bytes = stored_hash.encode("utf-8")
        else:
            stored_hash_bytes = stored_hash
        if not bcrypt.checkpw(payload.password.encode("utf-8"), stored_hash_bytes):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    except ValueError:
        # If hash format is invalid for bcrypt
        raise HTTPException(status_code=500, detail="Stored password hash is invalid format")

    user_id = str(user_doc.get("_id"))
    token, jti, exp_dt = create_access_token(subject=user_id, email=email_lower, expires_in=JWT_EXPIRES_IN)
    return TokenResponse(access_token=token, token_type="bearer", expires_in=JWT_EXPIRES_IN)

# PUBLIC_INTERFACE
@app.post(
    "/auth/logout",
    summary="Logout by revoking the current access token",
    tags=["Auth"],
    responses={
        200: {"description": "Token revoked"},
        401: {"description": "Unauthorized or already revoked"},
        500: {"description": "Server misconfiguration"},
    },
)
async def auth_logout(payload: Dict[str, Any] = Depends(get_jwt_payload)):
    """Invalidate the current access token by storing its jti in a blacklist collection with TTL.

    Behavior:
    - Reads the JWT from Authorization: Bearer <token>
    - Adds the token's jti to 'token_blacklist' with expiresAt equal to token's exp
    - TTL index on expiresAt (expireAfterSeconds=0) ensures revocation record is removed when token naturally expires
    """
    blacklist = app_state.get_blacklist_collection()
    if blacklist is None:
        raise HTTPException(status_code=500, detail="MongoDB not configured - set MONGODB_URI and install motor")

    jti = payload.get("jti")
    sub = payload.get("sub")
    exp = payload.get("exp")
    if not jti or not exp:
        raise HTTPException(status_code=401, detail="Invalid token")

    expires_at = datetime.fromtimestamp(int(exp), tz=timezone.utc)
    doc = {
        "jti": jti,
        "user_id": str(sub) if sub else None,
        "expiresAt": expires_at,
        "createdAt": datetime.now(timezone.utc),
    }
    try:
        await blacklist.insert_one(doc)
    except Exception as e:
        # Duplicate jti means already logged out
        if "E11000" in str(e):
            return {"message": "Token already revoked"}
        logger.error(f"Failed to insert blacklist record: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to revoke token")

    return {"message": "Logged out successfully"}

# PUBLIC_INTERFACE
@app.get(
    "/auth/me",
    response_model=UserPublic,
    summary="Get current user",
    tags=["Auth"],
)
async def auth_me(user: UserPublic = Depends(get_current_user)):
    """Return the current authenticated user's public profile."""
    return user

# -----------------------------------------------------------------------------
# API Endpoints (existing features)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Subtitle Synchronizer API",
        "version": "1.0.1",
        "status": "healthy",
        "endpoints": {
            "upload_files": "/upload-files/",
            "generate_subtitles": "/generate/",
            "sync_subtitles": "/correct_subtitles_async/",
            "get_status": "/sync_status/{task_id}",
            "download_result": "/download/{filename}",
            "list_files": "/files/",
            "docs": "/docs",
            "auth_login": "/auth/login",
            "auth_logout": "/auth/logout",
            "auth_me": "/auth/me",
        }
    }

# API endpoint for uploading video only (for generation)
@app.post("/upload-video/")
async def upload_video(
    video_file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
):
    """Upload video file for subtitle generation."""
    try:
        await validate_file_upload(video_file, ALLOWED_VIDEO_EXTENSIONS)
        video_path = safe_save_file(video_file, config.upload_dir)

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
    video_file: UploadFile = File(...),
    subtitle_file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
):
    """Upload video and subtitle files with validation."""
    try:
        # Validate files first
        await validate_file_upload(video_file, ALLOWED_VIDEO_EXTENSIONS)
        await validate_file_upload(subtitle_file, ALLOWED_SUBTITLE_EXTENSIONS)

        video_path = safe_save_file(video_file, config.upload_dir)
        subtitle_path = safe_save_file(subtitle_file, config.upload_dir)

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

# Try to import required modules for subtitle processing
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

try:
    from subtitle_correction import main as correct_subtitles_main
    from subtitle_generation_combined import generate_subtitles, write_subtitles

    subtitle_sync_module = safe_import_module("subtitle_sync", "subtitle_correction.py")
    SubtitleSynchronizer = subtitle_sync_module.SubtitleSynchronizer

    reposition_module = safe_import_module("reposition_subtitle", "reposition_subtitle.py")

except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    raise

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

# API endpoint for subtitle generation
@app.post("/generate/")
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Start subtitle generation task."""
    try:
        video_path = UPLOAD_DIR / request.video_filename
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
                output_path = OUTPUT_DIR / output_filename
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
    video: UploadFile = File(...),
    subtitle: UploadFile = File(...),
    translation_engine: str = Form("m2m100"),
    translation_credentials: Optional[str] = Form(None),
    task_manager: TaskManager = Depends(get_task_manager),
    config: AppConfig = Depends(get_app_config)
):
    """Start async subtitle correction with progress tracking."""
    try:
        # Validate files
        await validate_file_upload(video, ALLOWED_VIDEO_EXTENSIONS)
        await validate_file_upload(subtitle, ALLOWED_SUBTITLE_EXTENSIONS)

        # Generate task ID using injected TaskManager
        task_id = task_manager.create_task("correction")

        # Save uploaded files
        video_path = safe_save_file(video, config.upload_dir)
        subtitle_path = safe_save_file(subtitle, config.upload_dir)

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
                    output_dir=str(config.output_dir),
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
                        os.path.join(str(UPLOAD_DIR), output_filename),
                        os.path.join(str(OUTPUT_DIR), output_filename),  # Maybe it's already there
                    ]

                    source_path = None
                    for path in search_paths:
                        if os.path.exists(path):
                            source_path = path
                            logger.info(f"Found file at: {path}")
                            break

                    if source_path:
                        destination_path = OUTPUT_DIR / output_filename
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
    request: RepositionRequest,
    background_tasks: BackgroundTasks,
    executor: ThreadPoolExecutor = Depends(get_executor)
):
    """Start subtitle repositioning task (after sync)."""
    try:
        video_path = UPLOAD_DIR / request.video_filename
        subtitle_path_outputs = OUTPUT_DIR / request.subtitle_filename
        subtitle_path_uploads = UPLOAD_DIR / request.subtitle_filename

        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")

        if subtitle_path_outputs.exists():
            subtitle_path = subtitle_path_outputs
        elif subtitle_path_uploads.exists():
            subtitle_path = subtitle_path_uploads
        else:
            raise HTTPException(status_code=404, detail=f"Subtitle file not found: {request.subtitle_filename}")

        # Run repositioning in background
        def run_reposition():
            try:
                output_file = reposition_module.process_subtitle(str(video_path), str(subtitle_path))
                output_path = OUTPUT_DIR / Path(output_file).name
                if Path(output_file) != output_path:
                    # Always overwrite the output file in OUTPUT_DIR
                    if output_path.exists():
                        output_path.unlink()  # Remove old file
                    Path(output_file).rename(output_path)  # Use Path.rename for moving
                return output_path.name
            except Exception as e:
                raise e

        loop = asyncio.get_running_loop()
        output_filename = await loop.run_in_executor(executor, run_reposition)

        if output_filename:
            return {
                "success": "true",
                "message": "Subtitle repositioning completed",
                "output_filename": output_filename
            }
        else:
            raise HTTPException(status_code=500, detail="Subtitle repositioning failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during repositioning: {str(e)}")

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

# API endpoint for downloading result files
@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a synchronized subtitle file."""
    # Validate filename to prevent path traversal.
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = OUTPUT_DIR / filename

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
async def list_files():
    """List available files in upload and output directories."""
    try:
        # Use os.scandir for efficient directory traversal
        upload_files = [entry.name for entry in os.scandir(UPLOAD_DIR) if entry.is_file()]
        output_files = [entry.name for entry in os.scandir(OUTPUT_DIR) if entry.is_file()]

        return {
            "upload_files": upload_files,
            "output_files": output_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

# API endpoint for deleting files
@app.delete("/files/{filename}")
async def delete_file(filename: str, file_type: str = "upload"):
    """Delete a file from the upload or output directory."""
    try:
        # Validate filename to prevent path traversal.
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        if file_type == "upload":
            file_path = UPLOAD_DIR / filename
        elif file_type == "output":
            file_path = OUTPUT_DIR / filename
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
async def cleanup_files(task_manager: TaskManager = Depends(get_task_manager)):
    """Clean up all uploaded and output files."""
    try:
        # Batch delete files to reduce I/O overhead
        upload_files = list(UPLOAD_DIR.iterdir())
        output_files = list(OUTPUT_DIR.iterdir())

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
async def health_check(config: AppConfig = Depends(get_app_config)):
    """Enhanced health check endpoint."""
    try:
        task_manager = app_state.get_task_manager()

        # Optimize disk usage check
        upload_space, output_space = map(
            shutil.disk_usage, [config.upload_dir, config.output_dir]
        )

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "active_tasks": len([t for t in task_manager._tasks.values() if t.get("status") == "processing"]),
            "total_tasks": len(task_manager._tasks),
            "disk_space": {
                "upload_dir_free_gb": upload_space.free / (1024**3),
                "output_dir_free_gb": output_space.free / (1024**3)
            },
            "directories": {
                "upload_files": len(list(config.upload_dir.glob("*"))),
                "output_files": len(list(config.output_dir.glob("*")))
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
