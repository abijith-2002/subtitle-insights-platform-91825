#!/usr/bin/env python3
"""
main_api_calls_abort.py - Process-based task runner with abort-by-PID functionality.

This module provides a process-oriented alternative to main_api_calls.py.
It spawns long-running tasks (generation, correction, repositioning) as separate OS
processes (not threads), immediately returns the child's PID to the caller, and
allows aborting a task by PID.

Usage (HTTP):
- POST /tasks/generation/start -> { pid, status, task_type }
- POST /tasks/correction/start -> { pid, status, task_type }
- POST /tasks/repositioning/start -> { pid, status, task_type }
- POST /tasks/abort -> { pid, status }
- GET  /tasks/list -> { tasks: [...] }
- GET  /tasks/{pid}/status -> { ... }

Usage (imported functions):
- start_generation(params: dict) -> dict
- start_correction(params: dict) -> dict
- start_repositioning(params: dict) -> dict
- abort_kill(pid: int, force: bool=False, timeout: int=5) -> dict

Notes:
- Each task runs in a separate OS process using multiprocessing with the 'spawn' start method,
  which is safe and portable across POSIX/Windows.
- A small in-memory registry tracks processes by PID, including type, status, start_time, and a hash of parameters.
- Aborts validate that the PID belongs to a known spawned task to prevent arbitrary process termination.
- The module exposes a standalone FastAPI app and a Router so it can be run independently or included.

Environment:
- Uses per-user directories for uploads/outputs at: <this_dir>/uploads/{user_id}/ and <this_dir>/outputs/{user_id}/
- This file does not modify the original main_api_calls.py; it stands alongside it.

Dependencies expected from the project (already used in main_api_calls.py):
- fastapi, pydantic
- subtitle_generation_combined: generate_subtitles, write_subtitles
- subtitle_correction: main (aliased as correct_subtitles_main)
- repositioning.reposition_subtitle8: process_subtitle
"""

import os
import sys
import json
import time
import signal
import hashlib
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional, List

from multiprocessing import get_context
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from pydantic import BaseModel, Field

# External functions from project modules
# These are used by the child processes; imports are resolved in both parent and child.
try:
    from subtitle_correction import main as correct_subtitles_main
except Exception as e:
    # Defer raising to actual use, to avoid hard import failure during some tooling
    correct_subtitles_main = None  # type: ignore

try:
    from subtitle_generation_combined import generate_subtitles, write_subtitles
except Exception as e:
    generate_subtitles = None  # type: ignore
    write_subtitles = None  # type: ignore

# Prefer explicit path import for repositioning module that exists in repo
try:
    from repositioning.reposition_subtitle8 import process_subtitle as reposition_process_subtitle
except Exception as e:
    reposition_process_subtitle = None  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("main_api_calls_abort")

# Constants and paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
ALLOWED_SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}

# Create base directories if not exist (per project convention)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory process registry
# pid -> info dict
task_registry: Dict[int, Dict[str, Any]] = {}
# pid -> Process object (kept separate to avoid serialization issues)
_process_table: Dict[int, Any] = {}

# Cross-module constants
DEFAULT_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://<ipv4_address>:3000"]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _params_hash(params: Dict[str, Any]) -> str:
    """Compute a stable hash of the params for registry introspection."""
    try:
        # Ensure consistent ordering for hash
        payload = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        payload = str(params)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _get_user_dirs(user_id: str) -> Dict[str, Path]:
    """Return per-user upload/output directories, ensuring they exist."""
    udir = UPLOAD_DIR / user_id
    odir = OUTPUT_DIR / user_id
    udir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    return {"upload_dir": udir, "output_dir": odir}

def _normalize_translation_credentials(creds: Any) -> Optional[Dict[str, Any]]:
    """Accept dict or JSON string and return dict; otherwise None."""
    if creds is None:
        return None
    if isinstance(creds, dict):
        return creds
    if isinstance(creds, str):
        try:
            return json.loads(creds)
        except Exception:
            return None
    return None

def _check_exists_or_raise(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path.name}")

def _is_posix() -> bool:
    return os.name == "posix"

def _os_process_exists(pid: int) -> bool:
    """Return True if a process with PID exists."""
    if pid <= 0:
        return False
    try:
        # On POSIX, signal 0 checks for existence
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission
        return True
    except Exception:
        # On some systems, fall back to table
        p = _process_table.get(pid)
        return bool(p and p.is_alive())

def _platform_name() -> str:
    return platform.system().lower()

# ------------------------------------------------------------------------------
# Child process entrypoints
# Each entrypoint is spawn-safe and receives one params argument (dict).
# ------------------------------------------------------------------------------

def _generation_entry(params: Dict[str, Any]) -> None:
    """
    Child process entry for generation task.
    Expects keys: user_id, video_filename, subtitle_lang, subtitle_format,
                  output_filename (optional), translation_engine, translation_credentials
    """
    # Recompute paths in the child process
    user_id = params["user_id"]
    paths = _get_user_dirs(user_id)
    video_path = paths["upload_dir"] / params["video_filename"]

    subtitle_lang = params.get("subtitle_lang", "en")
    subtitle_format = params.get("subtitle_format", "srt")
    output_filename = params.get("output_filename")
    translation_engine = params.get("translation_engine")
    translation_credentials = _normalize_translation_credentials(params.get("translation_credentials"))

    if generate_subtitles is None or write_subtitles is None:
        raise RuntimeError("subtitle_generation_combined module is not available")

    # Determine output filename
    if not output_filename:
        base_name = Path(params["video_filename"]).stem
        output_filename = f"{base_name}_subtitles.{subtitle_format}"

    output_path = paths["output_dir"] / output_filename
    _check_exists_or_raise(video_path, "Video file")

    try:
        # Perform generation
        final_segments = generate_subtitles(
            str(video_path),
            subtitle_lang=subtitle_lang,
            translation_engine=translation_engine,
            translation_credentials=translation_credentials
        )
        # Write output
        write_subtitles(final_segments, subtitle_format, str(output_path))
        # Successful exit
        sys.exit(0)
    except Exception as e:
        log.exception("Generation task failed: %s", e)
        # Non-zero exitcode indicates failure
        sys.exit(1)

def _correction_entry(params: Dict[str, Any]) -> None:
    """
    Child process entry for correction task.
    Expects keys: user_id, video_filename, subtitle_filename, translation_engine, translation_credentials
    """
    user_id = params["user_id"]
    paths = _get_user_dirs(user_id)
    upload_dir = paths["upload_dir"]
    output_dir = paths["output_dir"]

    video_path = upload_dir / params["video_filename"]
    subtitle_path = upload_dir / params["subtitle_filename"]

    translation_engine = params.get("translation_engine", "m2m100")
    translation_credentials = _normalize_translation_credentials(params.get("translation_credentials"))

    if correct_subtitles_main is None:
        raise RuntimeError("subtitle_correction module is not available")

    _check_exists_or_raise(video_path, "Video file")
    _check_exists_or_raise(subtitle_path, "Subtitle file")

    try:
        # Run correction
        result = correct_subtitles_main(
            str(video_path),
            str(subtitle_path),
            output_dir=str(output_dir),
            translation_engine=translation_engine,
            translation_credentials=translation_credentials
        )

        # Locate and move the output file if needed (mirror logic from main_api_calls.py)
        if result and result.get("success") and result.get("output_file"):
            output_filename = result["output_file"]
            # Potential locations
            search_paths = [
                output_filename,
                os.path.join(os.getcwd(), output_filename),
                os.path.join(str(upload_dir), output_filename),
                os.path.join(str(output_dir), output_filename)
            ]
            source_path = None
            for p in search_paths:
                if os.path.exists(p):
                    source_path = p
                    break

            if source_path:
                dst_path = output_dir / output_filename
                if Path(source_path) != dst_path:
                    if dst_path.exists():
                        dst_path.unlink()
                    Path(source_path).rename(dst_path)

        sys.exit(0)
    except Exception as e:
        log.exception("Correction task failed: %s", e)
        sys.exit(1)

def _repositioning_entry(params: Dict[str, Any]) -> None:
    """
    Child process entry for repositioning task.
    Expects keys: user_id, video_filename, subtitle_filename
    """
    user_id = params["user_id"]
    paths = _get_user_dirs(user_id)
    upload_dir = paths["upload_dir"]
    output_dir = paths["output_dir"]

    if reposition_process_subtitle is None:
        raise RuntimeError("repositioning.reposition_subtitle8 module is not available")

    video_path = upload_dir / params["video_filename"]
    sub_outputs = output_dir / params["subtitle_filename"]
    sub_uploads = upload_dir / params["subtitle_filename"]

    _check_exists_or_raise(video_path, "Video file")

    if sub_outputs.exists():
        subtitle_path = sub_outputs
    elif sub_uploads.exists():
        subtitle_path = sub_uploads
    else:
        raise FileNotFoundError(f"Subtitle file not found: {params['subtitle_filename']}")

    try:
        output_file = reposition_process_subtitle(str(video_path), str(subtitle_path))
        final_output = output_dir / Path(output_file).name

        if Path(output_file) != final_output:
            if final_output.exists():
                final_output.unlink()
            Path(output_file).rename(final_output)

        sys.exit(0)
    except Exception as e:
        log.exception("Repositioning task failed: %s", e)
        sys.exit(1)

# ------------------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------------------

def _register_process(proc: Any, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Add a process to the registry and return a response."""
    pid = proc.pid
    info = {
        "pid": pid,
        "type": task_type,
        "status": "running",
        "start_time": time.time(),
        "params_hash": _params_hash(params),
        "params": params,
        "platform": _platform_name()
    }
    task_registry[pid] = info
    _process_table[pid] = proc
    return {"pid": pid, "status": "started", "task_type": task_type}

def _update_status_on_exit(pid: int, exitcode: Optional[int]) -> None:
    info = task_registry.get(pid)
    if not info:
        return
    info["end_time"] = time.time()
    info["exitcode"] = exitcode
    if exitcode == 0:
        info["status"] = "completed"
    elif exitcode is None:
        info["status"] = "unknown"
    else:
        info["status"] = "error"

# ------------------------------------------------------------------------------
# PUBLIC API (functions)
# ------------------------------------------------------------------------------

# PUBLIC_INTERFACE
def start_generation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start subtitle generation in a separate OS process and return its PID."""
    ctx = get_context("spawn")
    # Ensure required inputs
    user_id = params.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    video_filename = params.get("video_filename")
    if not video_filename:
        raise ValueError("video_filename is required")

    proc = ctx.Process(target=_generation_entry, args=(params,), daemon=True)
    proc.start()
    return _register_process(proc, "generation", params)

# PUBLIC_INTERFACE
def start_correction(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start subtitle correction in a separate OS process and return its PID."""
    ctx = get_context("spawn")
    user_id = params.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    if not params.get("video_filename") or not params.get("subtitle_filename"):
        raise ValueError("video_filename and subtitle_filename are required")

    proc = ctx.Process(target=_correction_entry, args=(params,), daemon=True)
    proc.start()
    return _register_process(proc, "correction", params)

# PUBLIC_INTERFACE
def start_repositioning(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start subtitle repositioning in a separate OS process and return its PID."""
    ctx = get_context("spawn")
    user_id = params.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    if not params.get("video_filename") or not params.get("subtitle_filename"):
        raise ValueError("video_filename and subtitle_filename are required")

    proc = ctx.Process(target=_repositioning_entry, args=(params,), daemon=True)
    proc.start()
    return _register_process(proc, "repositioning", params)

# PUBLIC_INTERFACE
def abort_kill(pid: int, force: bool = False, timeout: int = 5) -> Dict[str, Any]:
    """
    Abort (terminate) a running task by PID with cross-platform behavior.

    - On POSIX: send SIGTERM; wait up to timeout; then SIGKILL if still alive or force=True.
    - On Windows: try os.kill first; then fallback to 'taskkill /PID {pid} /T /F'.

    Only terminates processes that are in our registry to prevent arbitrary kills.
    """
    info = task_registry.get(pid)
    if not info:
        return {"pid": pid, "status": "not_found"}

    # Try graceful termination first
    try:
        if _is_posix():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                info["status"] = "not_found"
                return {"pid": pid, "status": "not_found"}
        else:
            # Windows: attempt os.kill first (available in Python on Windows)
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                # Fallback to taskkill
                try:
                    import subprocess  # noqa
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
                except Exception as e:
                    log.warning("taskkill fallback failed for PID %s: %s", pid, e)

        # Wait up to timeout for exit
        t0 = time.time()
        while time.time() - t0 < timeout:
            p = _process_table.get(pid)
            if p is not None:
                p.join(timeout=0.05)
                if not p.is_alive():
                    _update_status_on_exit(pid, p.exitcode)
                    # Cleanup table reference
                    _process_table.pop(pid, None)
                    return {"pid": pid, "status": "terminated"}
            else:
                # Fallback: check OS-level
                if not _os_process_exists(pid):
                    _update_status_on_exit(pid, exitcode=None)
                    _process_table.pop(pid, None)
                    return {"pid": pid, "status": "terminated"}
            time.sleep(0.1)

        # Force kill if still alive or requested
        if _is_posix():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            try:
                import subprocess  # noqa
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            except Exception:
                pass

        # Update status post-force
        p = _process_table.get(pid)
        if p:
            p.join(timeout=0.2)
            _update_status_on_exit(pid, p.exitcode)
            _process_table.pop(pid, None)
        info["status"] = "killed"
        return {"pid": pid, "status": "killed"}
    except Exception as e:
        log.exception("Abort failed for PID %s: %s", pid, e)
        info["status"] = "error"
        return {"pid": pid, "status": "error"}

# ------------------------------------------------------------------------------
# FastAPI app and schemas
# ------------------------------------------------------------------------------

class GenerationStartRequest(BaseModel):
    """Request payload for starting generation."""
    user_id: Optional[str] = Field(None, description="User ID; taken from auth if omitted")
    video_filename: str = Field(..., description="Uploaded video filename under /uploads/{user_id}/")
    subtitle_lang: str = Field("en", description="Target language for subtitles")
    subtitle_format: str = Field("srt", description="Output subtitle format (srt|vtt)")
    output_filename: Optional[str] = Field(None, description="Optional output filename override")
    translation_engine: Optional[str] = Field(None, description="Translation engine name or key")
    translation_credentials: Optional[Any] = Field(None, description="Engine credentials as dict or JSON string")

class CorrectionStartRequest(BaseModel):
    """Request payload for starting correction."""
    user_id: Optional[str] = Field(None, description="User ID; taken from auth if omitted")
    video_filename: str = Field(..., description="Uploaded video filename under /uploads/{user_id}/")
    subtitle_filename: str = Field(..., description="Uploaded subtitle filename under /uploads/{user_id}/")
    translation_engine: str = Field("m2m100", description="Translation engine name")
    translation_credentials: Optional[Any] = Field(None, description="Engine credentials as dict or JSON string")

class RepositioningStartRequest(BaseModel):
    """Request payload for starting repositioning."""
    user_id: Optional[str] = Field(None, description="User ID; taken from auth if omitted")
    video_filename: str = Field(..., description="Video filename under /uploads/{user_id}/")
    subtitle_filename: str = Field(..., description="Subtitle filename under /uploads/{user_id}/ or /outputs/{user_id}/")

class AbortRequest(BaseModel):
    """Request payload to abort a running task by PID."""
    pid: int = Field(..., description="Process ID to abort")
    force: bool = Field(False, description="Force kill immediately")
    timeout: int = Field(5, description="Seconds to wait before forcing termination")

openapi_tags = [
    {"name": "Tasks", "description": "Process-based task management endpoints"},
]

router = APIRouter()

def _resolve_user_id(request: Request, fallback: Optional[str]) -> str:
    """Try to resolve user_id from request.state.user; fallback to provided."""
    user_id = fallback
    try:
        req_user = getattr(request.state, "user", None)
        if isinstance(req_user, dict) and req_user.get("id"):
            user_id = req_user.get("id")
    except Exception:
        pass
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required (auth or body)")
    return user_id

@router.get("/tasks/usage", tags=["Tasks"], summary="Process-based API usage")
def tasks_usage() -> Dict[str, Any]:
    """
    Return usage information for the process-based task endpoints.

    This is a convenience endpoint explaining the available API calls and expected payloads.
    """
    return {
        "message": "Process-based task runner usage",
        "endpoints": {
            "start_generation": {"method": "POST", "path": "/tasks/generation/start"},
            "start_correction": {"method": "POST", "path": "/tasks/correction/start"},
            "start_repositioning": {"method": "POST", "path": "/tasks/repositioning/start"},
            "abort": {"method": "POST", "path": "/tasks/abort"},
            "list_tasks": {"method": "GET", "path": "/tasks/list"},
            "task_status": {"method": "GET", "path": "/tasks/{pid}/status"},
        },
        "notes": "All tasks spawn separate OS processes. The start endpoints return a PID. Use /tasks/abort to terminate by PID.",
    }

@router.post(
    "/tasks/generation/start",
    tags=["Tasks"],
    summary="Start generation task (process-based)",
    response_model=Dict[str, Any]
)
def http_start_generation(request: Request, body: GenerationStartRequest) -> Dict[str, Any]:
    """
    Start a subtitle generation task as a separate OS process.

    Parameters:
    - user_id: Optional string; if omitted, taken from request.state.user.id
    - video_filename: The video file previously uploaded to /uploads/{user_id}/
    - subtitle_lang: Target language code for subtitles (default 'en')
    - subtitle_format: Output format, e.g., 'srt' or 'vtt'
    - output_filename: Optional override for output filename
    - translation_engine: Optional translation engine key/name
    - translation_credentials: Optional dict or stringified JSON with credentials

    Returns:
    - JSON containing the spawned PID, status, and task_type
    """
    user_id = _resolve_user_id(request, body.user_id)
    params = body.dict()
    params["user_id"] = user_id
    try:
        return start_generation(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

@router.post(
    "/tasks/correction/start",
    tags=["Tasks"],
    summary="Start correction task (process-based)",
    response_model=Dict[str, Any]
)
def http_start_correction(request: Request, body: CorrectionStartRequest) -> Dict[str, Any]:
    """
    Start a subtitle correction task as a separate OS process.

    Parameters:
    - user_id: Optional string; if omitted, taken from request.state.user.id
    - video_filename: The video file previously uploaded to /uploads/{user_id}/
    - subtitle_filename: The subtitle file previously uploaded to /uploads/{user_id}/
    - translation_engine: Translation engine to use
    - translation_credentials: Optional dict or stringified JSON with credentials

    Returns:
    - JSON containing the spawned PID, status, and task_type
    """
    user_id = _resolve_user_id(request, body.user_id)
    params = body.dict()
    params["user_id"] = user_id
    try:
        return start_correction(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start correction: {str(e)}")

@router.post(
    "/tasks/repositioning/start",
    tags=["Tasks"],
    summary="Start repositioning task (process-based)",
    response_model=Dict[str, Any]
)
def http_start_repositioning(request: Request, body: RepositioningStartRequest) -> Dict[str, Any]:
    """
    Start a subtitle repositioning task as a separate OS process.

    Parameters:
    - user_id: Optional string; if omitted, taken from request.state.user.id
    - video_filename: The video file previously uploaded to /uploads/{user_id}/
    - subtitle_filename: The subtitle file located under /uploads/{user_id}/ or /outputs/{user_id}/

    Returns:
    - JSON containing the spawned PID, status, and task_type
    """
    user_id = _resolve_user_id(request, body.user_id)
    params = body.dict()
    params["user_id"] = user_id
    try:
        return start_repositioning(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start repositioning: {str(e)}")

@router.post(
    "/tasks/abort",
    tags=["Tasks"],
    summary="Abort a running task by PID",
    response_model=Dict[str, Any]
)
def http_abort(body: AbortRequest) -> Dict[str, Any]:
    """
    Abort a process by PID.

    Parameters:
    - pid: Process ID to abort (must be a PID known in this module's registry)
    - force: If true, skip grace period and force kill
    - timeout: Seconds to wait for graceful termination before force killing

    Returns:
    - JSON containing pid and status (terminated|killed|not_found|error)
    """
    try:
        return abort_kill(body.pid, force=body.force, timeout=body.timeout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Abort failed: {str(e)}")

@router.get(
    "/tasks/list",
    tags=["Tasks"],
    summary="List known tasks in registry",
    response_model=Dict[str, Any]
)
def http_list_tasks() -> Dict[str, Any]:
    """
    List tasks currently known to the in-memory registry.

    Returns:
    - tasks: List of task info dictionaries (one per PID)
    """
    return {"tasks": list(task_registry.values())}

@router.get(
    "/tasks/{pid}/status",
    tags=["Tasks"],
    summary="Get task status by PID",
    response_model=Dict[str, Any]
)
def http_task_status(pid: int) -> Dict[str, Any]:
    """
    Get the status of a task by PID.

    Returns:
    - Task info dictionary if found
    - 404 if not found
    """
    info = task_registry.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="PID not found")
    # Live update if we still have a process object
    proc = _process_table.get(pid)
    if proc and not proc.is_alive():
        _update_status_on_exit(pid, proc.exitcode)
        _process_table.pop(pid, None)
    return info

# ------------------------------------------------------------------------------
# FastAPI App with lifespan to monitor processes
# ------------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create the FastAPI app with middleware and routes."""
    app = FastAPI(
        title="Subtitle Process Task Runner API",
        description="Start and stop long-running subtitle tasks using OS processes (abortable by PID).",
        version="1.0.0",
        openapi_tags=openapi_tags
    )
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEFAULT_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # Add routes
    app.include_router(router)

    # Start a background monitor using startup/shutdown events
    @app.on_event("startup")
    async def _startup_monitor():
        log.info("Process Task Runner API starting up.")

    @app.on_event("shutdown")
    async def _shutdown_monitor():
        log.info("Process Task Runner API shutting down.")

    return app

app = create_app()

# ------------------------------------------------------------------------------
# Optional main entry
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Note: Running uvicorn with reload may spawn multiple processes; avoid reload for safety.
    import uvicorn
    log.info("Starting Process Task Runner API at http://0.0.0.0:8100 (docs at /docs)")
    uvicorn.run("main_api_calls_abort:app", host="0.0.0.0", port=8100, reload=False, log_level="info")
