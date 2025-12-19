"""Backend server for Sharp equirectangular to cubemap/splats pipeline.

A FastAPI server with WebSocket progress updates for the Sharp workflow.
Run with: uv run uvicorn server.main:app --reload --port 8765
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# Add parent to path for sharp imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sharp.cli.equirect_to_cubefaces import (
    CUBE_FACES,
    compute_focal_length_px,
    equirect_to_perspective,
    focal_length_to_35mm_equiv,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Thread pool for CPU-bound inference tasks
INFERENCE_POOL = ThreadPoolExecutor(max_workers=4)

# Global model cache - loaded once, reused for all requests
_cached_model: Any = None
_cached_device: str | None = None

app = FastAPI(title="Sharp Pipeline API", description="Equirectangular to Cubemap/Splats Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for active WebSocket connections
active_connections: dict[str, WebSocket] = {}

# Output directory for generated files (in viewer/public for Vite to serve)
OUTPUT_DIR = Path(__file__).parent.parent / "public" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also ensure splats directory exists
SPLATS_DIR = Path(__file__).parent.parent / "public" / "splats"
SPLATS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineProgress:
    """Track pipeline progress."""
    
    job_id: str
    total_steps: int = 0
    current_step: int = 0
    status: str = "pending"
    message: str = ""
    results: dict = field(default_factory=dict)
    
    @property
    def percent(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min((self.current_step / self.total_steps) * 100, 100.0)


async def send_progress(job_id: str, progress: PipelineProgress):
    """Send progress update via WebSocket."""
    if job_id in active_connections:
        ws = active_connections[job_id]
        try:
            await ws.send_json({
                "job_id": job_id,
                "percent": progress.percent,
                "current_step": progress.current_step,
                "total_steps": progress.total_steps,
                "status": progress.status,
                "message": progress.message,
                "results": progress.results,
            })
        except Exception as e:
            LOGGER.error(f"Failed to send progress: {e}")


async def run_equirect_to_cubefaces(
    input_path: Path,
    output_path: Path,
    fov: float,
    size: int | None,
    faces_to_extract: list[str],
    progress: PipelineProgress,
) -> dict:
    """Run equirectangular to cube faces conversion with progress updates."""
    
    progress.message = "Loading equirectangular image..."
    await send_progress(progress.job_id, progress)
    
    img = Image.open(input_path)
    equirect = np.array(img)
    
    # Ensure RGB
    if equirect.ndim == 2:
        equirect = np.stack([equirect] * 3, axis=-1)
    elif equirect.shape[-1] == 4:
        equirect = equirect[..., :3]
    
    h_eq, w_eq = equirect.shape[:2]
    
    # Determine output size
    if size is None or size == 0:
        size = h_eq // 2
        size = (size // 16) * 16
        size = max(size, 512)
    
    # Compute focal length
    f_px = compute_focal_length_px(fov, size)
    f_35mm = focal_length_to_35mm_equiv(f_px, size)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_faces = []
    
    # Extract each face
    for i, face_name in enumerate(faces_to_extract):
        progress.current_step += 1
        progress.message = f"Extracting {face_name} face ({i + 1}/{len(faces_to_extract)})"
        await send_progress(progress.job_id, progress)
        
        yaw, pitch = CUBE_FACES[face_name]
        face_image = equirect_to_perspective(equirect, fov, yaw, pitch, size)
        
        # Save the image
        output_filename = f"input_{face_name}.png"
        output_file = output_path / output_filename
        
        pil_image = Image.fromarray(face_image)
        exif = pil_image.getexif()
        exif[41989] = int(round(f_35mm))
        pil_image.save(output_file, "PNG")
        
        generated_faces.append({
            "name": face_name,
            "filename": output_filename,
            "url": f"/generated/{progress.job_id}/cubefaces/{output_filename}",
        })
        
        await asyncio.sleep(0.05)
    
    return {
        "faces": generated_faces,
        "metadata": {
            "source_size": f"{w_eq} x {h_eq}",
            "face_size": f"{size} x {size}",
            "fov": fov,
            "focal_length_px": round(f_px, 2),
            "focal_length_35mm": round(f_35mm, 2),
        }
    }


def get_cached_model_path() -> Path | None:
    """Check if the Sharp model is cached locally."""
    import torch
    
    # Check torch hub cache
    hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
    model_file = hub_dir / "sharp_2572gikvuh.pt"
    if model_file.exists():
        return model_file
    
    # Check common locations
    home = Path.home()
    alt_locations = [
        home / ".cache" / "torch" / "hub" / "checkpoints" / "sharp_2572gikvuh.pt",
        Path("sharp_2572gikvuh.pt"),  # Current directory
    ]
    for loc in alt_locations:
        if loc.exists():
            return loc
    
    return None


def get_best_device() -> str:
    """Determine the best available device for inference."""
    import torch
    import sys
    
    # Debug: show which Python and torch we're using
    LOGGER.info(f"Python executable: {sys.executable}")
    LOGGER.info(f"Torch version: {torch.__version__}")
    LOGGER.info(f"Torch file: {torch.__file__}")
    LOGGER.info(f"CUDA built: {torch.version.cuda}")
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        LOGGER.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        return "cuda"
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        LOGGER.info("MPS (Apple Silicon) available")
        return "mps"
    else:
        LOGGER.warning("No GPU available, using CPU (this will be slow!)")
        return "cpu"


def get_or_load_model():
    """Get the cached model or load it if not cached.
    
    This function ensures the model is loaded only once and kept in memory
    for fast subsequent predictions.
    """
    global _cached_model, _cached_device
    
    import torch
    from sharp.models import PredictorParams, create_predictor
    
    if _cached_model is not None:
        LOGGER.info(f"Using cached model on {_cached_device}")
        return _cached_model, _cached_device
    
    # Check if model weights are available
    model_path = get_cached_model_path()
    if model_path is None:
        raise RuntimeError(
            "Sharp model not found. Please run 'sharp predict' once first to download the model:\n"
            "  sharp predict -i <any_image> -o /tmp/test\n"
            "This will download and cache the model for future use."
        )
    
    # Determine best device
    _cached_device = get_best_device()
    
    LOGGER.info(f"Loading model from {model_path} onto {_cached_device}...")
    
    # Load model with map_location to target device directly
    state_dict = torch.load(model_path, weights_only=True, map_location=_cached_device)
    
    _cached_model = create_predictor(PredictorParams())
    _cached_model.load_state_dict(state_dict)
    _cached_model.eval()
    _cached_model.to(_cached_device)
    
    # Note: torch.compile requires Triton which isn't available on Windows
    # Skip compilation - eager mode is still fast on GPU
    
    LOGGER.info(f"Model loaded successfully on {_cached_device}")
    return _cached_model, _cached_device


def _predict_single_image_sync(
    model,
    device: str,
    image_path: Path,
    fov: float,
    output_path: Path,
    splats_output_path: Path,
    job_id: str,
) -> dict:
    """Synchronous prediction for a single image (runs in thread pool)."""
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import save_ply, unproject_gaussians
    
    internal_shape = (1536, 1536)
    
    # Load image
    img_pil = Image.open(image_path)
    image = np.array(img_pil)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    
    height, width = image.shape[:2]
    
    # Compute focal length from FOV
    size = min(width, height)
    f_px = size / (2 * np.tan(np.deg2rad(fov) / 2))
    LOGGER.info(f"Using FOV {fov}° -> focal length {f_px:.2f}px for {image_path.stem}")
    
    # Preprocess
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    
    # Predict
    with torch.no_grad():
        gaussians_ndc = model(image_resized_pt, disparity_factor)
    
    # Postprocess
    intrinsics = torch.tensor([
        [f_px, 0, width / 2, 0],
        [0, f_px, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).float().to(device)
    
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )
    
    # Save PLY to job output
    ply_filename = f"{image_path.stem}.ply"
    ply_path = output_path / ply_filename
    save_ply(gaussians, f_px, (height, width), ply_path)
    
    # Also copy to splats directory for viewer
    splats_ply_path = splats_output_path / ply_filename
    save_ply(gaussians, f_px, (height, width), splats_ply_path)
    
    return {
        "name": image_path.stem,
        "filename": ply_filename,
        "url": f"/generated/{job_id}/splats/{ply_filename}",
        "viewer_url": f"/splats/{ply_filename}",
    }


def _predict_batch_sync(
    model,
    device: str,
    image_paths: list[Path],
    fov: float,
    output_path: Path,
    splats_output_path: Path,
    job_id: str,
) -> list[dict]:
    """Process all images in a batch for maximum GPU utilization."""
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import save_ply, unproject_gaussians
    
    internal_shape = (1536, 1536)
    results = []
    
    # Pre-load and preprocess all images
    images_data = []
    for image_path in image_paths:
        img_pil = Image.open(image_path)
        image = np.array(img_pil)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        height, width = image.shape[:2]
        size = min(width, height)
        f_px = size / (2 * np.tan(np.deg2rad(fov) / 2))
        
        images_data.append({
            "path": image_path,
            "image": image,
            "height": height,
            "width": width,
            "f_px": f_px,
        })
        LOGGER.info(f"Using FOV {fov}° -> focal length {f_px:.2f}px for {image_path.stem}")
    
    # Process images - batch on GPU if memory allows, otherwise sequential
    # For safety, we'll do them sequentially but keep the model hot
    for data in images_data:
        image = data["image"]
        height, width = data["height"], data["width"]
        f_px = data["f_px"]
        image_path = data["path"]
        
        # Preprocess
        image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
        disparity_factor = torch.tensor([f_px / width]).float().to(device)
        
        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )
        
        # Predict - model is hot, this should be fast
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure previous ops complete
            gaussians_ndc = model(image_resized_pt, disparity_factor)
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure inference completes
        
        # Postprocess
        intrinsics = torch.tensor([
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]).float().to(device)
        
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height
        
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
        )
        
        # Save PLY
        ply_filename = f"{image_path.stem}.ply"
        ply_path = output_path / ply_filename
        save_ply(gaussians, f_px, (height, width), ply_path)
        
        splats_ply_path = splats_output_path / ply_filename
        save_ply(gaussians, f_px, (height, width), splats_ply_path)
        
        results.append({
            "name": image_path.stem,
            "filename": ply_filename,
            "url": f"/generated/{job_id}/splats/{ply_filename}",
            "viewer_url": f"/splats/{ply_filename}",
        })
    
    return results


async def run_predict(
    cubefaces_path: Path,
    output_path: Path,
    splats_output_path: Path,
    fov: float,
    faces_to_process: list[str],
    progress: PipelineProgress,
    device: str = "default",
) -> dict:
    """Run Gaussian splat prediction with progress updates.
    
    Uses cached model for fast inference and processes all faces efficiently.
    """
    from sharp.utils import io as sharp_io
    
    progress.message = "Loading Sharp model..."
    await send_progress(progress.job_id, progress)
    
    # Get or load the cached model (fast if already loaded)
    model, device = get_or_load_model()
    
    progress.current_step += 1
    progress.message = f"Model ready on {device.upper()}"
    await send_progress(progress.job_id, progress)
    
    # Find cube face images
    extensions = sharp_io.get_supported_image_extensions()
    image_paths_set = set()
    for ext in extensions:
        for p in cubefaces_path.glob(f"*{ext}"):
            image_paths_set.add(p.resolve())
    
    image_paths = list(image_paths_set)
    
    # Filter by face names
    if faces_to_process:
        image_paths = [
            p for p in image_paths
            if any(face in p.stem.lower() for face in faces_to_process)
        ]
    
    output_path.mkdir(parents=True, exist_ok=True)
    splats_output_path.mkdir(parents=True, exist_ok=True)
    
    progress.message = f"Generating splats for {len(image_paths)} faces..."
    await send_progress(progress.job_id, progress)
    
    # Run batch prediction in thread pool to not block the event loop
    loop = asyncio.get_event_loop()
    generated_splats = await loop.run_in_executor(
        INFERENCE_POOL,
        _predict_batch_sync,
        model,
        device,
        image_paths,
        fov,
        output_path,
        splats_output_path,
        progress.job_id,
    )
    
    # Update progress for all processed faces
    progress.current_step += len(image_paths)
    progress.message = f"Generated {len(generated_splats)} splat files"
    await send_progress(progress.job_id, progress)
    
    return {"splats": generated_splats}


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on server startup for fast first request."""
    LOGGER.info("Server starting up, pre-loading Sharp model...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(INFERENCE_POOL, get_or_load_model)
        LOGGER.info("Model pre-loaded successfully!")
    except Exception as e:
        LOGGER.warning(f"Could not pre-load model (will load on first request): {e}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/model-status")
async def model_status():
    """Check if model is loaded and what device it's using."""
    return {
        "loaded": _cached_model is not None,
        "device": _cached_device,
    }


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for progress updates."""
    await websocket.accept()
    active_connections[job_id] = websocket
    LOGGER.info(f"WebSocket connected: {job_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        LOGGER.info(f"WebSocket disconnected: {job_id}")
    finally:
        if job_id in active_connections:
            del active_connections[job_id]


@app.post("/api/process")
async def process_pipeline(
    file: UploadFile = File(...),
    fov: float = Form(110.0),
    output_size: int = Form(0),
    faces: str = Form("all"),
    generate_splats: bool = Form(True),
    job_id: str = Form(...),
):
    """Process the full pipeline: equirect to cubefaces, then optionally splats."""
    
    # Parse faces
    if faces.lower() == "all":
        faces_to_extract = list(CUBE_FACES.keys())
    else:
        faces_to_extract = [f.strip().lower() for f in faces.split(",")]
    
    # Calculate total steps
    total_steps = 1 + len(faces_to_extract)  # load + faces
    if generate_splats:
        total_steps += 1 + len(faces_to_extract)  # model load + predictions
    
    progress = PipelineProgress(
        job_id=job_id,
        total_steps=total_steps,
        status="running",
    )
    
    try:
        # Create job directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        progress.message = "Uploading image..."
        progress.current_step = 1
        await send_progress(job_id, progress)
        
        input_path = job_output_dir / file.filename
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run equirect to cubefaces
        cubefaces_output = job_output_dir / "cubefaces"
        size = output_size if output_size > 0 else None
        
        cubefaces_result = await run_equirect_to_cubefaces(
            input_path=input_path,
            output_path=cubefaces_output,
            fov=fov,
            size=size,
            faces_to_extract=faces_to_extract,
            progress=progress,
        )
        
        progress.results["cubefaces"] = cubefaces_result
        
        # Run splat prediction if requested
        if generate_splats:
            splats_output = job_output_dir / "splats"
            
            splats_result = await run_predict(
                cubefaces_path=cubefaces_output,
                output_path=splats_output,
                splats_output_path=SPLATS_DIR,
                fov=fov,
                faces_to_process=faces_to_extract,
                progress=progress,
            )
            
            progress.results["splats"] = splats_result
        
        progress.status = "completed"
        progress.message = "Pipeline completed successfully!"
        progress.current_step = progress.total_steps
        await send_progress(job_id, progress)
        
        return {
            "success": True,
            "job_id": job_id,
            "results": progress.results,
        }
        
    except Exception as e:
        LOGGER.exception(f"Pipeline error: {e}")
        progress.status = "error"
        progress.message = str(e)
        await send_progress(job_id, progress)
        return {
            "success": False,
            "error": str(e),
        }


@app.api_route("/generated/{job_id}/{filename:path}", methods=["GET", "HEAD"])
async def serve_generated(job_id: str, filename: str):
    """Serve generated output files (handles both root and subdirectory files)."""
    file_path = OUTPUT_DIR / job_id / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}


@app.get("/api/jobs")
async def list_jobs():
    """List all previous jobs with metadata."""
    from urllib.parse import quote
    
    jobs = []
    
    if not OUTPUT_DIR.exists():
        return {"jobs": []}
    
    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir() or not job_dir.name.startswith("job_"):
            continue
        
        job_id = job_dir.name
        
        # Find the original input image
        input_image = None
        thumbnail_url = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidates = list(job_dir.glob(f"*{ext}"))
            # Exclude cubeface outputs
            candidates = [c for c in candidates if not c.stem.startswith("input_")]
            if candidates:
                input_image = candidates[0].name
                # URL-encode the filename to handle special characters
                thumbnail_url = f"/generated/{job_id}/{quote(input_image)}"
                break
        
        # Count outputs
        cubefaces_dir = job_dir / "cubefaces"
        splats_dir = job_dir / "splats"
        
        cubeface_count = len(list(cubefaces_dir.glob("*.png"))) if cubefaces_dir.exists() else 0
        splat_count = len(list(splats_dir.glob("*.ply"))) if splats_dir.exists() else 0
        
        # Get creation time from job_id (timestamp is second part)
        try:
            timestamp = int(job_id.split("_")[1])
            created_at = timestamp
        except (IndexError, ValueError):
            created_at = int(job_dir.stat().st_mtime * 1000)
        
        jobs.append({
            "job_id": job_id,
            "created_at": created_at,
            "input_image": input_image,
            "thumbnail_url": thumbnail_url,
            "cubeface_count": cubeface_count,
            "splat_count": splat_count,
            "has_splats": splat_count > 0,
        })
    
    # Sort by creation time, newest first
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get full details for a specific job."""
    from urllib.parse import quote
    
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return {"error": "Job not found"}
    
    # Find input image
    input_image = None
    input_image_url = None
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidates = list(job_dir.glob(f"*{ext}"))
        candidates = [c for c in candidates if not c.stem.startswith("input_")]
        if candidates:
            input_image = candidates[0].name
            input_image_url = f"/generated/{job_id}/{quote(input_image)}"
            break
    
    # Get cubefaces
    cubefaces = []
    cubefaces_dir = job_dir / "cubefaces"
    if cubefaces_dir.exists():
        for f in sorted(cubefaces_dir.glob("*.png")):
            face_name = f.stem.replace("input_", "")
            cubefaces.append({
                "name": face_name,
                "filename": f.name,
                "url": f"/generated/{job_id}/cubefaces/{f.name}",
            })
    
    # Get splats
    splats = []
    splats_dir = job_dir / "splats"
    if splats_dir.exists():
        for f in sorted(splats_dir.glob("*.ply")):
            splats.append({
                "name": f.stem,
                "filename": f.name,
                "url": f"/generated/{job_id}/splats/{f.name}",
                "viewer_url": f"/splats/{f.name}",
            })
    
    # Get timestamp
    try:
        timestamp = int(job_id.split("_")[1])
    except (IndexError, ValueError):
        timestamp = int(job_dir.stat().st_mtime * 1000)
    
    return {
        "job_id": job_id,
        "created_at": timestamp,
        "input_image": input_image,
        "input_image_url": input_image_url,
        "cubefaces": {"faces": cubefaces},
        "splats": {"splats": splats},
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its files."""
    import shutil
    
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return {"error": "Job not found"}
    
    try:
        shutil.rmtree(job_dir)
        return {"success": True, "message": f"Job {job_id} deleted"}
    except Exception as e:
        return {"error": f"Failed to delete job: {e}"}


@app.get("/api/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all files for a job."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return {"error": "Job not found"}
    
    files = {
        "cubefaces": [],
        "splats": [],
    }
    
    cubefaces_dir = job_dir / "cubefaces"
    if cubefaces_dir.exists():
        for f in cubefaces_dir.glob("*.png"):
            files["cubefaces"].append({
                "name": f.stem,
                "url": f"/generated/{job_id}/cubefaces/{f.name}",
            })
    
    splats_dir = job_dir / "splats"
    if splats_dir.exists():
        for f in splats_dir.glob("*.ply"):
            files["splats"].append({
                "name": f.stem,
                "url": f"/generated/{job_id}/splats/{f.name}",
            })
    
    return files


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

