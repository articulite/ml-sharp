"""Backend server for Sharp equirectangular to cubemap/splats pipeline.

A FastAPI server with WebSocket progress updates for the Sharp workflow.
Run with: uv run uvicorn server.main:app --reload --port 8765
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


async def run_predict(
    cubefaces_path: Path,
    output_path: Path,
    splats_output_path: Path,
    fov: float,
    faces_to_process: list[str],
    progress: PipelineProgress,
    device: str = "default",
) -> dict:
    """Run Gaussian splat prediction with progress updates."""
    import torch
    import torch.nn.functional as F
    
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply, unproject_gaussians
    
    progress.message = "Checking for Sharp model..."
    await send_progress(progress.job_id, progress)
    
    # Check if model is cached
    model_path = get_cached_model_path()
    if model_path is None:
        raise RuntimeError(
            "Sharp model not found. Please run 'sharp predict' once first to download the model:\n"
            "  sharp predict -i <any_image> -o /tmp/test\n"
            "This will download and cache the model for future use."
        )
    
    # Determine device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    LOGGER.info(f"Using device: {device}")
    progress.message = f"Loading model on {device}..."
    await send_progress(progress.job_id, progress)
    
    # Load model from cache
    LOGGER.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, weights_only=True)
    
    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)
    
    progress.current_step += 1
    progress.message = "Model loaded successfully"
    await send_progress(progress.job_id, progress)
    
    # Find cube face images (use set to avoid duplicates from case-insensitive matching)
    extensions = sharp_io.get_supported_image_extensions()
    image_paths_set = set()
    for ext in extensions:
        for p in cubefaces_path.glob(f"*{ext}"):
            # Use resolve() to get canonical path, avoiding duplicates
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
    generated_splats = []
    
    internal_shape = (1536, 1536)
    
    for i, image_path in enumerate(image_paths):
        progress.current_step += 1
        progress.message = f"Generating splats for {image_path.stem} ({i + 1}/{len(image_paths)})"
        await send_progress(progress.job_id, progress)
        
        # Load image directly (skip sharp_io.load_rgb to avoid EXIF warning since we compute focal length from FOV)
        img_pil = Image.open(image_path)
        image = np.array(img_pil)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        height, width = image.shape[:2]
        
        # Compute focal length from FOV (this is what the user set in the dashboard)
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
            gaussians_ndc = gaussian_predictor(image_resized_pt, disparity_factor)
        
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
        
        generated_splats.append({
            "name": image_path.stem,
            "filename": ply_filename,
            "url": f"/generated/{progress.job_id}/splats/{ply_filename}",
            "viewer_url": f"/splats/{ply_filename}",
        })
        
        await asyncio.sleep(0.05)
    
    return {"splats": generated_splats}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


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


@app.get("/generated/{job_id}/{subdir}/{filename}")
async def serve_generated(job_id: str, subdir: str, filename: str):
    """Serve generated output files."""
    file_path = OUTPUT_DIR / job_id / subdir / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}


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

