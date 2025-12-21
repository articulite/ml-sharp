#!/usr/bin/env python3
"""Test script for cubemap-coherent depth prediction.

This script tests the two-pass prediction approach that produces
geometrically consistent depth across cube faces.

Usage:
    # From project root:
    uv run python tests/test_cubemap_coherent.py --job JOB_ID
    
    # Or with a custom path:
    uv run python tests/test_cubemap_coherent.py --input /path/to/cubefaces --fov 110
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)


def find_job_cubefaces(job_id: str) -> Path:
    """Find cubefaces directory for a job."""
    viewer_generated = Path(__file__).parent.parent / "viewer" / "public" / "generated"
    job_dir = viewer_generated / job_id / "cubefaces"
    if job_dir.exists():
        return job_dir
    raise FileNotFoundError(f"Job cubefaces not found: {job_dir}")


def get_fov_from_splat(job_id: str) -> float:
    """Read FOV from existing splat file metadata."""
    from plyfile import PlyData
    
    viewer_generated = Path(__file__).parent.parent / "viewer" / "public" / "generated"
    splats_dir = viewer_generated / job_id / "splats"
    
    if not splats_dir.exists():
        return None
    
    ply_files = list(splats_dir.glob("*.ply"))
    if not ply_files:
        return None
    
    try:
        plydata = PlyData.read(ply_files[0])
        intrinsic = plydata["intrinsic"]
        image_size = plydata["image_size"]
        
        fx = float(intrinsic.data[0][0])
        img_width = int(image_size.data[0][0])
        
        fov_rad = 2 * np.arctan(img_width / (2 * fx))
        fov_deg = np.rad2deg(fov_rad)
        return fov_deg
    except Exception as e:
        LOGGER.warning(f"Could not read FOV from PLY: {e}")
        return None


def compare_depth_consistency(independent_dir: Path, coherent_dir: Path) -> dict:
    """Compare depth consistency between independent and coherent predictions.
    
    Measures depth difference at EDGE BOUNDARIES where faces actually meet.
    This is the real test - do the floor/walls align at the seams?
    """
    from plyfile import PlyData
    
    # Edge definitions: (face1, face2, axis, face1_side, face2_side)
    # axis: 'x' or 'z' - the axis along which to measure edge
    # side: 'pos' or 'neg' - which side of that face is the edge
    edge_pairs = [
        ('front', 'left', 'x', 'neg', 'pos'),   # front's left edge meets left's right edge
        ('front', 'right', 'x', 'pos', 'neg'),  # front's right edge meets right's left edge
        ('left', 'back', 'x', 'neg', 'pos'),    # left's left edge meets back's right edge
        ('right', 'back', 'x', 'pos', 'neg'),   # right's right edge meets back's left edge
    ]
    
    def load_positions(ply_path: Path) -> np.ndarray:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        return np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    def get_edge_depths(positions: np.ndarray, axis: str, side: str, percentile: float = 10) -> np.ndarray:
        """Get depth values for Gaussians near the specified edge."""
        if axis == 'x':
            coord = positions[:, 0]
        else:  # z
            coord = positions[:, 2]
        
        # Get threshold for edge (top/bottom percentile of that axis)
        if side == 'pos':
            threshold = np.percentile(coord, 100 - percentile)
            mask = coord >= threshold
        else:
            threshold = np.percentile(coord, percentile)
            mask = coord <= threshold
        
        edge_positions = positions[mask]
        # Return Y coordinate (height/depth) at the edge
        return edge_positions[:, 1] if len(edge_positions) > 0 else np.array([0])
    
    results = {'independent': {}, 'coherent': {}}
    
    for method, dir_path in [('independent', independent_dir), ('coherent', coherent_dir)]:
        for face1, face2, axis, side1, side2 in edge_pairs:
            ply1 = dir_path / f"input_{face1}.ply"
            ply2 = dir_path / f"input_{face2}.ply"
            
            if not ply1.exists() or not ply2.exists():
                continue
            
            pos1 = load_positions(ply1)
            pos2 = load_positions(ply2)
            
            # Get Y values (floor height) at the shared edge
            edge_y1 = get_edge_depths(pos1, axis, side1)
            edge_y2 = get_edge_depths(pos2, axis, side2)
            
            # Compare median floor height at edge
            floor1 = np.percentile(edge_y1, 10)  # Low Y = floor
            floor2 = np.percentile(edge_y2, 10)
            
            # Absolute difference in floor height
            floor_diff = abs(floor1 - floor2)
            
            key = f"{face1}-{face2}"
            results[method][key] = {
                'floor1': floor1,
                'floor2': floor2,
                'floor_diff': floor_diff,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test cubemap-coherent prediction")
    parser.add_argument("--job", type=str, help="Job ID to use cubefaces from")
    parser.add_argument("--input", type=Path, help="Path to cubefaces folder")
    parser.add_argument("--output", type=Path, default=Path("test_output"), 
                        help="Output directory")
    parser.add_argument("--fov", type=float, default=None, 
                        help="FOV in degrees (auto-detected from splats if not provided)")
    parser.add_argument("--device", type=str, default="default",
                        help="Device: cuda, mps, cpu")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip running independent prediction for comparison")
    args = parser.parse_args()
    
    # Find input
    if args.job:
        input_path = find_job_cubefaces(args.job)
        LOGGER.info(f"Using job {args.job}: {input_path}")
        
        # Try to get FOV from existing splats
        if args.fov is None:
            detected_fov = get_fov_from_splat(args.job)
            if detected_fov:
                args.fov = detected_fov
                LOGGER.info(f"Auto-detected FOV from splats: {args.fov:.1f}°")
    elif args.input:
        input_path = args.input
    else:
        parser.error("Either --job or --input must be provided")
    
    if args.fov is None:
        args.fov = 110.0
        LOGGER.info(f"Using default FOV: {args.fov}°")
    
    overlap = args.fov - 90
    LOGGER.info(f"FOV: {args.fov}° → Overlap: {overlap}°")
    if overlap < 10:
        LOGGER.warning(f"⚠️  Low overlap ({overlap}°)! Coherence conditioning may be less effective.")
        LOGGER.warning("   Consider regenerating cubefaces with FOV >= 100°")
    
    # Find device
    if args.device == "default":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    LOGGER.info(f"Using device: {args.device}")
    
    # Import and run
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply
    from sharp.cli.predict_cubemap import CubemapPredictor, DEFAULT_MODEL_URL
    
    # Load model
    LOGGER.info("Loading model...")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(args.device)
    
    # Find cube face images
    extensions = io.get_supported_image_extensions()
    face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    images = {}
    for face in face_names:
        for ext in extensions:
            candidates = list(input_path.glob(f"*{face}*{ext}")) + \
                        list(input_path.glob(f"*{face.upper()}*{ext}"))
            if candidates:
                image, _, _ = io.load_rgb(candidates[0])
                images[face] = image
                LOGGER.info(f"  Found {face}: {candidates[0].name}")
                break
    
    if len(images) == 0:
        LOGGER.error("No cube face images found!")
        return 1
    
    LOGGER.info(f"Found {len(images)} cube faces")
    
    # Create output directories
    output_path = args.output
    output_path.mkdir(exist_ok=True, parents=True)
    coherent_dir = output_path / "coherent"
    coherent_dir.mkdir(exist_ok=True)
    
    # Run two-pass coherent prediction
    LOGGER.info("")
    LOGGER.info("="*60)
    LOGGER.info(" RUNNING TWO-PASS COHERENT PREDICTION")
    LOGGER.info("="*60)
    
    cubemap_predictor = CubemapPredictor(predictor, torch.device(args.device), fov=args.fov)
    results = cubemap_predictor.predict_coherent(images)
    
    # Save results
    for face, (gaussians, f_px, height, width) in results.items():
        ply_path = coherent_dir / f"input_{face}.ply"
        save_ply(gaussians, f_px, (height, width), ply_path)
        LOGGER.info(f"Saved: {ply_path}")
    
    # Run independent prediction for comparison
    if not args.skip_compare:
        LOGGER.info("")
        LOGGER.info("="*60)
        LOGGER.info(" RUNNING INDEPENDENT PREDICTION (COMPARISON)")
        LOGGER.info("="*60)
        
        from sharp.cli.predict import predict_image
        
        independent_dir = output_path / "independent"
        independent_dir.mkdir(exist_ok=True)
        
        for face, image in images.items():
            height, width = image.shape[:2]
            size = min(width, height)
            f_px = size / (2 * np.tan(np.deg2rad(args.fov) / 2))
            
            gaussians = predict_image(predictor, image, f_px, torch.device(args.device))
            ply_path = independent_dir / f"input_{face}.ply"
            save_ply(gaussians, f_px, (height, width), ply_path)
            LOGGER.info(f"Saved: {ply_path}")
        
        # Compare consistency
        LOGGER.info("")
        LOGGER.info("="*60)
        LOGGER.info(" EDGE ALIGNMENT COMPARISON (Floor Height at Seams)")
        LOGGER.info("="*60)
        
        comparison = compare_depth_consistency(independent_dir, coherent_dir)
        
        LOGGER.info("")
        LOGGER.info("Floor height difference at face boundaries:")
        LOGGER.info("-"*60)
        
        for pair in comparison['independent']:
            ind = comparison['independent'][pair]['floor_diff']
            coh = comparison['coherent'].get(pair, {}).get('floor_diff', float('nan'))
            improvement = ind - coh if not np.isnan(coh) else 0
            
            status = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
            LOGGER.info(f"  {pair:15s}: Independent={ind:6.3f}  Coherent={coh:6.3f}  {status} ({improvement:+.3f})")
        
        # Summary
        ind_avg = np.mean([v['floor_diff'] for v in comparison['independent'].values()])
        coh_avg = np.mean([v['floor_diff'] for v in comparison['coherent'].values()])
        
        LOGGER.info("")
        LOGGER.info(f"  Average:         Independent={ind_avg:6.3f}  Coherent={coh_avg:6.3f}")
        LOGGER.info(f"  (Lower is better - 0 means perfect edge alignment)")
        LOGGER.info("")
    
    # Copy coherent results to job folder for viewer
    if args.job:
        viewer_generated = Path(__file__).parent.parent / "viewer" / "public" / "generated"
        job_coherent_dir = viewer_generated / args.job / "splats_coherent"
        job_coherent_dir.mkdir(exist_ok=True)
        
        import shutil
        for ply_file in coherent_dir.glob("*.ply"):
            shutil.copy(ply_file, job_coherent_dir / ply_file.name)
        LOGGER.info(f"Copied coherent results to: {job_coherent_dir}")
    
    LOGGER.info("")
    LOGGER.info("="*60)
    LOGGER.info(" DONE!")
    LOGGER.info("="*60)
    LOGGER.info(f"Output: {output_path.absolute()}")
    LOGGER.info(f"  coherent/     - Two-pass coherent prediction")
    if not args.skip_compare:
        LOGGER.info(f"  independent/  - Single-pass independent prediction")
    if args.job:
        LOGGER.info(f"")
        LOGGER.info(f"View in browser: Load job {args.job}, toggle 'splats_coherent' folder")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

