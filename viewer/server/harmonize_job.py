"""Standalone script to harmonize splats for a job.

Usage:
    python harmonize_job.py <job_id>
    python harmonize_job.py job_1766294367815_2vhuxccqc

This will:
1. Load all PLY files from public/generated/<job_id>/splats/
2. Compute scale corrections using overlap correspondences
3. Save harmonized PLY files to public/generated/<job_id>/splats_harmonized/
4. Copy harmonized files to public/splats/ for viewer
"""

import sys
from pathlib import Path

# Add parent paths for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import logging
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("ERROR: plyfile not installed. Run: pip install plyfile")
    sys.exit(1)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# Cube face definitions
CUBE_FACES = ["front", "back", "left", "right", "top", "bottom"]

FACE_ADJACENCIES = [
    ("front", "right"),
    ("front", "left"),
    ("front", "top"),
    ("front", "bottom"),
    ("back", "right"),
    ("back", "left"),
    ("back", "top"),
    ("back", "bottom"),
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
]

# Face orientations: (yaw, pitch) in degrees
FACE_ORIENTATIONS = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "top": (0, 90),
    "bottom": (0, -90),
}


def get_rotation_matrix(face: str) -> np.ndarray:
    """Get rotation matrix for face-local to world transform."""
    yaw_deg, pitch_deg = FACE_ORIENTATIONS[face]
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    
    R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    
    return R_yaw @ R_pitch


def to_world(positions: np.ndarray, face: str) -> np.ndarray:
    """Transform positions from face-local to world coordinates."""
    R = get_rotation_matrix(face)
    pos = positions.copy()
    pos[:, 1] *= -1  # Flip Y
    return (R @ pos.T).T


def from_world(positions: np.ndarray, face: str) -> np.ndarray:
    """Transform positions from world to face-local coordinates."""
    R = get_rotation_matrix(face)
    pos = (R.T @ positions.T).T
    pos[:, 1] *= -1  # Flip Y back
    return pos


def spherical_to_cartesian(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Convert spherical to cartesian unit vector."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    return np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ])


def project_to_face(world_dir: np.ndarray, face: str, fov_deg: float) -> tuple | None:
    """Project world direction to face pixel coords. Returns None if outside."""
    R = get_rotation_matrix(face)
    local = R.T @ world_dir
    
    if local[2] <= 0:
        return None
    
    u = local[0] / local[2]
    v = local[1] / local[2]
    
    half_fov = np.tan(np.deg2rad(fov_deg / 2))
    if abs(u) > half_fov or abs(v) > half_fov:
        return None
    
    return (u / half_fov, v / half_fov)


def find_overlap_directions(face_a: str, face_b: str, fov_deg: float = 110, n: int = 40) -> list:
    """Find world directions visible in both faces."""
    ya, pa = FACE_ORIENTATIONS[face_a]
    yb, pb = FACE_ORIENTATIONS[face_b]
    
    # Midpoint between face centers
    mid_yaw = (ya + yb) / 2
    mid_pitch = (pa + pb) / 2
    
    # Handle wraparound
    if abs(ya - yb) > 180:
        mid_yaw = ((ya + 360 + yb) / 2) % 360
        if mid_yaw > 180:
            mid_yaw -= 360
    
    directions = []
    half = fov_deg / 2
    
    for dy in np.linspace(-half, half, n):
        for dp in np.linspace(-half, half, n):
            if pa == pb:  # Horizontal neighbors
                test_yaw = mid_yaw + dy * 0.3
                test_pitch = dp * 0.8
            elif ya == yb:  # Vertical neighbors
                test_yaw = ya + dy * 0.8
                test_pitch = mid_pitch + dp * 0.3
            else:  # Diagonal
                test_yaw = mid_yaw + dy * 0.5
                test_pitch = mid_pitch + dp * 0.5
            
            d = spherical_to_cartesian(test_yaw, test_pitch)
            
            if project_to_face(d, face_a, fov_deg) and project_to_face(d, face_b, fov_deg):
                directions.append(d)
    
    return directions


def get_depth_at_directions(positions: np.ndarray, directions: list, threshold_deg: float = 5.0) -> list:
    """Get depth of nearest Gaussian for each direction."""
    if len(positions) == 0:
        return [None] * len(directions)
    
    depths = np.linalg.norm(positions, axis=1)
    valid = depths > 1e-6
    
    dirs = np.zeros_like(positions)
    dirs[valid] = positions[valid] / depths[valid, np.newaxis]
    
    cos_thresh = np.cos(np.deg2rad(threshold_deg))
    
    results = []
    for qdir in directions:
        dots = dirs @ qdir
        best = np.argmax(dots)
        if dots[best] >= cos_thresh and valid[best]:
            results.append(depths[best])
        else:
            results.append(None)
    
    return results


def load_ply(path: Path) -> tuple:
    """Load positions and full plydata."""
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    positions = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    return positions, plydata


def save_ply(plydata: PlyData, new_positions: np.ndarray, path: Path):
    """Save PLY with new positions, preserving original format."""
    v = plydata["vertex"]
    dtype = v.data.dtype
    new_data = np.empty(len(v.data), dtype=dtype)
    
    for name in dtype.names:
        new_data[name] = v.data[name]
    
    new_data["x"] = new_positions[:, 0].astype(np.float32)
    new_data["y"] = new_positions[:, 1].astype(np.float32)
    new_data["z"] = new_positions[:, 2].astype(np.float32)
    
    new_vertex = PlyElement.describe(new_data, "vertex")
    other = [el for el in plydata.elements if el.name != "vertex"]
    
    # Write in binary_little_endian format (same as original Sharp output)
    new_plydata = PlyData([new_vertex] + other, text=False, byte_order='<')
    new_plydata.write(path)


def harmonize_job(job_path: Path, fov: float = 110.0) -> dict:
    """Harmonize all splats in a job folder."""
    
    splats_dir = job_path / "splats"
    if not splats_dir.exists():
        raise FileNotFoundError(f"No splats directory: {splats_dir}")
    
    output_dir = job_path / "splats_harmonized"
    output_dir.mkdir(exist_ok=True)
    
    # Load all faces
    LOGGER.info("Loading PLY files...")
    face_data = {}
    
    for face in CUBE_FACES:
        ply_path = splats_dir / f"input_{face}.ply"
        if not ply_path.exists():
            LOGGER.warning(f"  Missing: {face}")
            continue
        
        pos_local, plydata = load_ply(ply_path)
        pos_world = to_world(pos_local, face)
        
        face_data[face] = {
            "local": pos_local,
            "world": pos_world,
            "ply": plydata,
        }
        LOGGER.info(f"  {face}: {len(pos_local)} gaussians")
    
    if len(face_data) < 2:
        raise ValueError("Need at least 2 faces")
    
    # Collect depth observations from overlaps
    LOGGER.info("Finding overlap correspondences...")
    observations = []
    
    for fa, fb in FACE_ADJACENCIES:
        if fa not in face_data or fb not in face_data:
            continue
        
        dirs = find_overlap_directions(fa, fb, fov)
        if not dirs:
            continue
        
        depths_a = get_depth_at_directions(face_data[fa]["world"], dirs)
        depths_b = get_depth_at_directions(face_data[fb]["world"], dirs)
        
        count = 0
        for za, zb in zip(depths_a, depths_b):
            if za and zb and 0.1 < za < 100 and 0.1 < zb < 100:
                observations.append((fa, fb, za, zb))
                count += 1
        
        LOGGER.info(f"  {fa}-{fb}: {count} correspondences")
    
    LOGGER.info(f"Total observations: {len(observations)}")
    
    if len(observations) < 5:
        LOGGER.warning("Too few observations - using uniform scales")
        scales = {f: 1.0 for f in face_data}
    else:
        # Solve for scales using least squares
        scales = solve_scales(observations, list(face_data.keys()))
    
    LOGGER.info("Scale factors:")
    for face, scale in sorted(scales.items()):
        LOGGER.info(f"  {face}: {scale:.4f}")
    
    # Compute improvement
    if observations:
        err_before = np.sqrt(np.mean([(za - zb)**2 for _, _, za, zb in observations]))
        err_after = np.sqrt(np.mean([(scales[fa]*za - scales[fb]*zb)**2 
                                     for fa, fb, za, zb in observations]))
        LOGGER.info(f"RMS error: {err_before:.4f} -> {err_after:.4f}")
    
    # Apply and save
    LOGGER.info("Saving harmonized PLY files...")
    
    for face, data in face_data.items():
        scale = scales.get(face, 1.0)
        corrected_world = data["world"] * scale
        corrected_local = from_world(corrected_world, face)
        
        out_path = output_dir / f"input_{face}.ply"
        save_ply(data["ply"], corrected_local, out_path)
        
        # Verify file was written correctly
        if out_path.exists():
            size = out_path.stat().st_size
            LOGGER.info(f"  {out_path.name} ({size} bytes)")
            if size < 1000:
                LOGGER.warning(f"  WARNING: {out_path.name} seems too small!")
        else:
            LOGGER.error(f"  FAILED to write {out_path.name}")
    
    return {
        "scales": scales,
        "output_dir": output_dir,
        "n_observations": len(observations),
    }


def solve_scales(observations: list, faces: list, ref: str = "front") -> dict:
    """Solve least squares for scale factors."""
    face_idx = {f: i for i, f in enumerate(faces)}
    ref_idx = face_idx.get(ref, 0)
    other = [f for f in faces if f != ref]
    other_idx = {f: i for i, f in enumerate(other)}
    
    n = len(observations)
    rows, cols, data = [], [], []
    b = np.zeros(n)
    
    for i, (fa, fb, za, zb) in enumerate(observations):
        ia = face_idx[fa]
        ib = face_idx[fb]
        
        if ia == ref_idx:
            cols.append(other_idx[fb])
            rows.append(i)
            data.append(-zb)
            b[i] = -za
        elif ib == ref_idx:
            cols.append(other_idx[fa])
            rows.append(i)
            data.append(za)
            b[i] = zb
        else:
            cols.append(other_idx[fa])
            rows.append(i)
            data.append(za)
            cols.append(other_idx[fb])
            rows.append(i)
            data.append(-zb)
            b[i] = 0
    
    A = csr_matrix((data, (rows, cols)), shape=(n, len(other)))
    
    # Iteratively reweighted least squares for robustness
    weights = np.ones(n)
    for _ in range(3):
        W = np.sqrt(weights)
        result = lsqr(A.multiply(W[:, None]), b * W)
        x = result[0]
        
        residuals = A @ x - b
        mad = np.median(np.abs(residuals))
        if mad > 1e-6:
            weights = 1 / (1 + (residuals / (2 * mad))**2)
    
    scales = {ref: 1.0}
    for f, idx in other_idx.items():
        scales[f] = float(x[idx])
    
    return scales


def copy_to_viewer(job_path: Path, viewer_splats: Path):
    """Copy harmonized splats to viewer directory."""
    import shutil
    
    harm_dir = job_path / "splats_harmonized"
    if not harm_dir.exists():
        return
    
    for ply in harm_dir.glob("*.ply"):
        dest = viewer_splats / ply.name
        shutil.copy2(ply, dest)
        LOGGER.info(f"Copied to viewer: {dest.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python harmonize_job.py <job_id>")
        print("       python harmonize_job.py --all  (process all jobs)")
        sys.exit(1)
    
    viewer_dir = Path(__file__).parent.parent
    generated_dir = viewer_dir / "public" / "generated"
    viewer_splats = viewer_dir / "public" / "splats"
    
    if sys.argv[1] == "--all":
        # Process all jobs
        jobs = sorted(generated_dir.glob("job_*"))
        for job_path in jobs:
            if (job_path / "splats").exists():
                LOGGER.info(f"\n{'='*50}")
                LOGGER.info(f"Processing: {job_path.name}")
                LOGGER.info("="*50)
                try:
                    harmonize_job(job_path)
                    copy_to_viewer(job_path, viewer_splats)
                except Exception as e:
                    LOGGER.error(f"Failed: {e}")
    else:
        job_id = sys.argv[1]
        job_path = generated_dir / job_id
        
        if not job_path.exists():
            LOGGER.error(f"Job not found: {job_path}")
            sys.exit(1)
        
        result = harmonize_job(job_path)
        copy_to_viewer(job_path, viewer_splats)
        
        print(f"\nHarmonized splats saved to: {result['output_dir']}")
        print("Splats copied to viewer/public/splats/ - refresh viewer to see results")


if __name__ == "__main__":
    main()

