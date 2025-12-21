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


def get_fov_from_ply(plydata: PlyData) -> float:
    """Extract FOV from PLY intrinsics metadata."""
    try:
        intrinsic = plydata["intrinsic"]
        image_size = plydata["image_size"]
        
        # Intrinsic is a 3x3 matrix flattened: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = float(intrinsic.data[0][0])
        img_width = int(image_size.data[0][0])
        
        # FOV = 2 * atan(width / (2 * fx))
        fov_rad = 2 * np.arctan(img_width / (2 * fx))
        fov_deg = np.rad2deg(fov_rad)
        
        return fov_deg
    except Exception as e:
        LOGGER.warning(f"Could not extract FOV from PLY: {e}")
        return 110.0  # Default fallback


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


def get_edge_gaussians(positions: np.ndarray, face: str, edge_face: str, fov: float, edge_width: float = 0.15) -> np.ndarray:
    """
    Get mask of Gaussians near the boundary with another face.
    
    Args:
        positions: World-space positions (N, 3)
        face: The face these Gaussians belong to
        edge_face: The adjacent face we're checking boundary with
        fov: Field of view in degrees
        edge_width: How far from edge to consider (as fraction of FOV, 0.15 = 15%)
    
    Returns:
        Boolean mask of Gaussians near the edge
    """
    # Get directions to each Gaussian
    depths = np.linalg.norm(positions, axis=1)
    valid = depths > 1e-6
    dirs = np.zeros_like(positions)
    dirs[valid] = positions[valid] / depths[valid, np.newaxis]
    
    # Get face centers
    face_yaw, face_pitch = FACE_ORIENTATIONS[face]
    edge_yaw, edge_pitch = FACE_ORIENTATIONS[edge_face]
    
    # Compute angle from face center for each Gaussian
    face_center = spherical_to_cartesian(face_yaw, face_pitch)
    dots_to_center = dirs @ face_center
    angles_from_center = np.arccos(np.clip(dots_to_center, -1, 1))
    
    # Compute angle toward edge face
    edge_center = spherical_to_cartesian(edge_yaw, edge_pitch)
    dots_to_edge = dirs @ edge_center
    angles_to_edge = np.arccos(np.clip(dots_to_edge, -1, 1))
    
    # Near edge = close to edge face and within FOV of current face
    half_fov = np.deg2rad(fov / 2)
    edge_threshold = np.deg2rad(fov / 2 * (1 - edge_width))  # Inner boundary of edge zone
    
    # Gaussian is in edge zone if:
    # 1. It's within the face's FOV
    # 2. It's close to the edge (angle from center > threshold)
    # 3. It's pointing toward the edge face
    in_fov = angles_from_center < half_fov
    near_edge = angles_from_center > edge_threshold
    toward_edge = angles_to_edge < np.deg2rad(90)  # Pointing more toward edge than away
    
    return valid & in_fov & near_edge & toward_edge


def direction_to_hash(direction: np.ndarray, resolution: int = 1000) -> tuple:
    """Convert a 3D direction to a hash key for spatial bucketing."""
    # Use spherical coordinates binned to resolution
    x, y, z = direction
    theta = np.arctan2(x, z)  # [-pi, pi]
    phi = np.arcsin(np.clip(y, -1, 1))  # [-pi/2, pi/2]
    
    # Bin to integer grid
    theta_bin = int((theta + np.pi) / (2 * np.pi) * resolution) % resolution
    phi_bin = int((phi + np.pi/2) / np.pi * resolution) % resolution
    
    return (theta_bin, phi_bin)


def stitch_boundaries(face_data: dict, fov: float = 110.0, blend_strength: float = 0.8) -> dict:
    """
    Stitch Gaussian boundaries between adjacent faces.
    
    Uses spatial hashing for efficient nearest-direction matching.
    
    Args:
        face_data: Dict with face -> {"world": positions, ...}
        fov: Field of view in degrees
        blend_strength: How aggressively to blend (0-1, higher = more blending)
    
    Returns:
        Updated face_data with stitched positions
    """
    LOGGER.info("Stitching face boundaries...")
    
    # Work with copies
    stitched = {}
    for face, data in face_data.items():
        stitched[face] = data["world"].copy()
    
    total_stitched = 0
    hash_resolution = 2000  # Higher = more precise matching
    
    for face_a, face_b in FACE_ADJACENCIES:
        if face_a not in face_data or face_b not in face_data:
            continue
        
        pos_a = stitched[face_a]
        pos_b = stitched[face_b]
        
        # Get Gaussians near the shared edge
        edge_mask_a = get_edge_gaussians(pos_a, face_a, face_b, fov)
        edge_mask_b = get_edge_gaussians(pos_b, face_b, face_a, fov)
        
        edge_indices_a = np.where(edge_mask_a)[0]
        edge_indices_b = np.where(edge_mask_b)[0]
        
        if len(edge_indices_a) == 0 or len(edge_indices_b) == 0:
            LOGGER.info(f"  {face_a}-{face_b}: no edge gaussians found")
            continue
        
        LOGGER.info(f"  {face_a}-{face_b}: {len(edge_indices_a)} / {len(edge_indices_b)} edge gaussians")
        
        # Build spatial hash for face B's edge Gaussians
        edge_pos_b = pos_b[edge_indices_b]
        depths_b = np.linalg.norm(edge_pos_b, axis=1, keepdims=True)
        dirs_b = edge_pos_b / np.maximum(depths_b, 1e-6)
        
        # Hash: direction -> list of (local_idx, depth, position)
        b_hash = {}
        for local_idx, (dir_b, depth_b, pos) in enumerate(zip(dirs_b, depths_b.flatten(), edge_pos_b)):
            key = direction_to_hash(dir_b, hash_resolution)
            if key not in b_hash:
                b_hash[key] = []
            b_hash[key].append((local_idx, depth_b, pos))
        
        # Get face centers for blend weight calculation
        face_a_yaw, face_a_pitch = FACE_ORIENTATIONS[face_a]
        face_a_center = spherical_to_cartesian(face_a_yaw, face_a_pitch)
        half_fov_rad = np.deg2rad(fov / 2)
        
        # For each edge Gaussian in A, find match in B using spatial hash
        edge_pos_a = pos_a[edge_indices_a]
        depths_a = np.linalg.norm(edge_pos_a, axis=1, keepdims=True)
        dirs_a = edge_pos_a / np.maximum(depths_a, 1e-6)
        
        match_threshold = np.cos(np.deg2rad(2.0))  # Within 2 degrees
        n_matched = 0
        
        for local_idx_a, (dir_a, depth_a, idx_a) in enumerate(zip(dirs_a, depths_a.flatten(), edge_indices_a)):
            key = direction_to_hash(dir_a, hash_resolution)
            
            # Check this cell and neighbors (3x3 neighborhood)
            best_match = None
            best_sim = match_threshold
            
            for dk_theta in [-1, 0, 1]:
                for dk_phi in [-1, 0, 1]:
                    neighbor_key = ((key[0] + dk_theta) % hash_resolution, 
                                   (key[1] + dk_phi) % hash_resolution)
                    if neighbor_key not in b_hash:
                        continue
                    
                    for local_idx_b, depth_b, pos_b in b_hash[neighbor_key]:
                        dir_b = pos_b / max(np.linalg.norm(pos_b), 1e-6)
                        sim = np.dot(dir_a, dir_b)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = (local_idx_b, depth_b, pos_b)
            
            if best_match is None:
                continue
            
            local_idx_b, depth_b, pos_b_matched = best_match
            idx_b = edge_indices_b[local_idx_b]
            
            # Compute blend weight based on distance from face center
            dist_from_center = np.arccos(np.clip(np.dot(dir_a, face_a_center), -1, 1)) / half_fov_rad
            # More blending closer to edge (dist_from_center closer to 1.0)
            blend_weight = np.clip((dist_from_center - 0.7) / 0.3, 0, 1) * blend_strength
            
            # Get current positions
            p_a = stitched[face_a][idx_a].copy()
            p_b = stitched[face_b][idx_b].copy()
            
            # Compute target: average position (preserving direction, averaging depth)
            avg_depth = (depth_a + depth_b) / 2
            avg_dir = (dir_a + dirs_b[local_idx_b]) / 2
            avg_dir = avg_dir / max(np.linalg.norm(avg_dir), 1e-6)
            avg_pos = avg_dir * avg_depth
            
            # Blend toward average
            stitched[face_a][idx_a] = p_a + blend_weight * (avg_pos - p_a)
            stitched[face_b][idx_b] = p_b + blend_weight * (avg_pos - p_b)
            n_matched += 1
        
        LOGGER.info(f"    -> {n_matched} pairs stitched")
        total_stitched += n_matched
    
    LOGGER.info(f"Total stitch operations: {total_stitched}")
    
    return stitched


def harmonize_job(job_path: Path, fov: float = None) -> dict:
    """Harmonize all splats in a job folder.
    
    Args:
        job_path: Path to job folder
        fov: Field of view override. If None, auto-detects from PLY metadata.
    """
    
    splats_dir = job_path / "splats"
    if not splats_dir.exists():
        raise FileNotFoundError(f"No splats directory: {splats_dir}")
    
    output_dir = job_path / "splats_harmonized"
    output_dir.mkdir(exist_ok=True)
    
    # Load all faces
    LOGGER.info("Loading PLY files...")
    face_data = {}
    detected_fov = None
    
    for face in CUBE_FACES:
        ply_path = splats_dir / f"input_{face}.ply"
        if not ply_path.exists():
            LOGGER.warning(f"  Missing: {face}")
            continue
        
        pos_local, plydata = load_ply(ply_path)
        pos_world = to_world(pos_local, face)
        
        # Auto-detect FOV from first PLY file
        if detected_fov is None:
            detected_fov = get_fov_from_ply(plydata)
        
        face_data[face] = {
            "local": pos_local,
            "world": pos_world,
            "ply": plydata,
        }
        LOGGER.info(f"  {face}: {len(pos_local)} gaussians")
    
    # Use provided FOV or detected FOV
    if fov is None:
        fov = detected_fov if detected_fov else 110.0
    
    LOGGER.info(f"Using FOV: {fov:.1f}°")
    
    # Check if FOV is sufficient for stitching
    overlap_deg = fov - 90.0
    if overlap_deg < 5:
        LOGGER.warning(f"⚠️  FOV {fov:.1f}° gives only {overlap_deg:.1f}° overlap!")
        LOGGER.warning(f"   For good stitching, use FOV >= 100° (10°+ overlap)")
        LOGGER.warning(f"   Regenerate splats with --fov 110 for best results")
    
    if len(face_data) < 2:
        raise ValueError("Need at least 2 faces")
    
    # Step 1: Global scale correction (brings faces to similar scale)
    LOGGER.info("Step 1: Global scale correction...")
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
    
    # Apply global scale correction
    for face, data in face_data.items():
        scale = scales.get(face, 1.0)
        face_data[face]["world"] = data["world"] * scale
    
    # Compute improvement from global scaling
    if observations:
        err_before = np.sqrt(np.mean([(za - zb)**2 for _, _, za, zb in observations]))
        err_after_scale = np.sqrt(np.mean([(scales[fa]*za - scales[fb]*zb)**2 
                                     for fa, fb, za, zb in observations]))
        LOGGER.info(f"RMS error after scaling: {err_before:.4f} -> {err_after_scale:.4f}")
    
    # Step 2: Local boundary stitching (closes remaining gaps)
    LOGGER.info("\nStep 2: Boundary stitching...")
    stitched_world = stitch_boundaries(face_data, fov=fov, blend_strength=0.9)
    
    # Save results
    LOGGER.info("\nSaving harmonized PLY files...")
    
    for face, data in face_data.items():
        corrected_world = stitched_world[face]
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

