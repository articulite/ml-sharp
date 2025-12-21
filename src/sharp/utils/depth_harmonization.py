"""Depth harmonization for cubemap Gaussian splats.

Solves for per-face scale factors using overlap correspondences to ensure
consistent geometry across cube face boundaries.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

LOGGER = logging.getLogger(__name__)

# Cube face definitions: (yaw, pitch) in degrees
CUBE_FACES = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "top": (0, 90),
    "bottom": (0, -90),
}

# Face adjacency: (face_a, face_b, edge_description)
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


@dataclass
class HarmonizationResult:
    """Result of depth harmonization."""
    scale_factors: dict[str, float]
    residual_before: float
    residual_after: float
    n_correspondences: int
    correspondences_per_pair: dict[tuple[str, str], int]


def spherical_to_cartesian(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Convert spherical coordinates (yaw, pitch) to unit vector."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    
    return np.array([x, y, z])


def cartesian_to_spherical(xyz: np.ndarray) -> tuple[float, float]:
    """Convert unit vector to spherical coordinates (yaw_deg, pitch_deg)."""
    x, y, z = xyz
    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1, 1))
    return np.rad2deg(theta), np.rad2deg(phi)


def get_face_rotation_matrix(face: str) -> np.ndarray:
    """Get the rotation matrix that transforms from face-local to world coordinates."""
    yaw_deg, pitch_deg = CUBE_FACES[face]
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    
    # Rotation around Y (yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    
    # Rotation around X (pitch)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ])
    
    return R_yaw @ R_pitch


def project_to_face(world_dir: np.ndarray, face: str, fov_deg: float) -> tuple[float, float] | None:
    """
    Project a world direction onto a cube face.
    
    Returns (u, v) in normalized coordinates [-1, 1] or None if outside face.
    """
    R = get_face_rotation_matrix(face)
    # Transform world direction to face-local coordinates
    local_dir = R.T @ world_dir
    
    # In face-local coords, Z is the viewing direction
    if local_dir[2] <= 0:
        return None  # Behind the face
    
    # Perspective projection
    u = local_dir[0] / local_dir[2]
    v = local_dir[1] / local_dir[2]
    
    # Check if within FOV
    half_fov = np.tan(np.deg2rad(fov_deg / 2))
    if abs(u) > half_fov or abs(v) > half_fov:
        return None
    
    # Normalize to [-1, 1]
    u_norm = u / half_fov
    v_norm = v / half_fov
    
    return (u_norm, v_norm)


def compute_overlap_directions(
    face_a: str, 
    face_b: str, 
    fov_deg: float = 110.0,
    n_samples: int = 50
) -> list[np.ndarray]:
    """
    Compute world directions that fall in the overlap region of two faces.
    
    Returns list of unit vectors in world coordinates.
    """
    # Get face centers
    yaw_a, pitch_a = CUBE_FACES[face_a]
    yaw_b, pitch_b = CUBE_FACES[face_b]
    
    # Sample directions around the midpoint between faces
    mid_yaw = (yaw_a + yaw_b) / 2
    mid_pitch = (pitch_a + pitch_b) / 2
    
    # Handle wraparound for back face
    if abs(yaw_a - yaw_b) > 180:
        if yaw_a < yaw_b:
            mid_yaw = (yaw_a + 360 + yaw_b) / 2
        else:
            mid_yaw = (yaw_a + yaw_b - 360) / 2
        if mid_yaw > 180:
            mid_yaw -= 360
        elif mid_yaw < -180:
            mid_yaw += 360
    
    overlap_directions = []
    half_fov = fov_deg / 2
    
    # Sample a grid of directions and keep those visible in both faces
    for d_yaw in np.linspace(-half_fov, half_fov, n_samples):
        for d_pitch in np.linspace(-half_fov, half_fov, n_samples):
            # Try different sampling strategies based on face pair
            if pitch_a == pitch_b:  # Horizontal neighbors (front/back/left/right)
                test_yaw = mid_yaw + d_yaw * 0.3  # Focus on overlap region
                test_pitch = d_pitch * 0.8
            elif yaw_a == yaw_b:  # Vertical neighbors (with top/bottom)
                test_yaw = yaw_a + d_yaw * 0.8
                test_pitch = mid_pitch + d_pitch * 0.3
            else:  # Diagonal (left/right with top/bottom)
                test_yaw = mid_yaw + d_yaw * 0.5
                test_pitch = mid_pitch + d_pitch * 0.5
            
            world_dir = spherical_to_cartesian(test_yaw, test_pitch)
            
            # Check if visible in both faces
            proj_a = project_to_face(world_dir, face_a, fov_deg)
            proj_b = project_to_face(world_dir, face_b, fov_deg)
            
            if proj_a is not None and proj_b is not None:
                overlap_directions.append(world_dir)
    
    return overlap_directions


def extract_gaussian_depths_at_directions(
    positions: np.ndarray,
    directions: list[np.ndarray],
    angle_threshold_deg: float = 5.0
) -> list[tuple[int, float]]:
    """
    For each direction, find the closest Gaussian and return its depth.
    
    Args:
        positions: (N, 3) array of Gaussian positions
        directions: List of unit vectors to query
        angle_threshold_deg: Maximum angle to consider a match
    
    Returns:
        List of (gaussian_index, depth) tuples, one per direction.
        Returns (-1, 0) for directions with no nearby Gaussian.
    """
    if len(positions) == 0:
        return [(-1, 0.0)] * len(directions)
    
    # Normalize positions to get directions
    depths = np.linalg.norm(positions, axis=1)
    valid_mask = depths > 1e-6
    
    pos_dirs = np.zeros_like(positions)
    pos_dirs[valid_mask] = positions[valid_mask] / depths[valid_mask, np.newaxis]
    
    cos_threshold = np.cos(np.deg2rad(angle_threshold_deg))
    
    results = []
    for query_dir in directions:
        # Compute dot products with all Gaussians
        dots = pos_dirs @ query_dir
        
        # Find best match above threshold
        best_idx = np.argmax(dots)
        if dots[best_idx] >= cos_threshold and valid_mask[best_idx]:
            results.append((best_idx, depths[best_idx]))
        else:
            results.append((-1, 0.0))
    
    return results


def solve_scale_factors(
    observations: list[tuple[str, str, float, float]],
    reference_face: str = "front",
    use_robust: bool = True
) -> dict[str, float]:
    """
    Solve for per-face scale factors minimizing depth disagreement.
    
    Args:
        observations: List of (face_a, face_b, depth_a, depth_b) tuples
        reference_face: Face to fix at scale=1.0
        use_robust: Use iteratively reweighted least squares for robustness
    
    Returns:
        Dict mapping face_name -> scale_factor
    """
    faces = list(CUBE_FACES.keys())
    face_to_idx = {f: i for i, f in enumerate(faces)}
    n_faces = len(faces)
    ref_idx = face_to_idx[reference_face]
    
    if len(observations) < n_faces - 1:
        LOGGER.warning("Too few observations for robust scale estimation")
        return {f: 1.0 for f in faces}
    
    # Build sparse matrix for: s_a * z_a - s_b * z_b = 0
    # With s_ref = 1, we solve for the other 5 scales
    
    n_obs = len(observations)
    rows = []
    cols = []
    data = []
    b = np.zeros(n_obs)
    
    other_faces = [f for f in faces if f != reference_face]
    other_to_col = {f: i for i, f in enumerate(other_faces)}
    
    for i, (face_a, face_b, z_a, z_b) in enumerate(observations):
        idx_a = face_to_idx[face_a]
        idx_b = face_to_idx[face_b]
        
        if idx_a == ref_idx:
            # z_a - s_b * z_b = 0  =>  -z_b * s_b = -z_a
            col_b = other_to_col[face_b]
            rows.append(i)
            cols.append(col_b)
            data.append(-z_b)
            b[i] = -z_a
        elif idx_b == ref_idx:
            # s_a * z_a - z_b = 0  =>  z_a * s_a = z_b
            col_a = other_to_col[face_a]
            rows.append(i)
            cols.append(col_a)
            data.append(z_a)
            b[i] = z_b
        else:
            # s_a * z_a - s_b * z_b = 0
            col_a = other_to_col[face_a]
            col_b = other_to_col[face_b]
            rows.append(i)
            cols.append(col_a)
            data.append(z_a)
            rows.append(i)
            cols.append(col_b)
            data.append(-z_b)
            b[i] = 0.0
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_faces - 1))
    
    # Solve with optional iterative reweighting for robustness
    weights = np.ones(n_obs)
    
    for iteration in range(5 if use_robust else 1):
        # Weighted least squares
        W = sparse.diags(np.sqrt(weights))
        Aw = W @ A
        bw = W @ b
        
        result = lsqr(Aw, bw)
        scales_reduced = result[0]
        
        if use_robust and iteration < 4:
            # Compute residuals and update weights (Huber-like)
            residuals = A @ scales_reduced - b
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = 1.4826 * mad  # Robust scale estimate
            if sigma > 1e-6:
                weights = 1.0 / (1.0 + (residuals / (2 * sigma)) ** 2)
            else:
                break
    
    # Reconstruct full scale vector
    scales = {}
    scales[reference_face] = 1.0
    for face, col in other_to_col.items():
        scales[face] = float(scales_reduced[col])
    
    return scales


def harmonize_gaussians(
    gaussians_per_face: dict[str, np.ndarray],
    fov_deg: float = 110.0,
    n_samples: int = 50,
    reference_face: str = "front",
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> tuple[dict[str, np.ndarray], HarmonizationResult]:
    """
    Harmonize Gaussian positions across cube faces.
    
    Args:
        gaussians_per_face: Dict mapping face_name -> (N, 3) position array
        fov_deg: Field of view used for cube face extraction
        n_samples: Number of samples per dimension for overlap matching
        reference_face: Face to use as scale reference
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        Tuple of (corrected_positions_per_face, harmonization_result)
    """
    LOGGER.info("Computing overlap correspondences...")
    
    # Collect all depth observations from overlaps
    observations = []
    correspondences_per_pair = {}
    
    for face_a, face_b in FACE_ADJACENCIES:
        if face_a not in gaussians_per_face or face_b not in gaussians_per_face:
            continue
        
        # Get overlap directions
        overlap_dirs = compute_overlap_directions(face_a, face_b, fov_deg, n_samples)
        
        if len(overlap_dirs) == 0:
            LOGGER.warning(f"No overlap found between {face_a} and {face_b}")
            continue
        
        # Extract depths at these directions from each face
        depths_a = extract_gaussian_depths_at_directions(
            gaussians_per_face[face_a], overlap_dirs
        )
        depths_b = extract_gaussian_depths_at_directions(
            gaussians_per_face[face_b], overlap_dirs
        )
        
        # Pair up valid depth observations
        pair_count = 0
        for (idx_a, z_a), (idx_b, z_b) in zip(depths_a, depths_b):
            if idx_a >= 0 and idx_b >= 0:
                if min_depth < z_a < max_depth and min_depth < z_b < max_depth:
                    observations.append((face_a, face_b, z_a, z_b))
                    pair_count += 1
        
        correspondences_per_pair[(face_a, face_b)] = pair_count
        LOGGER.info(f"  {face_a}-{face_b}: {pair_count} correspondences from {len(overlap_dirs)} samples")
    
    if len(observations) < 10:
        LOGGER.warning(f"Only {len(observations)} observations - harmonization may be unreliable")
    
    # Compute residual before harmonization
    residual_before = 0.0
    for face_a, face_b, z_a, z_b in observations:
        residual_before += (z_a - z_b) ** 2
    residual_before = np.sqrt(residual_before / max(len(observations), 1))
    
    # Solve for scale factors
    LOGGER.info("Solving for scale factors...")
    scale_factors = solve_scale_factors(observations, reference_face)
    
    LOGGER.info("Scale factors:")
    for face, scale in sorted(scale_factors.items()):
        LOGGER.info(f"  {face}: {scale:.4f}")
    
    # Apply corrections
    corrected = {}
    for face, positions in gaussians_per_face.items():
        scale = scale_factors.get(face, 1.0)
        # Scale positions radially from origin
        corrected[face] = positions * scale
    
    # Compute residual after harmonization
    residual_after = 0.0
    for face_a, face_b, z_a, z_b in observations:
        s_a = scale_factors.get(face_a, 1.0)
        s_b = scale_factors.get(face_b, 1.0)
        residual_after += (s_a * z_a - s_b * z_b) ** 2
    residual_after = np.sqrt(residual_after / max(len(observations), 1))
    
    LOGGER.info(f"RMS residual: {residual_before:.4f} -> {residual_after:.4f}")
    
    result = HarmonizationResult(
        scale_factors=scale_factors,
        residual_before=residual_before,
        residual_after=residual_after,
        n_correspondences=len(observations),
        correspondences_per_pair=correspondences_per_pair,
    )
    
    return corrected, result

