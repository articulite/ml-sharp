"""Cubemap-coherent prediction with depth conditioning.

This module implements a two-pass prediction approach that produces
geometrically consistent depth across cube faces without retraining.

Pass 1: Predict all faces independently
Pass 2: Re-predict with pseudo-GT depths from neighbors in overlap regions

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import click
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import save_ply, unproject_gaussians, Gaussians3D

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Cube face adjacencies: face -> list of (neighbor, edge, neighbor_edge)
# edge: 'left', 'right', 'top', 'bottom'
CUBE_ADJACENCIES = {
    'front': [('left', 'left', 'right'), ('right', 'right', 'left'), 
              ('top', 'top', 'bottom'), ('bottom', 'bottom', 'top')],
    'back': [('right', 'left', 'right'), ('left', 'right', 'left'),
             ('top', 'top', 'bottom'), ('bottom', 'bottom', 'top')],
    'left': [('back', 'left', 'right'), ('front', 'right', 'left'),
             ('top', 'top', 'bottom'), ('bottom', 'bottom', 'top')],
    'right': [('front', 'left', 'right'), ('back', 'right', 'left'),
              ('top', 'top', 'bottom'), ('bottom', 'bottom', 'top')],
    'top': [('front', 'bottom', 'top'), ('back', 'bottom', 'top'),
            ('left', 'bottom', 'top'), ('right', 'bottom', 'top')],
    'bottom': [('front', 'top', 'bottom'), ('back', 'top', 'bottom'),
               ('left', 'top', 'bottom'), ('right', 'top', 'bottom')],
}

# World-space rotations for each face (yaw, pitch in degrees)
CUBE_ROTATIONS = {
    'front': (0, 0),
    'back': (180, 0),
    'left': (-90, 0),
    'right': (90, 0),
    'top': (0, -90),
    'bottom': (0, 90),
}


def compute_overlap_width(fov: float, image_size: int) -> int:
    """Compute overlap width in pixels for given FOV.
    
    Args:
        fov: Field of view in degrees
        image_size: Image dimension in pixels
        
    Returns:
        Number of pixels that overlap with adjacent face
    """
    overlap_angle = fov - 90  # degrees beyond cube face
    if overlap_angle <= 0:
        return 0
    # overlap pixels = image_size * (overlap_angle / fov)
    return int(image_size * (overlap_angle / fov) / 2)


def get_edge_region(tensor: torch.Tensor, edge: str, width: int) -> torch.Tensor:
    """Extract edge region from a tensor.
    
    Args:
        tensor: Input tensor (B, C, H, W)
        edge: 'left', 'right', 'top', 'bottom'
        width: Width of edge region in pixels
        
    Returns:
        Edge region tensor with same number of dimensions
    """
    # Ensure 4D
    orig_dim = tensor.dim()
    if orig_dim == 3:
        tensor = tensor.unsqueeze(0)
    
    if edge == 'left':
        result = tensor[:, :, :, :width]
    elif edge == 'right':
        result = tensor[:, :, :, -width:]
    elif edge == 'top':
        result = tensor[:, :, :width, :]
    elif edge == 'bottom':
        result = tensor[:, :, -width:, :]
    else:
        raise ValueError(f"Unknown edge: {edge}")
    
    return result


def set_edge_region(tensor: torch.Tensor, edge: str, width: int, value: torch.Tensor) -> None:
    """Set edge region of a tensor.
    
    Args:
        tensor: Input tensor to modify in-place (B, C, H, W)
        edge: 'left', 'right', 'top', 'bottom'
        width: Width of edge region in pixels
        value: Values to set (must match edge region shape)
    """
    # Ensure value has same dimensions as tensor
    while value.dim() < tensor.dim():
        value = value.unsqueeze(0)
    
    if edge == 'left':
        tensor[:, :, :, :width] = value
    elif edge == 'right':
        tensor[:, :, :, -width:] = value
    elif edge == 'top':
        tensor[:, :, :width, :] = value
    elif edge == 'bottom':
        tensor[:, :, -width:, :] = value


def transform_depth_to_neighbor(
    depth: torch.Tensor,
    src_face: str,
    dst_face: str,
    src_edge: str,
    dst_edge: str,
) -> torch.Tensor:
    """Transform depth from source face's edge to destination face's edge.
    
    This accounts for the geometric relationship between adjacent cube faces.
    For simplicity, we use a scale factor based on viewing angle differences.
    
    Args:
        depth: Depth values at source edge (B, C, H, W)
        src_face: Source face name
        dst_face: Destination face name
        src_edge: Edge on source face
        dst_edge: Edge on destination face
        
    Returns:
        Transformed depth for destination face's edge (B, C, H, W)
    """
    # Ensure 4D tensor (B, C, H, W)
    while depth.dim() < 4:
        depth = depth.unsqueeze(0)
    
    # For adjacent cube faces at 90 degrees, depths in overlap regions
    # correspond to the same world points viewed from different angles.
    # The relationship depends on the overlap angle from center.
    
    # For now, use direct depth (overlap regions see similar depths)
    # A more sophisticated version would account for viewing angle
    
    # Handle edge orientation differences
    transformed = depth.clone()
    
    # Flip if edges have opposite orientations
    if (src_edge in ['left', 'right']) != (dst_edge in ['left', 'right']):
        # Horizontal <-> Vertical edge transition (top/bottom faces)
        # This is complex - for now, skip these
        pass
    elif src_edge == 'left' and dst_edge == 'right':
        transformed = torch.flip(transformed, dims=[3])  # flip W dimension
    elif src_edge == 'right' and dst_edge == 'left':
        transformed = torch.flip(transformed, dims=[3])  # flip W dimension
    elif src_edge == 'top' and dst_edge == 'bottom':
        transformed = torch.flip(transformed, dims=[2])  # flip H dimension
    elif src_edge == 'bottom' and dst_edge == 'top':
        transformed = torch.flip(transformed, dims=[2])  # flip H dimension
    
    return transformed


class CubemapPredictor:
    """Two-pass cubemap-coherent predictor."""
    
    def __init__(self, predictor, device: torch.device, fov: float = 110.0):
        """Initialize cubemap predictor.
        
        Args:
            predictor: The RGBGaussianPredictor model
            device: Torch device
            fov: Field of view in degrees
        """
        self.predictor = predictor
        self.device = device
        self.fov = fov
        self.internal_shape = (1536, 1536)
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Preprocess image for inference.
        
        Returns:
            image_resized: Resized image tensor
            disparity_factor: Disparity factor
            height: Original height
            width: Original width
        """
        image_pt = torch.from_numpy(image.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        
        size = min(width, height)
        f_px = size / (2 * np.tan(np.deg2rad(self.fov) / 2))
        disparity_factor = torch.tensor([f_px / width]).float().to(self.device)
        
        image_resized = F.interpolate(
            image_pt[None],
            size=self.internal_shape,
            mode="bilinear",
            align_corners=True,
        )
        
        return image_resized, disparity_factor, height, width, f_px
    
    @torch.no_grad()
    def predict_pass1(self, images: Dict[str, np.ndarray]) -> Dict[str, dict]:
        """First pass: Independent prediction for all faces.
        
        Runs full prediction including Gaussian generation and caches all results.
        
        Args:
            images: Dict of face_name -> image array
            
        Returns:
            Dict of face_name -> {monodepth, gaussians_world, ...}
        """
        results = {}
        
        for face, image in images.items():
            LOGGER.info(f"Pass 1: Predicting {face}")
            
            image_resized, disparity_factor, height, width, f_px = self.preprocess_image(image)
            
            # Get monodepth output for scale factor computation
            monodepth_output = self.predictor.monodepth_model(image_resized)
            monodepth_disparity = monodepth_output.disparity
            
            disparity_factor_expanded = disparity_factor[:, None, None, None]
            monodepth = disparity_factor_expanded / monodepth_disparity.clamp(min=1e-4, max=1e4)
            
            # Model outputs 2 depth layers for sorting - use the first (primary) layer
            if monodepth.shape[1] > 1:
                monodepth_single = monodepth[:, 0:1, :, :]
            else:
                monodepth_single = monodepth
            
            # Also run full Gaussian prediction (cache for pass 2)
            gaussians_ndc = self.predictor(image_resized, disparity_factor, depth=None)
            
            # Convert to world space
            intrinsics = torch.tensor([
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).float().to(self.device)
            
            intrinsics_resized = intrinsics.clone()
            intrinsics_resized[0] *= self.internal_shape[0] / width
            intrinsics_resized[1] *= self.internal_shape[1] / height
            
            gaussians_world = unproject_gaussians(
                gaussians_ndc, 
                torch.eye(4).to(self.device), 
                intrinsics_resized, 
                self.internal_shape
            )
            
            results[face] = {
                'image_resized': image_resized,
                'disparity_factor': disparity_factor,
                'monodepth': monodepth_single,  # Single channel for scale factor computation
                'gaussians_world': gaussians_world,  # Cached Gaussians
                'height': height,
                'width': width,
                'f_px': f_px,
            }
        
        return results
    
    def solve_scale_factors(self, pass1_results: Dict[str, dict]) -> Dict[str, float]:
        """Solve for optimal scale factors using EDGE correspondence.
        
        Instead of using mean depth (which doesn't fix edge alignment),
        we sample depths at the actual boundaries where faces meet and
        optimize to minimize depth differences at those edges.
        
        Args:
            pass1_results: Results from pass 1 (contains gaussians_world)
            
        Returns:
            Dict of face_name -> scale factor
        """
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        
        faces = list(pass1_results.keys())
        face_to_idx = {f: i for i, f in enumerate(faces)}
        n_faces = len(faces)
        
        H, W = self.internal_shape
        overlap_width = compute_overlap_width(self.fov, W)
        
        LOGGER.info(f"Computing edge depths for scale factor optimization...")
        LOGGER.info(f"  FOV={self.fov}°, overlap_width={overlap_width}px")
        
        # Build linear system from edge correspondences
        rows, cols, data = [], [], []
        rhs = []
        row_idx = 0
        
        for face in faces:
            if face not in CUBE_ADJACENCIES:
                continue
                
            for neighbor, my_edge, neighbor_edge in CUBE_ADJACENCIES[face]:
                if neighbor not in pass1_results:
                    continue
                if face in ['top', 'bottom'] or neighbor in ['top', 'bottom']:
                    continue
                # Only process each pair once
                if face > neighbor:
                    continue
                
                # Get monodepth at edges (these are in the same local coordinate frame)
                my_depth = pass1_results[face]['monodepth']
                neighbor_depth = pass1_results[neighbor]['monodepth']
                
                # Sample depth at the edge boundary
                my_edge_depth = get_edge_region(my_depth, my_edge, max(overlap_width, 20))
                neighbor_edge_depth = get_edge_region(neighbor_depth, neighbor_edge, max(overlap_width, 20))
                
                # Transform neighbor edge to align with my edge
                transformed = transform_depth_to_neighbor(
                    neighbor_edge_depth, neighbor, face, neighbor_edge, my_edge
                )
                
                # Use median depth at edge (more robust than mean)
                d_i = my_edge_depth.median().item()
                d_j = transformed.median().item()
                
                LOGGER.info(f"  {face}({my_edge}) <-> {neighbor}({neighbor_edge}): d_i={d_i:.3f}, d_j={d_j:.3f}")
                
                if d_i > 0.01 and d_j > 0.01:
                    # Equation: s_i * d_i = s_j * d_j (depths should match at edge)
                    i_idx = face_to_idx[face]
                    j_idx = face_to_idx[neighbor]
                    
                    rows.extend([row_idx, row_idx])
                    cols.extend([i_idx, j_idx])
                    data.extend([d_i, -d_j])
                    rhs.append(0.0)
                    row_idx += 1
        
        if row_idx == 0:
            LOGGER.warning("No edge correspondences found!")
            return {face: 1.0 for face in faces}
        
        # Add regularization: scales should be close to 1
        for i in range(n_faces):
            rows.append(row_idx)
            cols.append(i)
            data.append(0.1)  # regularization weight
            rhs.append(0.1)   # target scale = 1
            row_idx += 1
        
        # Solve least squares
        A = sparse.csr_matrix((data, (rows, cols)), shape=(row_idx, n_faces))
        b = np.array(rhs)
        
        result = lsqr(A, b)
        scales = result[0]
        
        # Normalize so mean scale is 1
        scales = scales / np.mean(scales)
        
        scale_dict = {faces[i]: float(scales[i]) for i in range(n_faces)}
        LOGGER.info(f"Solved scale factors: {scale_dict}")
        
        return scale_dict
    
    def compute_pseudo_gt(self, pass1_results: Dict[str, dict], scale_factors: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute pseudo ground-truth depths from harmonized neighbor overlaps.
        
        For each face, create a depth map where overlap regions contain
        the SCALED depths from neighboring faces.
        
        Args:
            pass1_results: Results from pass 1
            scale_factors: Pre-computed scale factors for each face
            
        Returns:
            Dict of face_name -> pseudo_gt_depth tensor
        """
        pseudo_gts = {}
        
        # Get image size from internal shape
        H, W = self.internal_shape
        overlap_width = compute_overlap_width(self.fov, W)
        
        if overlap_width < 5:
            LOGGER.warning(f"Overlap width is only {overlap_width}px. FOV may be too low for effective conditioning.")
            return {}
        
        LOGGER.info(f"FOV={self.fov}°, overlap={self.fov - 90}°, overlap_width={overlap_width}px")
        
        for face in pass1_results:
            if face not in CUBE_ADJACENCIES:
                continue
                
            # Start with zeros (will be masked)
            pseudo_gt = torch.zeros((1, 1, H, W), device=self.device)
            mask = torch.zeros((1, 1, H, W), device=self.device)
            
            my_scale = scale_factors.get(face, 1.0)
            
            for neighbor, my_edge, neighbor_edge in CUBE_ADJACENCIES[face]:
                if neighbor not in pass1_results:
                    continue
                
                # Skip top/bottom for now (complex geometry)
                if face in ['top', 'bottom'] or neighbor in ['top', 'bottom']:
                    continue
                
                neighbor_scale = scale_factors.get(neighbor, 1.0)
                
                # Get neighbor's depth at the shared edge
                neighbor_depth = pass1_results[neighbor]['monodepth']
                neighbor_edge_depth = get_edge_region(neighbor_depth, neighbor_edge, overlap_width)
                
                # Apply scale factor to harmonize: scale neighbor's depth to my coordinate system
                # neighbor_scaled * neighbor_scale ≈ my * my_scale
                # So pseudo_gt for me = neighbor_scaled * (neighbor_scale / my_scale)
                scale_ratio = neighbor_scale / my_scale
                scaled_neighbor_depth = neighbor_edge_depth * scale_ratio
                
                # Transform to my coordinate system
                transformed_depth = transform_depth_to_neighbor(
                    scaled_neighbor_depth, neighbor, face, neighbor_edge, my_edge
                )
                
                # Set in my pseudo-GT
                set_edge_region(pseudo_gt, my_edge, overlap_width, transformed_depth)
                
                # Update mask
                edge_mask = torch.ones_like(transformed_depth)
                set_edge_region(mask, my_edge, overlap_width, edge_mask)
            
            # Only store if we have some overlap data
            if mask.sum() > 0:
                # For regions without overlap, use our own scaled depth
                own_depth = pass1_results[face]['monodepth']
                pseudo_gt = pseudo_gt * mask + own_depth * (1 - mask)
                pseudo_gts[face] = pseudo_gt
        
        return pseudo_gts
    
    def stitch_boundaries(self, pass1_results: Dict[str, dict]) -> Dict[str, Gaussians3D]:
        """Stitch face boundaries by averaging positions at edges.
        
        This directly aligns the boundary Gaussians between adjacent faces
        by finding corresponding points and moving them toward each other.
        
        Args:
            pass1_results: Results from pass 1 (contains gaussians_world)
            
        Returns:
            Dict of face_name -> stitched Gaussians3D
        """
        import numpy as np
        
        # First, get world-space positions for all faces
        face_positions = {}
        face_data = {}
        
        for face, data in pass1_results.items():
            gaussians = data['gaussians_world']
            positions = gaussians.mean_vectors.cpu().numpy().reshape(-1, 3)  # (N, 3)
            face_positions[face] = positions
            face_data[face] = {
                'gaussians': gaussians,
                'height': data['height'],
                'width': data['width'],
                'f_px': data['f_px'],
            }
        
        # Define edge boundaries and apply world-space rotations
        # Edges are defined by angular position in world space
        edge_adjacencies = [
            ('front', 'left', 'x', -1),   # front's negative-x edge meets left's positive-x edge
            ('front', 'right', 'x', +1),  # front's positive-x edge meets right's negative-x edge
            ('left', 'back', 'x', -1),    # left's negative-x edge meets back's positive-x edge
            ('right', 'back', 'x', +1),   # right's positive-x edge meets back's negative-x edge
        ]
        
        # Compute boundary corrections
        corrections = {face: np.zeros_like(face_positions[face]) for face in face_positions}  # (N, 3)
        correction_weights = {face: np.zeros(face_positions[face].shape[0]) for face in face_positions}  # (N,)
        
        edge_pct = 0.15  # Consider 15% of points nearest to edge
        
        for face1, face2, axis, direction in edge_adjacencies:
            if face1 not in face_positions or face2 not in face_positions:
                continue
            
            pos1 = face_positions[face1].reshape(-1, 3)  # Ensure 2D (N, 3)
            pos2 = face_positions[face2].reshape(-1, 3)
            
            # Find boundary points on face1 (edge in direction)
            axis_idx = 0 if axis == 'x' else 2
            if direction > 0:
                threshold1 = np.percentile(pos1[:, axis_idx], 100 - edge_pct * 100)
                mask1 = pos1[:, axis_idx] >= threshold1
            else:
                threshold1 = np.percentile(pos1[:, axis_idx], edge_pct * 100)
                mask1 = pos1[:, axis_idx] <= threshold1
            
            # Find boundary points on face2 (opposite edge)
            if direction > 0:
                threshold2 = np.percentile(pos2[:, axis_idx], edge_pct * 100)
                mask2 = pos2[:, axis_idx] <= threshold2
            else:
                threshold2 = np.percentile(pos2[:, axis_idx], 100 - edge_pct * 100)
                mask2 = pos2[:, axis_idx] >= threshold2
            
            edge1 = pos1[mask1]
            edge2 = pos2[mask2]
            
            if len(edge1) == 0 or len(edge2) == 0:
                continue
            
            # Compute median Y (floor height) at each edge
            median_y1 = np.median(edge1[:, 1])
            median_y2 = np.median(edge2[:, 1])
            
            # Target: average of the two
            target_y = (median_y1 + median_y2) / 2
            
            # Correction for each face's edge points
            correction_y1 = target_y - median_y1
            correction_y2 = target_y - median_y2
            
            LOGGER.info(f"  {face1}<->{face2}: floor1={median_y1:.2f}, floor2={median_y2:.2f}, target={target_y:.2f}")
            
            # Apply corrections with falloff from edge
            indices1 = np.where(mask1)[0]
            indices2 = np.where(mask2)[0]
            
            for idx in indices1:
                corrections[face1][idx, 1] += correction_y1
                correction_weights[face1][idx] += 1
            
            for idx in indices2:
                corrections[face2][idx, 1] += correction_y2
                correction_weights[face2][idx] += 1
        
        # Apply corrections
        results = {}
        for face, data in face_data.items():
            gaussians = data['gaussians']
            positions = face_positions[face]  # Already (N, 3)
            original_shape = gaussians.mean_vectors.shape
            
            # Average corrections where multiple edges affect same point
            weights = correction_weights[face]
            weights = np.where(weights > 0, weights, 1)
            avg_corrections = corrections[face] / weights[:, None]
            
            # Apply correction
            new_positions = positions + avg_corrections
            
            # Create new Gaussians with corrected positions
            new_gaussians = Gaussians3D(
                mean_vectors=torch.from_numpy(new_positions.astype(np.float32)).reshape(original_shape).to(gaussians.mean_vectors.device),
                singular_values=gaussians.singular_values,
                quaternions=gaussians.quaternions,
                colors=gaussians.colors,
                opacities=gaussians.opacities,
            )
            
            results[face] = (new_gaussians, data['f_px'], data['height'], data['width'])
        
        return results
    
    def predict_coherent(self, images: Dict[str, np.ndarray]) -> Dict[str, Gaussians3D]:
        """Run coherent prediction with boundary stitching.
        
        This approach:
        1. Predicts all faces independently (full Gaussian prediction)
        2. Stitches boundaries by aligning floor heights at edges
        
        Args:
            images: Dict of face_name -> image array
            
        Returns:
            Dict of face_name -> (Gaussians3D, f_px, height, width)
        """
        LOGGER.info("="*50)
        LOGGER.info("Predicting all faces (with caching)")
        LOGGER.info("="*50)
        pass1_results = self.predict_pass1(images)
        
        LOGGER.info("="*50)
        LOGGER.info("Stitching boundaries (aligning floor heights at edges)")
        LOGGER.info("="*50)
        gaussians = self.stitch_boundaries(pass1_results)
        
        return gaussians


@click.command()
@click.option("-i", "--input-path", type=click.Path(path_type=Path, exists=True), required=True,
              help="Path to folder containing cube face images (front.png, left.png, etc.)")
@click.option("-o", "--output-path", type=click.Path(path_type=Path), required=True,
              help="Path to save output PLY files")
@click.option("-c", "--checkpoint-path", type=click.Path(path_type=Path), default=None,
              help="Path to model checkpoint (downloads default if not provided)")
@click.option("--fov", type=float, default=110.0, help="Field of view in degrees")
@click.option("--device", type=str, default="default", help="Device: cuda, mps, cpu")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--compare/--no-compare", default=True, 
              help="Also run single-pass prediction for comparison")
def predict_cubemap_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    fov: float,
    device: str,
    verbose: bool,
    compare: bool,
):
    """Predict Gaussians from cube face images with depth coherence."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    
    # Find device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info(f"Using device: {device}")
    
    # Load model
    if checkpoint_path is None:
        LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)
    
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    
    # Find cube face images
    extensions = io.get_supported_image_extensions()
    face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    images = {}
    for face in face_names:
        for ext in extensions:
            candidates = list(input_path.glob(f"*{face}*{ext}")) + \
                        list(input_path.glob(f"*{face.upper()}*{ext}"))
            if candidates:
                LOGGER.info(f"Found {face}: {candidates[0].name}")
                image, _, _ = io.load_rgb(candidates[0])
                images[face] = image
                break
    
    if len(images) == 0:
        LOGGER.error("No cube face images found!")
        return
    
    LOGGER.info(f"Found {len(images)} cube faces: {list(images.keys())}")
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Run two-pass coherent prediction
    cubemap_predictor = CubemapPredictor(predictor, torch.device(device), fov=fov)
    results = cubemap_predictor.predict_coherent(images)
    
    # Save results
    coherent_dir = output_path / "coherent"
    coherent_dir.mkdir(exist_ok=True)
    
    for face, (gaussians, f_px, height, width) in results.items():
        ply_path = coherent_dir / f"input_{face}.ply"
        save_ply(gaussians, f_px, (height, width), ply_path)
        LOGGER.info(f"Saved coherent {face} to {ply_path}")
    
    # Optionally run single-pass for comparison
    if compare:
        LOGGER.info("="*50)
        LOGGER.info("Running single-pass (independent) for comparison")
        LOGGER.info("="*50)
        
        from .predict import predict_image
        
        independent_dir = output_path / "independent"
        independent_dir.mkdir(exist_ok=True)
        
        for face, image in images.items():
            height, width = image.shape[:2]
            size = min(width, height)
            f_px = size / (2 * np.tan(np.deg2rad(fov) / 2))
            
            gaussians = predict_image(predictor, image, f_px, torch.device(device))
            ply_path = independent_dir / f"input_{face}.ply"
            save_ply(gaussians, f_px, (height, width), ply_path)
            LOGGER.info(f"Saved independent {face} to {ply_path}")
    
    LOGGER.info("="*50)
    LOGGER.info("DONE!")
    LOGGER.info(f"Coherent splats: {coherent_dir}")
    if compare:
        LOGGER.info(f"Independent splats: {independent_dir}")
    LOGGER.info("="*50)


if __name__ == "__main__":
    predict_cubemap_cli()

