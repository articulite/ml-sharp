"""DAP (Depth Any Panoramas) integration for metric depth guidance.

This module provides:
1. DAP model loading and caching
2. ERP (equirectangular) depth estimation
3. ERP depth to perspective cube face projection

The metric depth from DAP can guide Sharp's Gaussian prediction for
globally consistent depth across all cube faces.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add DAP to path
REPO_ROOT = Path(__file__).parent.parent.parent
DAP_ROOT = REPO_ROOT / "DAP"
sys.path.insert(0, str(DAP_ROOT))

LOGGER = logging.getLogger(__name__)

# Global DAP model cache
_dap_model: Any = None
_dap_device: str | None = None

# Default checkpoint path
DAP_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "dap" / "model.pth"

# DAP outputs depth scaled by max_depth (1.0 = 100m by default)
DAP_MAX_DEPTH_METERS = 100.0


def is_dap_available() -> bool:
    """Check if DAP model weights are available."""
    return DAP_CHECKPOINT_PATH.exists()


def get_dap_status() -> dict:
    """Get DAP availability and status."""
    return {
        "available": is_dap_available(),
        "checkpoint_path": str(DAP_CHECKPOINT_PATH),
        "loaded": _dap_model is not None,
        "device": _dap_device,
    }


def get_or_load_dap_model(device: str = "cpu"):
    """Load and cache the DAP model.
    
    Args:
        device: Device to load model on ('cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of (model, device)
    """
    global _dap_model, _dap_device
    
    if _dap_model is not None:
        LOGGER.info(f"Using cached DAP model on {_dap_device}")
        return _dap_model, _dap_device
    
    if not is_dap_available():
        raise RuntimeError(
            f"DAP model not found at {DAP_CHECKPOINT_PATH}\n"
            "Download from: https://huggingface.co/Insta360-Research/DAP-weights\n"
            "Place model.pth in checkpoints/dap/"
        )
    
    import torch
    from networks.models import make
    
    LOGGER.info(f"Loading DAP model from {DAP_CHECKPOINT_PATH}...")
    
    # Create model
    model = make({
        'name': 'dap',
        'args': {
            'midas_model_type': 'vitl',
            'fine_tune_type': 'hypersim',
            'min_depth': 0.01,
            'max_depth': 1.0,  # Output is 0-1, scaled to meters later
            'train_decoder': True,
        }
    })
    
    # Load weights
    state_dict = torch.load(DAP_CHECKPOINT_PATH, map_location=device)
    
    # Handle DataParallel weights
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    _dap_model = model
    _dap_device = device
    
    LOGGER.info(f"DAP model loaded successfully on {device}")
    return _dap_model, _dap_device


def estimate_erp_depth(
    erp_image: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Estimate metric depth for an equirectangular panorama image.
    
    IMPORTANT: This returns RAW METRIC DEPTH in meters as float32.
    The full 0-100m range is preserved - NOT compressed or normalized for visualization.
    This raw depth is used directly to guide Sharp's Gaussian prediction.
    
    Args:
        erp_image: RGB image array, shape (H, W, 3), uint8 or float
        device: Device for inference
    
    Returns:
        Depth map in METERS, shape (H, W), float32.
        Typical range: 0.1m (10cm) to 100m depending on scene.
        Indoor scenes: ~1-10m
        Outdoor scenes: up to 100m
    """
    import torch
    
    model, device = get_or_load_dap_model(device)
    
    # Ensure correct format
    if erp_image.dtype != np.uint8:
        erp_image = (erp_image * 255).astype(np.uint8)
    
    # DAP expects BGR input for infer_image
    import cv2
    erp_bgr = cv2.cvtColor(erp_image, cv2.COLOR_RGB2BGR)
    
    LOGGER.info(f"Running DAP inference on {erp_image.shape[1]}x{erp_image.shape[0]} image...")
    
    with torch.no_grad():
        # DAP's infer_image handles preprocessing internally
        # Returns normalized depth in range [0, 1] where 1.0 = max_depth (100m)
        depth_normalized = model.infer_image(erp_bgr, input_size=518)
    
    # CRITICAL: Scale to actual meters using full range (0-100m)
    # This is RAW METRIC DEPTH - not compressed for visualization!
    depth_meters = depth_normalized * DAP_MAX_DEPTH_METERS
    
    LOGGER.info(
        f"DAP metric depth (raw float32): "
        f"min={depth_meters.min():.3f}m, max={depth_meters.max():.3f}m, "
        f"mean={depth_meters.mean():.3f}m, dtype={depth_meters.dtype}"
    )
    
    # Flip ERP depth vertically to match the coordinate convention expected by erp_depth_to_perspective
    # DAP outputs Y=0 at top, but our spherical projection math expects Y=0 at bottom
    # This single flip at the source correctly propagates to all 6 cube faces
    depth_meters = np.flipud(depth_meters)
    
    return depth_meters.astype(np.float32)


def erp_depth_to_perspective(
    erp_depth: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    output_size: int,
) -> np.ndarray:
    """Project ERP depth to a perspective cube face view.
    
    This performs the geometric conversion from spherical (ERP) depth
    to perspective (z-buffer) depth for a given viewing direction.
    
    Args:
        erp_depth: Equirectangular depth map (H, W), in meters
        fov_deg: Field of view in degrees
        yaw_deg: Yaw angle (horizontal rotation) in degrees
        pitch_deg: Pitch angle (vertical rotation) in degrees
        output_size: Output face size in pixels
    
    Returns:
        Perspective depth map (output_size, output_size), in meters
    """
    import cv2
    
    h_erp, w_erp = erp_depth.shape[:2]
    size = output_size
    
    # Create pixel grid for output face
    # Pixel centers at (i+0.5, j+0.5)
    jj, ii = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32)
    )
    
    # Convert pixel coords to normalized device coords [-1, 1]
    # Center of image is (0, 0)
    half_fov = np.tan(np.deg2rad(fov_deg / 2))
    u = (2.0 * (jj + 0.5) / size - 1.0) * half_fov
    v = (2.0 * (ii + 0.5) / size - 1.0) * half_fov
    
    # Ray direction in camera space (looking down +Z)
    # x = right, y = down, z = forward
    dx = u
    dy = v
    dz = np.ones_like(u)
    
    # Normalize ray directions
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx, dy, dz = dx/norm, dy/norm, dz/norm
    
    # Rotate rays to world space based on yaw and pitch
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    
    # Rotation matrices
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    
    # Apply pitch (rotation around X axis)
    dy_pitched = dy * cp - dz * sp
    dz_pitched = dy * sp + dz * cp
    dy, dz = dy_pitched, dz_pitched
    
    # Apply yaw (rotation around Y axis)
    dx_yawed = dx * cy + dz * sy
    dz_yawed = -dx * sy + dz * cy
    dx, dz = dx_yawed, dz_yawed
    
    # Convert world ray directions to ERP coordinates
    # ERP convention: theta (longitude) from -pi to pi, phi (latitude) from -pi/2 to pi/2
    # X = right, Y = up, Z = forward in world space
    
    # For ERP sampling, we need to match the convention used in the ERP image
    # Assuming standard ERP: theta=0 is center (forward), phi=0 is horizon
    theta = np.arctan2(dx, dz)  # Horizontal angle
    phi = np.arcsin(np.clip(dy, -1, 1))  # Vertical angle (negative because Y is down in image)
    
    # Convert spherical to ERP pixel coordinates
    # theta: [-pi, pi] -> [0, w_erp]
    # phi: [-pi/2, pi/2] -> [h_erp, 0] (flip because image Y is down)
    erp_x = (theta + np.pi) / (2 * np.pi) * w_erp
    erp_y = (0.5 - phi / np.pi) * h_erp
    
    # Handle wrap-around for x
    erp_x = erp_x % w_erp
    
    # Clamp y to valid range
    erp_y = np.clip(erp_y, 0, h_erp - 1)
    
    # Sample depth from ERP using bilinear interpolation
    map_x = erp_x.astype(np.float32)
    map_y = erp_y.astype(np.float32)
    
    # Use cv2.remap with border wrap for horizontal continuity
    erp_depth_radial = cv2.remap(
        erp_depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    
    # Convert radial depth to z-buffer depth
    # Radial depth = distance from camera center
    # Z-buffer depth = depth along the view ray = radial * cos(angle_from_center)
    # The angle from center is encoded in the ray direction
    # cos(angle) = dot(ray, forward) = dz (after rotation, this is the z component)
    
    # Actually, for a perspective projection, the depth we want is:
    # z_perspective = radial_depth * cos(angle_from_optical_axis)
    # Since our rays are unit vectors, and the optical axis is (0, 0, 1) in camera space,
    # the cos of angle is just the z component before rotation: 1/norm
    # But we need to be careful about the convention...
    
    # For Sharp, depth should be the distance along the view ray (z-buffer style)
    # The relationship is: z_buffer = radial * cos(half_angle_from_center)
    # where half_angle = atan(sqrt(u^2 + v^2))
    
    # Simpler approach: z_buffer = radial / norm, where norm = sqrt(u^2 + v^2 + 1)
    # This gives us the depth along the principal ray direction
    perspective_depth = erp_depth_radial / norm
    
    return perspective_depth.astype(np.float32)


def extract_depth_cubefaces(
    erp_depth: np.ndarray,
    fov_deg: float,
    output_size: int,
    faces: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Extract perspective depth maps for all cube faces.
    
    Args:
        erp_depth: Equirectangular depth map (H, W), in meters
        fov_deg: Field of view in degrees
        output_size: Output face size in pixels
        faces: Dict mapping face name to (yaw, pitch) angles
    
    Returns:
        Dict mapping face name to perspective depth array
    """
    depth_faces = {}
    
    for face_name, (yaw, pitch) in faces.items():
        LOGGER.info(f"Extracting depth for {face_name} face (yaw={yaw}°, pitch={pitch}°)")
        depth_face = erp_depth_to_perspective(
            erp_depth, fov_deg, yaw, pitch, output_size
        )
        depth_faces[face_name] = depth_face
    
    return depth_faces


def prepare_depth_for_sharp(
    depth_face: np.ndarray,
    internal_shape: tuple[int, int],
    device: str,
) -> "torch.Tensor":
    """Prepare a depth face for Sharp's predictor.
    
    Resizes and formats the depth map for Sharp's expected input format.
    
    Args:
        depth_face: Perspective depth map (H, W) in meters
        internal_shape: Sharp's internal processing size (H, W)
        device: Device for the output tensor
    
    Returns:
        Depth tensor ready for Sharp's forward() method
    """
    import torch
    import torch.nn.functional as F
    
    # Convert to tensor
    depth_tensor = torch.from_numpy(depth_face).float().to(device)
    
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    
    # Resize to Sharp's internal resolution
    depth_resized = F.interpolate(
        depth_tensor,
        size=internal_shape,
        mode="bilinear",
        align_corners=True,
    )
    
    # Sharp expects depth with shape (B, 1, H, W)
    return depth_resized


# Cleanup function for graceful shutdown
def unload_dap_model():
    """Unload the DAP model to free memory."""
    global _dap_model, _dap_device
    
    if _dap_model is not None:
        import torch
        del _dap_model
        _dap_model = None
        _dap_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("DAP model unloaded")

