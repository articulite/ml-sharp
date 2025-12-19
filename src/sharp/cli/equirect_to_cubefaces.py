"""Converts equirectangular panoramas to 6 cube face images with configurable FOV.

This script extracts perspective views from equirectangular (360°) images,
generating labeled cube faces suitable for per-face Gaussian splat generation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
from PIL import Image

from sharp.utils import io as sharp_io
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

# Cube face definitions: name and (yaw, pitch) in degrees
# yaw: rotation around vertical axis (left/right)
# pitch: rotation around horizontal axis (up/down)
CUBE_FACES = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "top": (0, 90),
    "bottom": (0, -90),
}


def equirect_to_perspective(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    output_size: int,
) -> np.ndarray:
    """Convert equirectangular image to perspective projection.
    
    Args:
        equirect: Input equirectangular image (H, W, C).
        fov_deg: Field of view in degrees.
        yaw_deg: Yaw angle in degrees (rotation around vertical axis).
        pitch_deg: Pitch angle in degrees (rotation around horizontal axis).
        output_size: Size of the output square image.
    
    Returns:
        Perspective projected image (output_size, output_size, C).
    """
    h_eq, w_eq = equirect.shape[:2]
    
    # Convert angles to radians
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    
    # Focal length in pixels for the output image
    f = output_size / (2 * np.tan(fov / 2))
    
    # Create pixel grid for output image
    # Center the grid at (0, 0)
    u = np.arange(output_size) - output_size / 2 + 0.5
    v = np.arange(output_size) - output_size / 2 + 0.5
    u, v = np.meshgrid(u, v)
    
    # Convert pixel coordinates to 3D ray directions (camera space)
    # Camera looks along +Z, X is right, Y is down
    x = u / f
    y = v / f
    z = np.ones_like(x)
    
    # Normalize to unit vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm
    
    # Rotation matrix for pitch (around X axis)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    # Rotation matrix for yaw (around Y axis)  
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Apply pitch rotation first (rotate around X)
    y_rot = y * cos_p - z * sin_p
    z_rot = y * sin_p + z * cos_p
    y, z = y_rot, z_rot
    
    # Apply yaw rotation (rotate around Y)
    x_rot = x * cos_y + z * sin_y
    z_rot = -x * sin_y + z * cos_y
    x, z = x_rot, z_rot
    
    # Convert 3D direction to spherical coordinates
    # longitude: atan2(x, z) maps to [-pi, pi]
    # latitude: asin(y) maps to [-pi/2, pi/2] (but y points down, so negate)
    longitude = np.arctan2(x, z)
    latitude = -np.arcsin(np.clip(y, -1, 1))
    
    # Convert spherical coordinates to equirectangular pixel coordinates
    # longitude [-pi, pi] -> x [0, w_eq]
    # latitude [-pi/2, pi/2] -> y [h_eq, 0] (note: top of image is +latitude)
    eq_x = (longitude / np.pi + 1) / 2 * w_eq
    eq_y = (0.5 - latitude / np.pi) * h_eq
    
    # Wrap coordinates
    eq_x = eq_x % w_eq
    eq_y = np.clip(eq_y, 0, h_eq - 1)
    
    # Bilinear interpolation
    x0 = np.floor(eq_x).astype(int)
    y0 = np.floor(eq_y).astype(int)
    x1 = (x0 + 1) % w_eq
    y1 = np.minimum(y0 + 1, h_eq - 1)
    
    wx = eq_x - x0
    wy = eq_y - y0
    
    # Expand weights for broadcasting with color channels
    wx = wx[..., np.newaxis]
    wy = wy[..., np.newaxis]
    
    # Sample the four neighboring pixels
    c00 = equirect[y0, x0]
    c01 = equirect[y0, x1]
    c10 = equirect[y1, x0]
    c11 = equirect[y1, x1]
    
    # Bilinear interpolation
    c0 = c00 * (1 - wx) + c01 * wx
    c1 = c10 * (1 - wx) + c11 * wx
    result = c0 * (1 - wy) + c1 * wy
    
    return result.astype(np.uint8)


def compute_focal_length_px(fov_deg: float, image_size: int) -> float:
    """Compute focal length in pixels from FOV and image size.
    
    Args:
        fov_deg: Field of view in degrees.
        image_size: Image width/height in pixels.
    
    Returns:
        Focal length in pixels.
    """
    fov_rad = np.deg2rad(fov_deg)
    return image_size / (2 * np.tan(fov_rad / 2))


def focal_length_to_35mm_equiv(f_px: float, image_size: int) -> float:
    """Convert focal length in pixels to 35mm equivalent.
    
    Args:
        f_px: Focal length in pixels.
        image_size: Image width/height in pixels.
    
    Returns:
        Focal length in 35mm equivalent mm.
    """
    # For a square image, diagonal = sqrt(2) * size
    diagonal_px = np.sqrt(2) * image_size
    # 35mm film diagonal = sqrt(36^2 + 24^2) = 43.27mm
    diagonal_35mm = np.sqrt(36**2 + 24**2)
    return f_px * diagonal_35mm / diagonal_px


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to equirectangular panorama image.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory for cube face images.",
    required=True,
)
@click.option(
    "--fov",
    type=float,
    default=110.0,
    help="Field of view in degrees for each face (default: 110).",
)
@click.option(
    "--size",
    type=int,
    default=None,
    help="Output image size in pixels (default: auto from input).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["png", "jpg", "tiff"]),
    default="png",
    help="Output image format.",
)
@click.option(
    "--faces",
    type=str,
    default="all",
    help="Comma-separated list of faces to extract (front,back,left,right,top,bottom) or 'all'.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def equirect_to_cubefaces_cli(
    input_path: Path,
    output_path: Path,
    fov: float,
    size: int | None,
    output_format: str,
    faces: str,
    verbose: bool,
):
    """Convert equirectangular panorama to cube face images.
    
    Extracts perspective views from a 360° equirectangular image,
    generating labeled cube faces suitable for per-face Gaussian splat generation.
    
    Example usage:
    
        python -m sharp.cli.equirect_to_cubefaces -i pano.jpg -o ./cubefaces/ --fov 110
    
    The output images are named with the face label and include EXIF metadata
    with the appropriate focal length for downstream processing.
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    
    LOGGER.info("Loading equirectangular image from %s", input_path)
    
    # Load the input image
    img = Image.open(input_path)
    equirect = np.array(img)
    
    # Ensure RGB
    if equirect.ndim == 2:
        equirect = np.stack([equirect] * 3, axis=-1)
    elif equirect.shape[-1] == 4:
        equirect = equirect[..., :3]
    
    h_eq, w_eq = equirect.shape[:2]
    LOGGER.info("Input image size: %d x %d", w_eq, h_eq)
    
    # Verify it looks like an equirectangular image (2:1 aspect ratio)
    aspect_ratio = w_eq / h_eq
    if not (1.8 < aspect_ratio < 2.2):
        LOGGER.warning(
            "Input aspect ratio %.2f is not close to 2:1. "
            "Are you sure this is an equirectangular image?",
            aspect_ratio,
        )
    
    # Determine output size
    if size is None:
        # Default: use height of equirect divided by 2 (gives roughly similar detail)
        size = h_eq // 2
        # Round to nearest multiple of 16 for compatibility
        size = (size // 16) * 16
        size = max(size, 512)
    
    LOGGER.info("Output face size: %d x %d", size, size)
    LOGGER.info("Field of view: %.1f degrees", fov)
    
    # Parse faces to extract
    if faces.lower() == "all":
        faces_to_extract = list(CUBE_FACES.keys())
    else:
        faces_to_extract = [f.strip().lower() for f in faces.split(",")]
        for face_name in faces_to_extract:
            if face_name not in CUBE_FACES:
                raise click.BadParameter(
                    f"Unknown face '{face_name}'. Valid faces: {list(CUBE_FACES.keys())}"
                )
    
    LOGGER.info("Extracting faces: %s", faces_to_extract)
    
    # Compute focal length for metadata
    f_px = compute_focal_length_px(fov, size)
    f_35mm = focal_length_to_35mm_equiv(f_px, size)
    LOGGER.info("Focal length: %.1f px (%.1f mm @ 35mm equiv)", f_px, f_35mm)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract each face
    for face_name in faces_to_extract:
        yaw, pitch = CUBE_FACES[face_name]
        LOGGER.info("Extracting %s face (yaw=%.0f°, pitch=%.0f°)", face_name, yaw, pitch)
        
        face_image = equirect_to_perspective(equirect, fov, yaw, pitch, size)
        
        # Save the image
        output_filename = f"{input_path.stem}_{face_name}.{output_format}"
        output_file = output_path / output_filename
        
        # Use PIL to save with EXIF metadata
        pil_image = Image.fromarray(face_image)
        
        # Create EXIF data with focal length
        from PIL.ExifTags import Base as ExifBase
        exif = pil_image.getexif()
        # Set FocalLengthIn35mmFilm (tag 41989)
        exif[41989] = int(round(f_35mm))
        
        if output_format == "jpg":
            pil_image.save(output_file, "JPEG", quality=95, exif=exif)
        elif output_format == "png":
            # PNG doesn't support EXIF in the same way, save as-is
            pil_image.save(output_file, "PNG")
        else:
            pil_image.save(output_file, "TIFF")
        
        LOGGER.info("Saved %s", output_file)
    
    # Write a summary file with metadata
    summary_file = output_path / f"{input_path.stem}_metadata.txt"
    with open(summary_file, "w") as f:
        f.write(f"Source: {input_path.name}\n")
        f.write(f"Source size: {w_eq} x {h_eq}\n")
        f.write(f"Face size: {size} x {size}\n")
        f.write(f"Field of view: {fov} degrees\n")
        f.write(f"Focal length: {f_px:.2f} px\n")
        f.write(f"Focal length (35mm equiv): {f_35mm:.2f} mm\n")
        f.write(f"Faces extracted: {', '.join(faces_to_extract)}\n")
        f.write("\nFace orientations:\n")
        for face_name in faces_to_extract:
            yaw, pitch = CUBE_FACES[face_name]
            f.write(f"  {face_name}: yaw={yaw}°, pitch={pitch}°\n")
    
    LOGGER.info("Wrote metadata to %s", summary_file)
    LOGGER.info("Done! Extracted %d cube faces to %s", len(faces_to_extract), output_path)


if __name__ == "__main__":
    equirect_to_cubefaces_cli()

