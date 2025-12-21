"""CLI for depth harmonization of cubemap Gaussian splats.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import logging as logging_utils
from sharp.utils.depth_harmonization import (
    CUBE_FACES,
    harmonize_gaussians,
    HarmonizationResult,
)

LOGGER = logging.getLogger(__name__)


def load_ply_positions(path: Path) -> tuple[np.ndarray, PlyData]:
    """Load Gaussian positions from PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    
    positions = np.stack([
        np.asarray(vertices["x"]),
        np.asarray(vertices["y"]),
        np.asarray(vertices["z"]),
    ], axis=1)
    
    return positions, plydata


def save_ply_with_new_positions(
    plydata: PlyData, 
    new_positions: np.ndarray, 
    output_path: Path
) -> None:
    """Save PLY with updated positions, preserving all other attributes."""
    vertices = plydata["vertex"]
    
    # Create new vertex array with updated positions
    dtype = vertices.data.dtype
    new_data = np.empty(len(vertices.data), dtype=dtype)
    
    # Copy all existing data
    for name in dtype.names:
        new_data[name] = vertices.data[name]
    
    # Update positions
    new_data["x"] = new_positions[:, 0]
    new_data["y"] = new_positions[:, 1]
    new_data["z"] = new_positions[:, 2]
    
    # Also update scales proportionally if present
    # (This maintains splat shapes relative to their new positions)
    
    new_vertex = PlyElement.describe(new_data, "vertex")
    
    # Preserve other elements
    other_elements = [el for el in plydata.elements if el.name != "vertex"]
    
    new_plydata = PlyData([new_vertex] + other_elements)
    new_plydata.write(output_path)


def transform_positions_to_world(positions: np.ndarray, face: str) -> np.ndarray:
    """Transform positions from face-local to world coordinates."""
    from sharp.utils.depth_harmonization import get_face_rotation_matrix
    
    R = get_face_rotation_matrix(face)
    
    # The PLY positions are in a coordinate system where:
    # - Z points into the scene (depth)
    # - X points right
    # - Y points down
    # 
    # We need to rotate to world coordinates based on face orientation
    
    # First, flip Y to match world convention (Y up)
    positions_flipped = positions.copy()
    positions_flipped[:, 1] *= -1
    
    # Apply face rotation
    world_positions = (R @ positions_flipped.T).T
    
    return world_positions


def transform_positions_from_world(positions: np.ndarray, face: str) -> np.ndarray:
    """Transform positions from world to face-local coordinates."""
    from sharp.utils.depth_harmonization import get_face_rotation_matrix
    
    R = get_face_rotation_matrix(face)
    
    # Inverse rotation
    local_positions = (R.T @ positions.T).T
    
    # Flip Y back
    local_positions[:, 1] *= -1
    
    return local_positions


@click.command()
@click.option(
    "-i", "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Directory containing cube face PLY files (input_front.ply, etc.)",
)
@click.option(
    "-o", "--output-path", 
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for harmonized PLY files. Defaults to input_path/harmonized/",
)
@click.option(
    "--fov",
    type=float,
    default=110.0,
    help="Field of view used for cube face extraction (default: 110)",
)
@click.option(
    "--reference",
    type=click.Choice(list(CUBE_FACES.keys())),
    default="front",
    help="Reference face to fix at scale=1.0 (default: front)",
)
@click.option(
    "--samples",
    type=int,
    default=50,
    help="Samples per dimension for overlap matching (default: 50)",
)
@click.option(
    "--min-depth",
    type=float,
    default=0.1,
    help="Minimum valid depth (default: 0.1)",
)
@click.option(
    "--max-depth",
    type=float,
    default=100.0,
    help="Maximum valid depth (default: 100.0)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def harmonize_cli(
    input_path: Path,
    output_path: Path | None,
    fov: float,
    reference: str,
    samples: int,
    min_depth: float,
    max_depth: float,
    verbose: bool,
):
    """Harmonize depth across cube face Gaussian splats.
    
    This tool reads PLY files for each cube face, computes scale corrections
    using overlap correspondences, and outputs harmonized PLY files.
    
    Example:
    
        python -m sharp.cli.harmonize -i ./job_xxx/splats/ -v
    
    The input directory should contain files named input_front.ply, input_back.ply, etc.
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    
    if output_path is None:
        output_path = input_path / "harmonized"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find and load all cube face PLY files
    LOGGER.info(f"Loading PLY files from {input_path}")
    
    face_data = {}  # face -> (positions_world, plydata)
    
    for face in CUBE_FACES.keys():
        ply_path = input_path / f"input_{face}.ply"
        if not ply_path.exists():
            LOGGER.warning(f"Missing {ply_path}")
            continue
        
        positions_local, plydata = load_ply_positions(ply_path)
        positions_world = transform_positions_to_world(positions_local, face)
        
        face_data[face] = {
            "positions_local": positions_local,
            "positions_world": positions_world,
            "plydata": plydata,
        }
        
        LOGGER.info(f"  {face}: {len(positions_local)} Gaussians")
    
    if len(face_data) < 2:
        LOGGER.error("Need at least 2 faces for harmonization")
        return
    
    # Extract world positions for harmonization
    positions_per_face = {
        face: data["positions_world"] 
        for face, data in face_data.items()
    }
    
    # Run harmonization
    LOGGER.info("Running depth harmonization...")
    corrected_world, result = harmonize_gaussians(
        positions_per_face,
        fov_deg=fov,
        n_samples=samples,
        reference_face=reference,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    
    # Save results
    LOGGER.info(f"Saving harmonized PLY files to {output_path}")
    
    for face, data in face_data.items():
        if face not in corrected_world:
            continue
        
        # Transform back to face-local coordinates
        corrected_local = transform_positions_from_world(corrected_world[face], face)
        
        output_file = output_path / f"input_{face}.ply"
        save_ply_with_new_positions(data["plydata"], corrected_local, output_file)
        
        scale = result.scale_factors.get(face, 1.0)
        LOGGER.info(f"  {face}: scale={scale:.4f} -> {output_file}")
    
    # Print summary
    LOGGER.info("")
    LOGGER.info("=" * 50)
    LOGGER.info("HARMONIZATION SUMMARY")
    LOGGER.info("=" * 50)
    LOGGER.info(f"Total correspondences: {result.n_correspondences}")
    LOGGER.info(f"RMS depth residual: {result.residual_before:.4f} -> {result.residual_after:.4f}")
    LOGGER.info(f"Improvement: {(1 - result.residual_after/max(result.residual_before, 1e-6))*100:.1f}%")
    LOGGER.info("")
    LOGGER.info("Scale factors:")
    for face in CUBE_FACES.keys():
        if face in result.scale_factors:
            marker = " (ref)" if face == reference else ""
            LOGGER.info(f"  {face:8s}: {result.scale_factors[face]:.4f}{marker}")
    LOGGER.info("")
    LOGGER.info("Correspondences per face pair:")
    for (fa, fb), count in sorted(result.correspondences_per_pair.items()):
        LOGGER.info(f"  {fa:8s} - {fb:8s}: {count}")


if __name__ == "__main__":
    harmonize_cli()

