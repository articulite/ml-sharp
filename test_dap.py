"""Quick test script to verify DAP model works.

Run from project root:
    uv run python test_dap.py
"""

import sys
import os
from pathlib import Path

# Add DAP to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "DAP"))

import torch
import numpy as np
import cv2

def test_dap():
    print("=" * 50)
    print("DAP Model Test")
    print("=" * 50)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Paths
    weights_path = PROJECT_ROOT / "checkpoints" / "dap" / "model.pth"
    image_path = PROJECT_ROOT / "data" / "input.jpg"
    output_path = PROJECT_ROOT / "data" / "input_depth.png"
    output_npy_path = PROJECT_ROOT / "data" / "input_depth.npy"
    
    print(f"\nWeights: {weights_path}")
    print(f"Exists: {weights_path.exists()}")
    print(f"\nInput: {image_path}")
    print(f"Exists: {image_path.exists()}")
    
    if not weights_path.exists():
        print("ERROR: Weights not found!")
        return False
    
    if not image_path.exists():
        print("ERROR: Input image not found!")
        return False
    
    # Load image
    print("\n[1/4] Loading image...")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("ERROR: Failed to load image!")
        return False
    
    h, w = img_bgr.shape[:2]
    print(f"Image size: {w}x{h} (aspect ratio: {w/h:.2f})")
    
    # Create model
    print("\n[2/4] Creating DAP model...")
    from networks.models import make
    
    model = make({
        'name': 'dap',
        'args': {
            'midas_model_type': 'vitl',
            'fine_tune_type': 'hypersim',
            'min_depth': 0.01,
            'max_depth': 1.0,  # Note: output is 0-1, represents 0-100m
            'train_decoder': True,
        }
    })
    
    # Load weights
    print("\n[3/4] Loading weights...")
    state_dict = torch.load(weights_path, map_location=device)
    
    # Handle DataParallel weights
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Run inference
    print("\n[4/4] Running inference...")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        depth = model.infer_image(img_rgb, input_size=518)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Depth mean: {depth.mean():.4f}")
    
    # Save raw depth as numpy
    np.save(output_npy_path, depth)
    print(f"\nSaved raw depth: {output_npy_path}")
    
    # Visualize depth
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_path), depth_colored)
    print(f"Saved visualization: {output_path}")
    
    print("\n" + "=" * 50)
    print("SUCCESS! DAP model is working.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_dap()
    sys.exit(0 if success else 1)

