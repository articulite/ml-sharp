# Depth Harmonization for Cubemap Gaussian Splats

## Problem Statement

When generating 3D Gaussian splats from a 360° equirectangular image, the standard approach is:

1. Project the equirectangular image onto 6 cube faces
2. Run monocular depth prediction independently on each face
3. Unproject each face's depth to 3D Gaussians
4. Combine all 6 faces into a single scene

**The problem:** Each face is processed by the depth model independently, with no knowledge of neighboring faces. This causes:

- **Scale ambiguity**: Face A might predict depths in range [1, 5], Face B might predict [1.2, 6] for the same scene
- **Geometric discontinuities**: Objects that span face boundaries appear at different depths on each side
- **Visual seams**: When rendered, the edges between faces show obvious misalignment

```
    FRONT FACE                    RIGHT FACE
   ┌──────────────┐              ┌──────────────┐
   │              │              │              │
   │   depth=2.0  │◄── SEAM ───►│   depth=2.4  │
   │              │              │              │
   │  (same obj)  │              │  (same obj)  │
   └──────────────┘              └──────────────┘
   
   The same object appears at different depths!
```

---

## Key Insight: Overlapping Field of View

The Sharp pipeline extracts cube faces with **110° FOV** instead of the minimum 90° required for a cube. This creates a **20° overlap** between adjacent faces.

```
                    90° (minimum)
              ◄─────────────────────►
        ┌─────────────────────────────────┐
        │                                 │
        │     Standard Cube Face          │
        │                                 │
        └─────────────────────────────────┘
        
              ◄───────────────────────────────────►
                         110° (actual)
        ┌─────────────────────────────────────────┐
        │ OVR │                           │ OVR │
        │ LAP │     Extended Cube Face    │ LAP │
        │     │                           │     │
        └─────────────────────────────────────────┘
              ◄───►                       ◄───►
               10°                         10°
              overlap                    overlap
              with                       with
              left                       right
              face                       face
```

**This overlap is the key to solving the problem.**

In the overlap region:
- The **same world point** is visible in **two different faces**
- We get **two independent depth predictions** for **identical 3D locations**
- The ratio between these predictions tells us the **relative scale error** between faces

---

## Mathematical Formulation

### Coordinate Systems

For a point in the equirectangular image at spherical coordinates (θ, φ):
- θ = azimuth (longitude), range [-π, π]
- φ = elevation (latitude), range [-π/2, π/2]

Each cube face covers a region of the sphere. With 110° FOV:
- Front face: θ ∈ [-55°, +55°], φ ∈ [-55°, +55°]
- Right face: θ ∈ [+35°, +145°], φ ∈ [-55°, +55°]
- Overlap (front-right): θ ∈ [+35°, +55°], φ ∈ [-55°, +55°]

### The Overlap Correspondence

For any world direction **d** = (θ, φ) in an overlap region:

```
d_world ──┬──► Face A pixel (u_A, v_A) ──► Depth prediction z_A
          │
          └──► Face B pixel (u_B, v_B) ──► Depth prediction z_B
```

Both `z_A` and `z_B` are predictions for the **same 3D point**. In a perfect world:
```
z_A = z_B
```

In reality, due to scale ambiguity:
```
z_A ≈ s_AB · z_B
```

where `s_AB` is an unknown scale factor.

### Global Scale Optimization

We want to find a scale correction `s_i` for each face `i` such that scaled depths agree in all overlaps.

**Objective function:**

```
minimize  Σ     Σ        w(p) · (s_i · z_i(p) - s_j · z_j(p))²
         (i,j) p∈Overlap(i,j)
```

Where:
- Sum is over all adjacent face pairs (i, j)
- p iterates over all correspondence points in the overlap
- z_i(p) is face i's depth prediction at world direction p
- s_i is the scale correction for face i
- w(p) is an optional weight (e.g., confidence, distance from edge)

**Constraint:** Fix one face's scale to avoid trivial solution:
```
s_front = 1.0  (or any reference face)
```

### Solving the System

This is a **linear least squares** problem. Let's reformulate:

For each correspondence point p between faces i and j:
```
s_i · z_i(p) = s_j · z_j(p)
s_i · z_i(p) - s_j · z_j(p) = 0
```

Stack all equations into matrix form:
```
A · s = 0
```

Where:
- s = [s_front, s_back, s_left, s_right, s_top, s_bottom]ᵀ
- A is a sparse matrix with one row per correspondence

With the constraint s_front = 1:
```
A_reduced · s_reduced = b
```

Solve via least squares:
```
s_reduced = (AᵀA)⁻¹ Aᵀ b
```

---

## Algorithm

### Step 1: Generate Overlap Correspondences

For each pair of adjacent faces, compute the pixel-to-pixel mapping in their overlap region.

```python
def compute_overlap_correspondences(face_a, face_b, fov_deg=110):
    """
    Compute pixel correspondences between two adjacent cube faces.
    
    Returns:
        List of (pixel_a, pixel_b, world_direction) tuples
    """
    correspondences = []
    
    # Determine overlap region in spherical coordinates
    overlap_theta_min, overlap_theta_max = get_overlap_bounds(face_a, face_b, fov_deg)
    
    # Sample world directions in overlap
    for theta in linspace(overlap_theta_min, overlap_theta_max, num_samples):
        for phi in linspace(-fov_deg/2, fov_deg/2, num_samples):
            world_dir = spherical_to_cartesian(theta, phi)
            
            # Project to both faces
            pixel_a = project_to_face(world_dir, face_a, fov_deg)
            pixel_b = project_to_face(world_dir, face_b, fov_deg)
            
            if pixel_a is not None and pixel_b is not None:
                correspondences.append((pixel_a, pixel_b, world_dir))
    
    return correspondences
```

### Step 2: Extract Depth at Correspondences

After running depth prediction on all faces, sample the predicted depths at correspondence points.

```python
def extract_depth_correspondences(faces_depth, correspondences_by_pair):
    """
    Extract depth values at all correspondence points.
    
    Args:
        faces_depth: Dict mapping face_name -> depth_map (H, W)
        correspondences_by_pair: Dict mapping (face_a, face_b) -> correspondences
    
    Returns:
        List of (face_a, face_b, z_a, z_b) observations
    """
    observations = []
    
    for (face_a, face_b), correspondences in correspondences_by_pair.items():
        depth_a = faces_depth[face_a]
        depth_b = faces_depth[face_b]
        
        for (pixel_a, pixel_b, _) in correspondences:
            z_a = bilinear_sample(depth_a, pixel_a)
            z_b = bilinear_sample(depth_b, pixel_b)
            
            # Filter outliers (sky, invalid depth)
            if z_a > 0 and z_b > 0 and z_a < max_depth and z_b < max_depth:
                observations.append((face_a, face_b, z_a, z_b))
    
    return observations
```

### Step 3: Solve for Scale Factors

Build and solve the linear system.

```python
def solve_scale_factors(observations, reference_face='front'):
    """
    Solve for per-face scale factors that minimize depth disagreement.
    
    Args:
        observations: List of (face_a, face_b, z_a, z_b) tuples
        reference_face: Face to fix at scale=1.0
    
    Returns:
        Dict mapping face_name -> scale_factor
    """
    faces = ['front', 'back', 'left', 'right', 'top', 'bottom']
    face_to_idx = {f: i for i, f in enumerate(faces)}
    n_faces = len(faces)
    ref_idx = face_to_idx[reference_face]
    
    # Build linear system: for each observation, s_a * z_a - s_b * z_b = 0
    # With s_ref = 1, this becomes: s_a * z_a - s_b * z_b = 0 (if neither is ref)
    # Or: z_a - s_b * z_b = 0 (if a is ref)
    # Or: s_a * z_a - z_b = 0 (if b is ref)
    
    # Use robust estimation (e.g., RANSAC or iteratively reweighted least squares)
    # to handle outliers from depth prediction errors
    
    A = []
    b = []
    
    for (face_a, face_b, z_a, z_b) in observations:
        idx_a = face_to_idx[face_a]
        idx_b = face_to_idx[face_b]
        
        row = [0] * n_faces
        rhs = 0
        
        if idx_a == ref_idx:
            # z_a - s_b * z_b = 0  =>  -z_b * s_b = -z_a
            row[idx_b] = -z_b
            rhs = -z_a
        elif idx_b == ref_idx:
            # s_a * z_a - z_b = 0  =>  z_a * s_a = z_b
            row[idx_a] = z_a
            rhs = z_b
        else:
            # s_a * z_a - s_b * z_b = 0
            row[idx_a] = z_a
            row[idx_b] = -z_b
            rhs = 0
        
        A.append(row)
        b.append(rhs)
    
    A = np.array(A)
    b = np.array(b)
    
    # Remove reference face column (it's fixed at 1.0)
    A_reduced = np.delete(A, ref_idx, axis=1)
    
    # Solve least squares
    scales_reduced, residuals, rank, s = np.linalg.lstsq(A_reduced, b, rcond=None)
    
    # Insert reference scale back
    scales = np.insert(scales_reduced, ref_idx, 1.0)
    
    return {face: scales[i] for i, face in enumerate(faces)}
```

### Step 4: Apply Scale Corrections

Apply the computed scale factors to the Gaussian positions.

```python
def apply_scale_corrections(gaussians_per_face, scale_factors):
    """
    Scale the depth (Z coordinate) of Gaussians in each face.
    
    Args:
        gaussians_per_face: Dict mapping face_name -> Gaussians3D
        scale_factors: Dict mapping face_name -> scale
    
    Returns:
        Corrected gaussians_per_face
    """
    corrected = {}
    
    for face, gaussians in gaussians_per_face.items():
        scale = scale_factors[face]
        
        # Scale the Z coordinate (depth) of mean vectors
        # In face-local coordinates, Z is the depth direction
        corrected_means = gaussians.mean_vectors.clone()
        corrected_means[..., 2] *= scale
        
        # Also scale the Z component of the Gaussian ellipsoids
        corrected_scales = gaussians.singular_values.clone()
        corrected_scales[..., 2] *= scale
        
        corrected[face] = Gaussians3D(
            mean_vectors=corrected_means,
            singular_values=corrected_scales,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )
    
    return corrected
```

---

## Cube Face Adjacency Graph

The 6 cube faces form a graph where edges represent adjacency (shared boundaries):

```
                    ┌─────────┐
                    │   TOP   │
                    │    ↑    │
                    └────┬────┘
                         │
         ┌─────────┬─────┴─────┬─────────┐
         │  LEFT   │   FRONT   │  RIGHT  │
         │    ←    │     ●     │    →    │
         └────┬────┴─────┬─────┴────┬────┘
              │          │          │
              │     ┌────┴────┐     │
              │     │  BACK   │     │
              │     │    ↓    │     │
              │     └────┬────┘     │
              │          │          │
              │     ┌────┴────┐     │
              └─────┤ BOTTOM  ├─────┘
                    │         │
                    └─────────┘
```

**Adjacency pairs (12 edges):**

| Face A | Face B | Shared Edge |
|--------|--------|-------------|
| front  | right  | right edge of front |
| front  | left   | left edge of front |
| front  | top    | top edge of front |
| front  | bottom | bottom edge of front |
| back   | right  | left edge of back |
| back   | left   | right edge of back |
| back   | top    | top edge of back |
| back   | bottom | bottom edge of back |
| left   | top    | left edge of top |
| left   | bottom | left edge of bottom |
| right  | top    | right edge of top |
| right  | bottom | right edge of bottom |

---

## Handling Edge Cases

### 1. Sky Regions (Infinite Depth)

The depth model may predict very large depths for sky regions. These should be:
- Filtered out from correspondence matching (set a max_depth threshold)
- Or given low weight in the optimization

```python
# Filter sky regions
valid_mask = (z_a < max_depth) & (z_b < max_depth) & (z_a > min_depth) & (z_b > min_depth)
```

### 2. Outlier Predictions

Some correspondence points may have grossly incorrect depth predictions. Use robust estimation:

```python
# Option 1: RANSAC
from sklearn.linear_model import RANSACRegressor

# Option 2: Iteratively Reweighted Least Squares (IRLS)
for iteration in range(max_iterations):
    residuals = compute_residuals(A, scales, b)
    weights = huber_weights(residuals)
    scales = weighted_least_squares(A, b, weights)

# Option 3: Median-based initialization
median_ratios = {}
for (face_a, face_b), obs in observations_by_pair.items():
    ratios = [z_a / z_b for (_, _, z_a, z_b) in obs]
    median_ratios[(face_a, face_b)] = np.median(ratios)
```

### 3. Weak Overlap Signal

Some face pairs may have poor depth predictions in their overlap (e.g., uniform textureless regions). Weight observations by local texture/gradient strength:

```python
def compute_observation_weight(pixel, image, depth):
    """Weight based on local texture and depth confidence."""
    # Texture weight: high gradient = more reliable
    gradient = sobel(image)[pixel]
    texture_weight = min(gradient / gradient_threshold, 1.0)
    
    # Depth confidence: middle depths are more reliable than extremes
    depth_weight = gaussian(depth, mean=typical_depth, std=depth_range/4)
    
    return texture_weight * depth_weight
```

### 4. Scale Factor Sanity Checks

After solving, verify scale factors are reasonable:

```python
def validate_scales(scale_factors):
    scales = list(scale_factors.values())
    
    # Check range
    if max(scales) / min(scales) > 2.0:
        warnings.warn("Large scale variation detected - may indicate poor overlap matching")
    
    # Check for negative scales (should never happen)
    if any(s <= 0 for s in scales):
        raise ValueError("Negative scale factor computed - check input data")
    
    return True
```

---

## Integration Points

### Option A: Pre-Splat Harmonization (Recommended)

Apply harmonization to depth maps before Gaussian prediction:

```
Equirect → Cube Faces → [Depth Prediction] → [HARMONIZE DEPTHS] → Gaussian Prediction → PLY
```

**Advantages:**
- Corrects depth before any 3D computation
- Cleaner separation of concerns
- Can visualize harmonized depth maps for debugging

### Option B: Post-Splat Harmonization

Apply harmonization to Gaussian positions after prediction:

```
Equirect → Cube Faces → Depth Prediction → Gaussian Prediction → [HARMONIZE POSITIONS] → PLY
```

**Advantages:**
- Doesn't require modifying the prediction pipeline
- Can be applied to existing PLY files
- Simpler to implement as a post-process

### Option C: Hybrid (Depth + Position)

1. Harmonize depths (Option A)
2. Run Gaussian prediction
3. Fine-tune positions using overlap Gaussian correspondences (Option B)

---

## Expected Results

### Before Harmonization
```
Face      | Depth Range  | Scale Factor
----------|--------------|-------------
front     | [0.8, 12.3]  | 1.00 (ref)
right     | [0.9, 14.1]  | ~1.15
back      | [0.7, 11.8]  | ~0.96
left      | [1.0, 15.2]  | ~1.24
top       | [0.6, 10.1]  | ~0.82
bottom    | [0.8, 13.0]  | ~1.06
```

### After Harmonization
```
Face      | Depth Range  | Scale Factor | Corrected Range
----------|--------------|--------------|----------------
front     | [0.8, 12.3]  | 1.00         | [0.8, 12.3]
right     | [0.9, 14.1]  | 0.87         | [0.8, 12.3]
back      | [0.7, 11.8]  | 1.04         | [0.8, 12.3]
left      | [1.0, 15.2]  | 0.81         | [0.8, 12.3]
top       | [0.6, 10.1]  | 1.22         | [0.8, 12.3]
bottom    | [0.8, 13.0]  | 0.95         | [0.8, 12.3]
```

---

## Visualization & Debugging

### 1. Overlap Region Visualization

```python
def visualize_overlaps(cube_faces, fov_deg=110):
    """Highlight overlap regions on each cube face."""
    for face_name, face_image in cube_faces.items():
        overlay = face_image.copy()
        
        # Mark overlap regions in semi-transparent red
        for adjacent_face in get_adjacent_faces(face_name):
            overlap_mask = compute_overlap_mask(face_name, adjacent_face, fov_deg)
            overlay[overlap_mask] = 0.5 * overlay[overlap_mask] + 0.5 * RED
        
        save_image(overlay, f"{face_name}_overlaps.png")
```

### 2. Depth Correspondence Scatter Plot

```python
def plot_depth_correspondences(observations, face_a, face_b):
    """Scatter plot of depth_a vs depth_b for a face pair."""
    depths_a = [z_a for (fa, fb, z_a, z_b) in observations if fa == face_a and fb == face_b]
    depths_b = [z_b for (fa, fb, z_a, z_b) in observations if fa == face_a and fb == face_b]
    
    plt.scatter(depths_a, depths_b, alpha=0.3)
    plt.plot([0, max_depth], [0, max_depth], 'r--', label='Perfect agreement')
    plt.xlabel(f'{face_a} depth')
    plt.ylabel(f'{face_b} depth')
    plt.title(f'Depth Correspondence: {face_a} vs {face_b}')
    
    # Fit line shows actual relationship
    slope = np.polyfit(depths_a, depths_b, 1)[0]
    plt.plot([0, max_depth], [0, slope * max_depth], 'g-', label=f'Fit (slope={slope:.2f})')
    
    plt.legend()
    plt.savefig(f'depth_corr_{face_a}_{face_b}.png')
```

### 3. Before/After Seam Comparison

```python
def visualize_seam_alignment(gaussians_before, gaussians_after, face_a, face_b):
    """Render the seam region before and after harmonization."""
    # Extract gaussians near the shared edge
    edge_gaussians_before = extract_edge_region(gaussians_before, face_a, face_b)
    edge_gaussians_after = extract_edge_region(gaussians_after, face_a, face_b)
    
    # Render side-by-side
    render_comparison(edge_gaussians_before, edge_gaussians_after, 
                     title=f'Seam: {face_a}-{face_b}')
```

---

## Performance Considerations

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Compute correspondences | O(n_samples² × n_pairs) | ~100ms |
| Extract depths | O(n_correspondences) | ~50ms |
| Solve linear system | O(n_observations × n_faces²) | ~10ms |
| Apply corrections | O(n_gaussians) | ~20ms |
| **Total** | | **~200ms** |

The harmonization adds negligible overhead compared to the depth prediction (~2-5 seconds per face).

---

## Future Enhancements

### 1. Affine Depth Correction
Instead of just scale, solve for scale + offset:
```
z_corrected = scale * z_predicted + offset
```

This handles cases where the depth model has both multiplicative and additive bias.

### 2. Spatially-Varying Scale
Instead of one scale per face, compute a smooth scale field:
```
z_corrected(u, v) = scale_field(u, v) * z_predicted(u, v)
```

This handles cases where scale error varies across the face (e.g., worse at corners).

### 3. Multi-Resolution Matching
Compute correspondences at multiple resolutions:
- Coarse: overall scale correction
- Fine: local adjustments near edges

### 4. Temporal Consistency (for video)
When processing video panoramas, add temporal smoothness:
```
scale_t = α * scale_computed + (1-α) * scale_{t-1}
```

---

## References

1. Monocular Depth Estimation scale ambiguity: [Eigen et al., 2014]
2. Multi-view depth fusion: [Galliani et al., 2015]
3. Panoramic depth estimation: [Wang et al., 2020]
4. Bundle adjustment for scale recovery: [Triggs et al., 2000]

