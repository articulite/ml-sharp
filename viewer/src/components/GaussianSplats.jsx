import { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { loadPLY } from '../utils/plyLoader';

/**
 * Frustum culling modes:
 * 0 = None - render all gaussians
 * 1 = CPU Pre-filter - filter gaussians on load, only keep those inside frustum
 * 2 = GPU Vertex Discard - check in vertex shader, move outside-frustum splats off-screen
 * 3 = GPU Fragment Discard - transform to world space in fragment and discard outside frustum
 */

// Vertex shader with proper 2D covariance projection
const vertexShader = `
  precision highp float;
  
  attribute vec3 splatCenter;
  attribute vec3 splatColor;
  attribute float splatOpacity;
  attribute vec3 splatScale;
  attribute vec4 splatRotation;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vConicA;  // conic matrix elements for ellipse
  varying float vConicB;
  varying vec2 vCenterOffset;
  varying vec3 vWorldPos;  // For GPU fragment culling
  
  uniform vec2 viewport;
  uniform vec2 focal;
  uniform float splatScaleMult;
  uniform int cullMode;
  uniform vec3 frustumDir;  // Direction the frustum faces (normalized)
  uniform float frustumAngle; // Half angle in radians (45° for 90° FOV)
  uniform int orientMode;  // 0=stored rotation, 1=orient to center, 2=billboard to camera
  uniform vec3 cameraWorldPos;  // Camera position for billboard mode
  
  // Build rotation matrix from quaternion (w, x, y, z)
  mat3 quatToMat3(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;  // PLY stores as rot_0=w, rot_1=x, rot_2=y, rot_3=z
    return mat3(
      1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
      2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
      2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
    );
  }
  
  // Build rotation matrix that orients Z axis toward a target point
  mat3 lookAtRotationToward(vec3 from, vec3 to) {
    vec3 forward = normalize(to - from);
    
    // Handle degenerate case
    if (length(to - from) < 0.0001) {
      return mat3(1.0);
    }
    
    // Find a suitable "up" vector (avoid parallel with forward)
    vec3 up = abs(forward.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    
    vec3 right = normalize(cross(up, forward));
    up = cross(forward, right);
    
    // Build rotation matrix: columns are right, up, forward
    return mat3(right, up, forward);
  }
  
  // Check if point is inside 90-degree square pyramidal frustum
  bool isInsideFrustum(vec3 worldPos) {
    // Depth along frustum direction
    float depth = dot(worldPos, frustumDir);
    if (depth <= 0.0) return false;  // Behind the apex
    
    // Project point onto plane perpendicular to frustum direction
    vec3 projOnAxis = depth * frustumDir;
    vec3 perpComponent = worldPos - projOnAxis;
    
    // For a SQUARE pyramid, check each perpendicular axis separately
    // Max allowed distance along each axis = depth * tan(halfAngle)
    float maxDist = depth * tan(frustumAngle);
    
    // Get the two perpendicular axes based on frustum direction
    // For axis-aligned frustums, we check the other two world axes
    vec3 absDir = abs(frustumDir);
    
    if (absDir.z > 0.5) {
      // Frustum along Z axis - check X and Y
      return abs(perpComponent.x) <= maxDist && abs(perpComponent.y) <= maxDist;
    } else if (absDir.x > 0.5) {
      // Frustum along X axis - check Y and Z
      return abs(perpComponent.y) <= maxDist && abs(perpComponent.z) <= maxDist;
    } else {
      // Frustum along Y axis - check X and Z
      return abs(perpComponent.x) <= maxDist && abs(perpComponent.z) <= maxDist;
    }
  }
  
  void main() {
    vColor = splatColor;
    vOpacity = splatOpacity;
    
    // Compute world position for frustum culling
    vec4 worldPos4 = modelMatrix * vec4(splatCenter, 1.0);
    vWorldPos = worldPos4.xyz;
    
    // GPU Vertex culling mode (mode 2)
    if (cullMode == 2 && !isInsideFrustum(vWorldPos)) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // Move to clip space
      return;
    }
    
    // Transform center to view space
    vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
    
    // Skip splats behind camera
    if (viewCenter.z > -0.1) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
    
    // Build 3D covariance in view space
    // Cov = M * diag(s^2) * M^T where M = ViewRot * GaussianRot
    mat3 R;
    vec3 s = splatScale * splatScaleMult;
    
    if (orientMode == 1) {
      // Orient gaussian to face toward origin (0,0,0)
      R = lookAtRotationToward(splatCenter, vec3(0.0));
    } else if (orientMode == 2) {
      // Billboard: orient gaussian to face the camera
      // Use the two largest scales to make a flat disc, ensures no stretching from any angle
      R = lookAtRotationToward(splatCenter, cameraWorldPos);
      float maxScale = max(s.x, max(s.y, s.z));
      float midScale = s.x + s.y + s.z - maxScale - min(s.x, min(s.y, s.z));
      s = vec3(maxScale, midScale, 0.001);  // Flat disc with 2 largest scales
    } else {
      // Use stored rotation from PLY
      R = quatToMat3(splatRotation);
    }
    mat3 M = mat3(modelViewMatrix) * R;
    
    // Scaled axes: each column of M scaled by corresponding scale
    vec3 a0 = M[0] * s.x;
    vec3 a1 = M[1] * s.y;
    vec3 a2 = M[2] * s.z;
    
    // 3D covariance = sum of outer products: a0*a0^T + a1*a1^T + a2*a2^T
    // We only need the upper triangle (symmetric matrix)
    float c00 = a0.x*a0.x + a1.x*a1.x + a2.x*a2.x;
    float c01 = a0.x*a0.y + a1.x*a1.y + a2.x*a2.y;
    float c02 = a0.x*a0.z + a1.x*a1.z + a2.x*a2.z;
    float c11 = a0.y*a0.y + a1.y*a1.y + a2.y*a2.y;
    float c12 = a0.y*a0.z + a1.y*a1.z + a2.y*a2.z;
    float c22 = a0.z*a0.z + a1.z*a1.z + a2.z*a2.z;
    
    // Project to 2D using Jacobian of perspective projection
    // J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    float z = -viewCenter.z;
    float z2 = z * z;
    float fx = focal.x;
    float fy = focal.y;
    
    float j00 = fx / z;
    float j02 = -fx * viewCenter.x / z2;
    float j11 = fy / z;
    float j12 = -fy * viewCenter.y / z2;
    
    // Cov2D = J * Cov3D * J^T (2x2 result)
    // First compute J * Cov3D (2x3 matrix)
    // Row 0: [j00*c00 + j02*c02, j00*c01 + j02*c12, j00*c02 + j02*c22]
    // Row 1: [j11*c01 + j12*c02, j11*c11 + j12*c12, j11*c12 + j12*c22]
    float t00 = j00*c00 + j02*c02;
    float t01 = j00*c01 + j02*c12;
    float t02 = j00*c02 + j02*c22;
    float t10 = j11*c01 + j12*c02;
    float t11 = j11*c11 + j12*c12;
    float t12 = j11*c12 + j12*c22;
    
    // Then (J*Cov3D) * J^T
    float cov2D_00 = t00*j00 + t02*j02;
    float cov2D_01 = t00*0.0 + t01*j11 + t02*j12;  // t00*0
    float cov2D_11 = t10*0.0 + t11*j11 + t12*j12;  // t10*0
    
    // Add low-pass filter to avoid aliasing (variance of 0.3 pixels)
    cov2D_00 += 0.3;
    cov2D_11 += 0.3;
    
    // Compute eigenvalues for bounding box size
    float det = cov2D_00 * cov2D_11 - cov2D_01 * cov2D_01;
    float mid = 0.5 * (cov2D_00 + cov2D_11);
    float disc = max(0.0, mid * mid - det);
    float lambda1 = mid + sqrt(disc);
    float lambda2 = max(0.0, mid - sqrt(disc));
    
    // Splat radius: 3 sigma of larger eigenvalue
    float radius = ceil(3.0 * sqrt(lambda1));
    radius = clamp(radius, 1.0, 1024.0);
    
    // Compute inverse covariance (conic) for fragment shader
    float invDet = 1.0 / max(det, 1e-6);
    vConicA = vec2(cov2D_11 * invDet, cov2D_00 * invDet);
    vConicB = -cov2D_01 * invDet;
    
    // Screen position of splat center
    vec4 clipPos = projectionMatrix * viewCenter;
    vec2 ndcCenter = clipPos.xy / clipPos.w;
    vec2 screenCenter = (ndcCenter * 0.5 + 0.5) * viewport;
    
    // Billboard offset in pixels
    vec2 offset = position.xy * radius;
    vCenterOffset = offset;
    
    // Convert back to clip space
    vec2 screenPos = screenCenter + offset;
    vec2 ndcPos = (screenPos / viewport) * 2.0 - 1.0;
    
    gl_Position = vec4(ndcPos * clipPos.w, clipPos.z, clipPos.w);
  }
`;

const fragmentShader = `
  precision highp float;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vConicA;
  varying float vConicB;
  varying vec2 vCenterOffset;
  varying vec3 vWorldPos;
  
  uniform int cullMode;
  uniform vec3 frustumDir;
  uniform float frustumAngle;
  
  // Check if point is inside 90-degree square pyramidal frustum
  bool isInsideFrustum(vec3 worldPos) {
    float depth = dot(worldPos, frustumDir);
    if (depth <= 0.0) return false;
    
    vec3 projOnAxis = depth * frustumDir;
    vec3 perpComponent = worldPos - projOnAxis;
    float maxDist = depth * tan(frustumAngle);
    
    vec3 absDir = abs(frustumDir);
    
    if (absDir.z > 0.5) {
      return abs(perpComponent.x) <= maxDist && abs(perpComponent.y) <= maxDist;
    } else if (absDir.x > 0.5) {
      return abs(perpComponent.y) <= maxDist && abs(perpComponent.z) <= maxDist;
    } else {
      return abs(perpComponent.x) <= maxDist && abs(perpComponent.z) <= maxDist;
    }
  }
  
  void main() {
    // GPU Fragment culling mode (mode 3)
    if (cullMode == 3 && !isInsideFrustum(vWorldPos)) {
      discard;
    }
    
    // Compute Mahalanobis distance using inverse covariance (conic)
    // d^2 = x^T * Cov^-1 * x = a*x^2 + 2*b*x*y + c*y^2
    float x = vCenterOffset.x;
    float y = vCenterOffset.y;
    float power = -0.5 * (vConicA.x * x * x + 2.0 * vConicB * x * y + vConicA.y * y * y);
    
    if (power > 0.0) discard;
    
    float alpha = exp(power) * vOpacity;
    if (alpha < 0.004) discard;
    
    gl_FragColor = vec4(vColor * alpha, alpha);
  }
`;

// Get frustum direction based on face
// Directions must match where gaussians ACTUALLY end up after CUBE_FACE_ROTATIONS transforms
// Empirically determined: left/right needed flipping, others kept original
function getFrustumDirection(face) {
  const SQRT2_2 = Math.SQRT2 / 2;  // ~0.707
  
  switch (face) {
    // Cubemap faces
    case 'front':  return new THREE.Vector3(0, 0, -1);
    case 'back':   return new THREE.Vector3(0, 0, 1);
    case 'left':   return new THREE.Vector3(-1, 0, 0);  // Flipped from original (+1)
    case 'right':  return new THREE.Vector3(1, 0, 0);   // Flipped from original (-1)
    case 'top':    return new THREE.Vector3(0, 1, 0);
    case 'bottom': return new THREE.Vector3(0, -1, 0);
    
    // Cylinder 8 faces (compass directions)
    case 'n':  return new THREE.Vector3(0, 0, -1);            // Same as front
    case 'ne': return new THREE.Vector3(SQRT2_2, 0, -SQRT2_2);   // 45° (flipped X)
    case 'e':  return new THREE.Vector3(1, 0, 0);             // Same as right (flipped)
    case 'se': return new THREE.Vector3(SQRT2_2, 0, SQRT2_2);    // 135° (flipped X)
    case 's':  return new THREE.Vector3(0, 0, 1);             // Same as back
    case 'sw': return new THREE.Vector3(-SQRT2_2, 0, SQRT2_2);   // 225° (flipped X)
    case 'w':  return new THREE.Vector3(-1, 0, 0);            // Same as left (flipped)
    case 'nw': return new THREE.Vector3(-SQRT2_2, 0, -SQRT2_2);  // 315° (flipped X)
    
    default:   return new THREE.Vector3(0, 0, -1);
  }
}

// CPU pre-filter: check if a point is inside the square pyramidal frustum
function isInsideFrustumCPU(x, y, z, frustumDir, halfAngle) {
  // Depth along frustum direction
  const depth = x * frustumDir.x + y * frustumDir.y + z * frustumDir.z;
  if (depth <= 0) return false; // Behind apex
  
  // Perpendicular component from frustum axis
  const projX = depth * frustumDir.x;
  const projY = depth * frustumDir.y;
  const projZ = depth * frustumDir.z;
  
  const perpX = x - projX;
  const perpY = y - projY;
  const perpZ = z - projZ;
  
  // For a SQUARE pyramid, check each perpendicular axis separately
  const maxDist = depth * Math.tan(halfAngle);
  
  const absDirX = Math.abs(frustumDir.x);
  const absDirY = Math.abs(frustumDir.y);
  const absDirZ = Math.abs(frustumDir.z);
  
  if (absDirZ > 0.5) {
    // Frustum along Z axis - check X and Y
    return Math.abs(perpX) <= maxDist && Math.abs(perpY) <= maxDist;
  } else if (absDirX > 0.5) {
    // Frustum along X axis - check Y and Z
    return Math.abs(perpY) <= maxDist && Math.abs(perpZ) <= maxDist;
  } else {
    // Frustum along Y axis - check X and Z
    return Math.abs(perpX) <= maxDist && Math.abs(perpZ) <= maxDist;
  }
}

// Apply CPU frustum culling to splat data
function filterByFrustum(splatData, face, modelMatrix) {
  const frustumDir = getFrustumDirection(face);
  const halfAngle = Math.PI / 4; // 45° for 90° FOV
  
  const count = splatData.count;
  const validIndices = [];
  
  // Transform frustum direction by inverse of model matrix rotation
  // Actually we need to transform positions to world space and check
  const m = modelMatrix;
  
  for (let i = 0; i < count; i++) {
    const lx = splatData.positions[i * 3];
    const ly = splatData.positions[i * 3 + 1];
    const lz = splatData.positions[i * 3 + 2];
    
    // Transform to world space
    const wx = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
    const wy = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
    const wz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
    
    if (isInsideFrustumCPU(wx, wy, wz, frustumDir, halfAngle)) {
      validIndices.push(i);
    }
  }
  
  // Create filtered arrays
  const newCount = validIndices.length;
  const newPositions = new Float32Array(newCount * 3);
  const newColors = new Float32Array(newCount * 3);
  const newOpacities = new Float32Array(newCount);
  const newScales = new Float32Array(newCount * 3);
  const newRotations = new Float32Array(newCount * 4);
  
  for (let i = 0; i < newCount; i++) {
    const srcIdx = validIndices[i];
    
    newPositions[i * 3] = splatData.positions[srcIdx * 3];
    newPositions[i * 3 + 1] = splatData.positions[srcIdx * 3 + 1];
    newPositions[i * 3 + 2] = splatData.positions[srcIdx * 3 + 2];
    
    newColors[i * 3] = splatData.colors[srcIdx * 3];
    newColors[i * 3 + 1] = splatData.colors[srcIdx * 3 + 1];
    newColors[i * 3 + 2] = splatData.colors[srcIdx * 3 + 2];
    
    newOpacities[i] = splatData.opacities[srcIdx];
    
    newScales[i * 3] = splatData.scales[srcIdx * 3];
    newScales[i * 3 + 1] = splatData.scales[srcIdx * 3 + 1];
    newScales[i * 3 + 2] = splatData.scales[srcIdx * 3 + 2];
    
    newRotations[i * 4] = splatData.rotations[srcIdx * 4];
    newRotations[i * 4 + 1] = splatData.rotations[srcIdx * 4 + 1];
    newRotations[i * 4 + 2] = splatData.rotations[srcIdx * 4 + 2];
    newRotations[i * 4 + 3] = splatData.rotations[srcIdx * 4 + 3];
  }
  
  console.log(`Frustum culling (${face}): ${count} -> ${newCount} splats (${((1 - newCount/count) * 100).toFixed(1)}% culled)`);
  
  return {
    positions: newPositions,
    colors: newColors,
    opacities: newOpacities,
    scales: newScales,
    rotations: newRotations,
    count: newCount
  };
}

// Sort splats by depth (back to front) for correct alpha blending
function sortSplatsByDepth(splatData, cameraPos, modelMatrix) {
  const count = splatData.count;
  const positions = splatData.positions;
  
  // Create array of [index, depth] pairs
  const depths = new Float32Array(count);
  const indices = new Uint32Array(count);
  
  // Compute view-space depth for each splat
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    
    // Transform by model matrix and compute distance to camera
    const wx = modelMatrix[0] * x + modelMatrix[4] * y + modelMatrix[8] * z + modelMatrix[12];
    const wy = modelMatrix[1] * x + modelMatrix[5] * y + modelMatrix[9] * z + modelMatrix[13];
    const wz = modelMatrix[2] * x + modelMatrix[6] * y + modelMatrix[10] * z + modelMatrix[14];
    
    // Distance squared from camera (we sort by this, no need for sqrt)
    depths[i] = (wx - cameraPos.x) ** 2 + (wy - cameraPos.y) ** 2 + (wz - cameraPos.z) ** 2;
    indices[i] = i;
  }
  
  // Sort indices by depth (farthest first for back-to-front)
  indices.sort((a, b) => depths[b] - depths[a]);
  
  // Reorder all attribute arrays
  const newPositions = new Float32Array(count * 3);
  const newColors = new Float32Array(count * 3);
  const newOpacities = new Float32Array(count);
  const newScales = new Float32Array(count * 3);
  const newRotations = new Float32Array(count * 4);
  
  for (let i = 0; i < count; i++) {
    const srcIdx = indices[i];
    
    newPositions[i * 3] = splatData.positions[srcIdx * 3];
    newPositions[i * 3 + 1] = splatData.positions[srcIdx * 3 + 1];
    newPositions[i * 3 + 2] = splatData.positions[srcIdx * 3 + 2];
    
    newColors[i * 3] = splatData.colors[srcIdx * 3];
    newColors[i * 3 + 1] = splatData.colors[srcIdx * 3 + 1];
    newColors[i * 3 + 2] = splatData.colors[srcIdx * 3 + 2];
    
    newOpacities[i] = splatData.opacities[srcIdx];
    
    newScales[i * 3] = splatData.scales[srcIdx * 3];
    newScales[i * 3 + 1] = splatData.scales[srcIdx * 3 + 1];
    newScales[i * 3 + 2] = splatData.scales[srcIdx * 3 + 2];
    
    newRotations[i * 4] = splatData.rotations[srcIdx * 4];
    newRotations[i * 4 + 1] = splatData.rotations[srcIdx * 4 + 1];
    newRotations[i * 4 + 2] = splatData.rotations[srcIdx * 4 + 2];
    newRotations[i * 4 + 3] = splatData.rotations[srcIdx * 4 + 3];
  }
  
  return {
    positions: newPositions,
    colors: newColors,
    opacities: newOpacities,
    scales: newScales,
    rotations: newRotations,
    count
  };
}

// Cube face rotations (must match App.jsx)
const CUBE_FACE_ROTATIONS = {
  front:  [Math.PI, 0, 0],
  back:   [Math.PI, Math.PI, 0],
  left:   [Math.PI, -Math.PI / 2, 0],  // Swapped with right
  right:  [Math.PI, Math.PI / 2, 0],   // Swapped with left
  top:    [-Math.PI / 2, 0, 0],
  bottom: [Math.PI / 2, 0, 0],
};

// Cylinder 8 face rotations (8 faces around horizontal cylinder)
// Each face is rotated 45° from the previous, all at pitch=0
const CYLINDER_FACE_ROTATIONS = {
  n:  [Math.PI, 0, 0],                    // 0° - North (front)
  ne: [Math.PI, Math.PI / 4, 0],          // 45° - Northeast
  e:  [Math.PI, Math.PI / 2, 0],          // 90° - East (right)
  se: [Math.PI, 3 * Math.PI / 4, 0],      // 135° - Southeast
  s:  [Math.PI, Math.PI, 0],              // 180° - South (back)
  sw: [Math.PI, -3 * Math.PI / 4, 0],     // 225° / -135° - Southwest
  w:  [Math.PI, -Math.PI / 2, 0],         // 270° / -90° - West (left)
  nw: [Math.PI, -Math.PI / 4, 0],         // 315° / -45° - Northwest
};

// Pipeline type definitions
const PIPELINE_TYPES = {
  cubemap_6: {
    faces: ['front', 'back', 'left', 'right', 'top', 'bottom'],
    rotations: CUBE_FACE_ROTATIONS,
  },
  cylinder_8: {
    faces: ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'],
    rotations: CYLINDER_FACE_ROTATIONS,
  },
};

// Get face rotations for a pipeline type
function getFaceRotations(pipelineType) {
  return PIPELINE_TYPES[pipelineType]?.rotations || CUBE_FACE_ROTATIONS;
}

// Get face list for a pipeline type
function getFaceList(pipelineType) {
  return PIPELINE_TYPES[pipelineType]?.faces || PIPELINE_TYPES.cubemap_6.faces;
}

// Build model matrix for a face
function buildFaceMatrix(face, pipelineType = 'cubemap_6') {
  const rotations = getFaceRotations(pipelineType);
  const rotation = rotations[face];
  if (!rotation) {
    console.warn(`Unknown face "${face}" for pipeline "${pipelineType}"`);
    return new THREE.Matrix4();
  }
  const m = new THREE.Matrix4();
  const euler = new THREE.Euler(rotation[0], rotation[1], rotation[2]);
  m.makeRotationFromEuler(euler);
  // No X flip - keep correct orientation
  return m;
}

// Transform position by matrix
function transformPosition(x, y, z, matrix) {
  const e = matrix.elements;
  return {
    x: e[0] * x + e[4] * y + e[8] * z + e[12],
    y: e[1] * x + e[5] * y + e[9] * z + e[13],
    z: e[2] * x + e[6] * y + e[10] * z + e[14]
  };
}

// Transform rotation quaternion by matrix rotation
function transformRotation(qw, qx, qy, qz, matrix) {
  // Extract rotation quaternion from matrix
  const matQuat = new THREE.Quaternion();
  matQuat.setFromRotationMatrix(matrix);
  
  // Original quaternion
  const origQuat = new THREE.Quaternion(qx, qy, qz, qw);
  
  // Combine: matrix rotation * original rotation
  matQuat.multiply(origQuat);
  
  return { w: matQuat.w, x: matQuat.x, y: matQuat.y, z: matQuat.z };
}

/**
 * MergedGaussianSplats - Loads multiple faces, merges into one mesh, sorts by distance from origin
 * This ensures correct depth ordering for a viewer at center looking outward
 */
function MergedGaussianSplats({ basePath, enabledFaces, splatScale = 1.0, orientMode = 0, pipelineType = 'cubemap_6', faceDistance = 0, cullMode = 0 }) {
  const meshRef = useRef();
  const [mergedData, setMergedData] = useState(null);
  const { camera, size } = useThree();
  
  // Load and merge all enabled faces
  useEffect(() => {
    const loadAndMerge = async () => {
      const faceNames = Object.entries(enabledFaces)
        .filter(([_, enabled]) => enabled)
        .map(([face]) => face);
      
      if (faceNames.length === 0) {
        setMergedData(null);
        return;
      }
      
      console.log(`Loading faces for ${pipelineType}: ${faceNames.join(', ')}${cullMode === 1 ? ' (with CPU frustum culling)' : ''}`);
      
      // Load all PLY files in parallel
      const faceDataPromises = faceNames.map(async (face) => {
        try {
          const data = await loadPLY(`${basePath}input_${face}.ply`);
          return { face, data };
        } catch (e) {
          console.warn(`Failed to load ${face}:`, e);
          return null;
        }
      });
      
      const faceResults = (await Promise.all(faceDataPromises)).filter(Boolean);
      
      if (faceResults.length === 0) {
        setMergedData(null);
        return;
      }
      
      // Apply CPU frustum culling per face if enabled (cullMode === 1)
      let processedResults = faceResults;
      if (cullMode === 1) {
        processedResults = faceResults.map(({ face, data }) => {
          const matrix = buildFaceMatrix(face, pipelineType);
          const filteredData = filterByFrustum(data, face, matrix.elements);
          return { face, data: filteredData };
        });
      }
      
      // Count total splats
      const totalCount = processedResults.reduce((sum, { data }) => sum + data.count, 0);
      console.log(`Total splats across ${processedResults.length} faces: ${totalCount}`);
      
      // Allocate merged arrays
      const positions = new Float32Array(totalCount * 3);
      const colors = new Float32Array(totalCount * 3);
      const opacities = new Float32Array(totalCount);
      const scales = new Float32Array(totalCount * 3);
      const rotations = new Float32Array(totalCount * 4);
      const depths = new Float32Array(totalCount);
      
      // Merge all faces, transforming to world space
      let offset = 0;
      for (const { face, data } of processedResults) {
        const matrix = buildFaceMatrix(face, pipelineType);
        
        // Get face direction for distance offset (pushes face outward from center)
        const faceDir = getFrustumDirection(face);
        
        for (let i = 0; i < data.count; i++) {
          const idx = offset + i;
          
          // Transform position to world space
          const localPos = {
            x: data.positions[i * 3],
            y: data.positions[i * 3 + 1],
            z: data.positions[i * 3 + 2]
          };
          const worldPos = transformPosition(localPos.x, localPos.y, localPos.z, matrix);
          
          // Apply face distance offset (push face outward in its direction)
          const finalX = worldPos.x + faceDir.x * faceDistance;
          const finalY = worldPos.y + faceDir.y * faceDistance;
          const finalZ = worldPos.z + faceDir.z * faceDistance;
          
          positions[idx * 3] = finalX;
          positions[idx * 3 + 1] = finalY;
          positions[idx * 3 + 2] = finalZ;
          
          // Compute distance from origin for sorting (squared, no sqrt needed)
          depths[idx] = finalX * finalX + finalY * finalY + finalZ * finalZ;
          
          // Copy colors directly
          colors[idx * 3] = data.colors[i * 3];
          colors[idx * 3 + 1] = data.colors[i * 3 + 1];
          colors[idx * 3 + 2] = data.colors[i * 3 + 2];
          
          opacities[idx] = data.opacities[i];
          
          // Copy scales directly (they don't need transformation for uniform scaling)
          scales[idx * 3] = data.scales[i * 3];
          scales[idx * 3 + 1] = data.scales[i * 3 + 1];
          scales[idx * 3 + 2] = data.scales[i * 3 + 2];
          
          // Transform rotation to world space
          const localRot = {
            w: data.rotations[i * 4],
            x: data.rotations[i * 4 + 1],
            y: data.rotations[i * 4 + 2],
            z: data.rotations[i * 4 + 3]
          };
          const worldRot = transformRotation(localRot.w, localRot.x, localRot.y, localRot.z, matrix);
          
          rotations[idx * 4] = worldRot.w;
          rotations[idx * 4 + 1] = worldRot.x;
          rotations[idx * 4 + 2] = worldRot.y;
          rotations[idx * 4 + 3] = worldRot.z;
        }
        
        offset += data.count;
      }
      
      // Sort ALL splats by distance from origin (farthest first for back-to-front)
      console.log('Sorting all splats by distance from origin...');
      const indices = new Uint32Array(totalCount);
      for (let i = 0; i < totalCount; i++) indices[i] = i;
      
      // Sort indices by depth (farthest first)
      indices.sort((a, b) => depths[b] - depths[a]);
      
      // Reorder all arrays according to sorted indices
      const sortedPositions = new Float32Array(totalCount * 3);
      const sortedColors = new Float32Array(totalCount * 3);
      const sortedOpacities = new Float32Array(totalCount);
      const sortedScales = new Float32Array(totalCount * 3);
      const sortedRotations = new Float32Array(totalCount * 4);
      
      for (let i = 0; i < totalCount; i++) {
        const src = indices[i];
        
        sortedPositions[i * 3] = positions[src * 3];
        sortedPositions[i * 3 + 1] = positions[src * 3 + 1];
        sortedPositions[i * 3 + 2] = positions[src * 3 + 2];
        
        sortedColors[i * 3] = colors[src * 3];
        sortedColors[i * 3 + 1] = colors[src * 3 + 1];
        sortedColors[i * 3 + 2] = colors[src * 3 + 2];
        
        sortedOpacities[i] = opacities[src];
        
        sortedScales[i * 3] = scales[src * 3];
        sortedScales[i * 3 + 1] = scales[src * 3 + 1];
        sortedScales[i * 3 + 2] = scales[src * 3 + 2];
        
        sortedRotations[i * 4] = rotations[src * 4];
        sortedRotations[i * 4 + 1] = rotations[src * 4 + 1];
        sortedRotations[i * 4 + 2] = rotations[src * 4 + 2];
        sortedRotations[i * 4 + 3] = rotations[src * 4 + 3];
      }
      
      console.log('Merge and sort complete');
      
      setMergedData({
        positions: sortedPositions,
        colors: sortedColors,
        opacities: sortedOpacities,
        scales: sortedScales,
        rotations: sortedRotations,
        count: totalCount
      });
    };
    
    loadAndMerge();
  }, [basePath, enabledFaces, pipelineType, faceDistance, cullMode]);
  
  const { geometry, material } = useMemo(() => {
    if (!mergedData) return { geometry: null, material: null };
    
    // Create instanced geometry for quads
    const baseGeometry = new THREE.BufferGeometry();
    const quadVertices = new Float32Array([
      -1, -1, 0,
       1, -1, 0,
       1,  1, 0,
      -1, -1, 0,
       1,  1, 0,
      -1,  1, 0
    ]);
    baseGeometry.setAttribute('position', new THREE.BufferAttribute(quadVertices, 3));
    
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', baseGeometry.getAttribute('position'));
    
    // Instance attributes (already in world space, sorted by depth)
    geometry.setAttribute('splatCenter', new THREE.InstancedBufferAttribute(mergedData.positions, 3));
    geometry.setAttribute('splatColor', new THREE.InstancedBufferAttribute(mergedData.colors, 3));
    geometry.setAttribute('splatOpacity', new THREE.InstancedBufferAttribute(mergedData.opacities, 1));
    geometry.setAttribute('splatScale', new THREE.InstancedBufferAttribute(mergedData.scales, 3));
    geometry.setAttribute('splatRotation', new THREE.InstancedBufferAttribute(mergedData.rotations, 4));
    
    geometry.instanceCount = mergedData.count;
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    
    // Simplified shader - no frustum culling needed, positions are already in world space
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        viewport: { value: new THREE.Vector2(size.width, size.height) },
        focal: { value: new THREE.Vector2(size.height / 2, size.height / 2) },
        splatScaleMult: { value: splatScale },
        cullMode: { value: 0 },  // No culling for merged splats
        frustumDir: { value: new THREE.Vector3(0, 0, -1) },
        frustumAngle: { value: Math.PI / 4 },
        orientMode: { value: orientMode },
        cameraWorldPos: { value: new THREE.Vector3(0, 0, 0) }
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      side: THREE.DoubleSide
    });
    
    return { geometry, material };
  }, [mergedData, size, orientMode, splatScale]);
  
  useFrame(() => {
    if (material && camera) {
      material.uniforms.viewport.value.set(size.width, size.height);
      material.uniforms.splatScaleMult.value = splatScale;
      material.uniforms.orientMode.value = orientMode;
      material.uniforms.cameraWorldPos.value.copy(camera.position);
      
      const fovY = camera.fov * Math.PI / 180;
      const fy = size.height / (2 * Math.tan(fovY / 2));
      material.uniforms.focal.value.set(fy, fy);
    }
  });
  
  if (!geometry || !material) {
    return null;
  }
  
  // No rotation or scale on mesh - positions are already in world space
  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry} 
      material={material}
    />
  );
}

function GaussianSplatCloud({ url, rotation = [0, 0, 0], splatScale = 1.0, cullMode = 0, face = 'front', orientMode = 0 }) {
  const meshRef = useRef();
  const [sortedData, setSortedData] = useState(null);
  const { camera, size } = useThree();
  
  // Get frustum direction for this face
  const frustumDir = useMemo(() => getFrustumDirection(face), [face]);
  
  // Build model matrix from rotation props
  const modelMatrix = useMemo(() => {
    const m = new THREE.Matrix4();
    const euler = new THREE.Euler(rotation[0], rotation[1], rotation[2]);
    m.makeRotationFromEuler(euler);
    // No X flip - keep correct orientation
    return m.elements;
  }, [rotation]);
  
  // Load and sort once
  useEffect(() => {
    loadPLY(url).then(data => {
      let processedData = data;
      
      // Apply CPU frustum culling if mode is 1
      if (cullMode === 1) {
        processedData = filterByFrustum(data, face, modelMatrix);
      }
      
      // Sort based on initial camera position
      const sorted = sortSplatsByDepth(processedData, camera.position, modelMatrix);
      setSortedData(sorted);
    }).catch(console.error);
  }, [url, modelMatrix, cullMode, face]);
  
  const { geometry, material } = useMemo(() => {
    if (!sortedData) return { geometry: null, material: null };
    
    // Create instanced geometry for quads
    const baseGeometry = new THREE.BufferGeometry();
    
    // Quad vertices (2 triangles)
    const quadVertices = new Float32Array([
      -1, -1, 0,
       1, -1, 0,
       1,  1, 0,
      -1, -1, 0,
       1,  1, 0,
      -1,  1, 0
    ]);
    baseGeometry.setAttribute('position', new THREE.BufferAttribute(quadVertices, 3));
    
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', baseGeometry.getAttribute('position'));
    
    // Instance attributes (using sorted data)
    geometry.setAttribute('splatCenter', new THREE.InstancedBufferAttribute(sortedData.positions, 3));
    geometry.setAttribute('splatColor', new THREE.InstancedBufferAttribute(sortedData.colors, 3));
    geometry.setAttribute('splatOpacity', new THREE.InstancedBufferAttribute(sortedData.opacities, 1));
    geometry.setAttribute('splatScale', new THREE.InstancedBufferAttribute(sortedData.scales, 3));
    geometry.setAttribute('splatRotation', new THREE.InstancedBufferAttribute(sortedData.rotations, 4));
    
    geometry.instanceCount = sortedData.count;
    
    // Set bounding sphere for frustum culling
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        viewport: { value: new THREE.Vector2(size.width, size.height) },
        focal: { value: new THREE.Vector2(size.height / 2, size.height / 2) },
        splatScaleMult: { value: splatScale },
        cullMode: { value: cullMode },
        frustumDir: { value: frustumDir },
        frustumAngle: { value: Math.PI / 4 }, // 45° half-angle for 90° FOV
        orientMode: { value: orientMode },
        cameraWorldPos: { value: new THREE.Vector3(0, 0, 0) }
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      side: THREE.DoubleSide
    });
    
    return { geometry, material };
  }, [sortedData, size, cullMode, frustumDir, orientMode]);
  
  useFrame(() => {
    if (material && camera) {
      material.uniforms.viewport.value.set(size.width, size.height);
      material.uniforms.splatScaleMult.value = splatScale;
      material.uniforms.cullMode.value = cullMode;
      material.uniforms.frustumDir.value.copy(frustumDir);
      material.uniforms.orientMode.value = orientMode;
      material.uniforms.cameraWorldPos.value.copy(camera.position);
      
      // Compute focal length from camera
      const fovY = camera.fov * Math.PI / 180;
      const fy = size.height / (2 * Math.tan(fovY / 2));
      const fx = fy; // Assuming square pixels
      material.uniforms.focal.value.set(fx, fy);
    }
  });
  
  if (!geometry || !material) {
    return null;
  }
  
  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry} 
      material={material} 
      rotation={rotation}
    />
  );
}

export default GaussianSplatCloud;
export { MergedGaussianSplats, PIPELINE_TYPES, getFaceList, getFaceRotations };
