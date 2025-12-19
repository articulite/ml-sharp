import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Visualizes a 90-degree pyramidal frustum from origin.
 * The frustum extends in the -Z direction (Three.js convention: camera looks down -Z)
 * 
 * @param {Array} direction - Direction the frustum faces: 'front', 'back', 'left', 'right', 'top', 'bottom'
 * @param {number} depth - How far the frustum extends
 * @param {boolean} visible - Whether to show the visualization
 */
function FrustumVisualizer({ direction = 'front', depth = 2.0, visible = true }) {
  const geometry = useMemo(() => {
    // 90-degree FOV means at distance d, the half-width is also d (tan(45°) = 1)
    // So at depth d, the frustum spans from -d to +d in both X and Y
    const d = depth;
    
    // Vertices: apex at origin, base is a square at z = -depth
    // For a frustum facing -Z:
    //   apex: (0, 0, 0)
    //   base corners: (±d, ±d, -d)
    const vertices = new Float32Array([
      // Apex
      0, 0, 0,
      // Base corners (at z = -depth)
      -d, -d, -d,  // bottom-left
       d, -d, -d,  // bottom-right
       d,  d, -d,  // top-right
      -d,  d, -d,  // top-left
    ]);
    
    // Indices for 4 triangular faces + 2 triangles for the base (optional)
    const indices = new Uint16Array([
      // Left face (apex, bottom-left, top-left)
      0, 1, 4,
      // Bottom face (apex, bottom-right, bottom-left)
      0, 2, 1,
      // Right face (apex, top-right, bottom-right)
      0, 3, 2,
      // Top face (apex, top-left, top-right)
      0, 4, 3,
      // Base (two triangles)
      1, 2, 3,
      1, 3, 4,
    ]);
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    
    return geo;
  }, [depth]);
  
  // Rotation to orient the frustum based on cube face direction
  const rotation = useMemo(() => {
    switch (direction) {
      case 'front':  return [0, 0, 0];                    // -Z
      case 'back':   return [0, Math.PI, 0];              // +Z
      case 'left':   return [0, -Math.PI / 2, 0];         // -X
      case 'right':  return [0, Math.PI / 2, 0];          // +X
      case 'top':    return [Math.PI / 2, 0, 0];          // +Y
      case 'bottom': return [-Math.PI / 2, 0, 0];         // -Y
      default:       return [0, 0, 0];
    }
  }, [direction]);
  
  if (!visible) return null;
  
  return (
    <mesh geometry={geometry} rotation={rotation}>
      <meshBasicMaterial 
        color={0x00ff00}
        transparent={true}
        opacity={0.15}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

/**
 * Wireframe edges for the frustum (for better visibility)
 */
function FrustumWireframe({ direction = 'front', depth = 2.0, visible = true }) {
  const geometry = useMemo(() => {
    const d = depth;
    
    // Line segments for edges
    const vertices = new Float32Array([
      // From apex to each corner
      0, 0, 0,  -d, -d, -d,
      0, 0, 0,   d, -d, -d,
      0, 0, 0,   d,  d, -d,
      0, 0, 0,  -d,  d, -d,
      // Base square
      -d, -d, -d,   d, -d, -d,
       d, -d, -d,   d,  d, -d,
       d,  d, -d,  -d,  d, -d,
      -d,  d, -d,  -d, -d, -d,
    ]);
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    return geo;
  }, [depth]);
  
  const rotation = useMemo(() => {
    switch (direction) {
      case 'front':  return [0, 0, 0];
      case 'back':   return [0, Math.PI, 0];
      case 'left':   return [0, -Math.PI / 2, 0];
      case 'right':  return [0, Math.PI / 2, 0];
      case 'top':    return [Math.PI / 2, 0, 0];
      case 'bottom': return [-Math.PI / 2, 0, 0];
      default:       return [0, 0, 0];
    }
  }, [direction]);
  
  if (!visible) return null;
  
  return (
    <lineSegments geometry={geometry} rotation={rotation}>
      <lineBasicMaterial color={0x00ff00} linewidth={2} />
    </lineSegments>
  );
}

export { FrustumVisualizer, FrustumWireframe };

