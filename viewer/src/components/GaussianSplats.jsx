import { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { loadPLY } from '../utils/plyLoader';

const vertexShader = `
  precision highp float;
  
  attribute vec3 splatCenter;
  attribute vec3 splatColor;
  attribute float splatOpacity;
  attribute vec3 splatScale;
  attribute vec4 splatRotation;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vPosition;
  
  uniform vec2 viewport;
  uniform vec2 focal;
  
  mat3 quatToMat3(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
    );
  }
  
  void main() {
    vColor = splatColor;
    vOpacity = splatOpacity;
    vPosition = position.xy;
    
    // Transform center to view space
    vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
    
    // Skip splats behind camera
    if (viewCenter.z > 0.0) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
    
    // Build covariance matrix from rotation and scale
    mat3 R = quatToMat3(splatRotation);
    mat3 S = mat3(
      splatScale.x, 0.0, 0.0,
      0.0, splatScale.y, 0.0,
      0.0, 0.0, splatScale.z
    );
    mat3 M = R * S;
    mat3 Sigma = M * transpose(M);
    
    // Transform to view space
    mat3 viewRot = mat3(modelViewMatrix);
    mat3 Sigma_view = viewRot * Sigma * transpose(viewRot);
    
    // Project to 2D
    float z2 = viewCenter.z * viewCenter.z;
    mat2 J = mat2(
      focal.x / viewCenter.z, 0.0,
      0.0, focal.y / viewCenter.z
    );
    
    mat2 cov2D = J * mat2(Sigma_view[0][0], Sigma_view[0][1], 
                          Sigma_view[1][0], Sigma_view[1][1]) * transpose(J);
    
    // Add low-pass filter
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;
    
    // Compute eigenvalues for extent
    float a = cov2D[0][0];
    float b = cov2D[0][1];
    float c = cov2D[1][1];
    float det = a * c - b * b;
    float trace = a + c;
    float discriminant = max(0.0, trace * trace / 4.0 - det);
    float sqrtDisc = sqrt(discriminant);
    float lambda1 = trace / 2.0 + sqrtDisc;
    float lambda2 = trace / 2.0 - sqrtDisc;
    
    // Splat radius (3 sigma)
    float radius = 3.0 * sqrt(max(lambda1, lambda2));
    
    // Compute eigenvectors for orientation
    vec2 v1;
    if (abs(b) > 0.0001) {
      v1 = normalize(vec2(lambda1 - c, b));
    } else {
      v1 = vec2(1.0, 0.0);
    }
    vec2 v2 = vec2(-v1.y, v1.x);
    
    float s1 = 3.0 * sqrt(max(lambda1, 0.0001));
    float s2 = 3.0 * sqrt(max(lambda2, 0.0001));
    
    // Transform quad vertex
    vec2 offset = v1 * position.x * s1 + v2 * position.y * s2;
    
    // Project center to screen
    vec4 projCenter = projectionMatrix * viewCenter;
    vec2 screenCenter = projCenter.xy / projCenter.w;
    
    // Add offset in screen space
    vec2 screenOffset = offset / viewport * 2.0;
    
    gl_Position = vec4(screenCenter + screenOffset, projCenter.z / projCenter.w, 1.0);
  }
`;

const fragmentShader = `
  precision highp float;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vPosition;
  
  void main() {
    // Gaussian falloff
    float d = dot(vPosition, vPosition);
    if (d > 1.0) discard;
    
    float alpha = exp(-0.5 * d * 9.0) * vOpacity;
    if (alpha < 0.01) discard;
    
    gl_FragColor = vec4(vColor, alpha);
  }
`;

function GaussianSplatCloud({ url, rotation = [0, 0, 0] }) {
  const meshRef = useRef();
  const [splatData, setSplatData] = useState(null);
  const { camera, size } = useThree();
  
  useEffect(() => {
    loadPLY(url).then(setSplatData).catch(console.error);
  }, [url]);
  
  const { geometry, material } = useMemo(() => {
    if (!splatData) return { geometry: null, material: null };
    
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
    geometry.index = baseGeometry.index;
    geometry.attributes.position = baseGeometry.attributes.position;
    
    // Instance attributes
    geometry.setAttribute('splatCenter', new THREE.InstancedBufferAttribute(splatData.positions, 3));
    geometry.setAttribute('splatColor', new THREE.InstancedBufferAttribute(splatData.colors, 3));
    geometry.setAttribute('splatOpacity', new THREE.InstancedBufferAttribute(splatData.opacities, 1));
    geometry.setAttribute('splatScale', new THREE.InstancedBufferAttribute(splatData.scales, 3));
    geometry.setAttribute('splatRotation', new THREE.InstancedBufferAttribute(splatData.rotations, 4));
    
    geometry.instanceCount = splatData.count;
    
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        viewport: { value: new THREE.Vector2(size.width, size.height) },
        focal: { value: new THREE.Vector2(size.width, size.height) }
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.SrcAlphaFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor
    });
    
    return { geometry, material };
  }, [splatData, size]);
  
  useFrame(() => {
    if (material && camera) {
      material.uniforms.viewport.value.set(size.width, size.height);
      
      // Compute focal length from camera
      const fovY = camera.fov * Math.PI / 180;
      const fy = size.height / (2 * Math.tan(fovY / 2));
      const fx = fy * camera.aspect;
      material.uniforms.focal.value.set(fx, fy);
    }
  });
  
  if (!geometry || !material) {
    return null;
  }
  
  return (
    <mesh ref={meshRef} geometry={geometry} material={material} rotation={rotation} />
  );
}

export default GaussianSplatCloud;

