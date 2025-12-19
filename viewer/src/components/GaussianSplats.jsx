import { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { loadPLY } from '../utils/plyLoader';

// Simpler vertex shader - billboard quads facing camera
const vertexShader = `
  precision highp float;
  
  attribute vec3 splatCenter;
  attribute vec3 splatColor;
  attribute float splatOpacity;
  attribute vec3 splatScale;
  attribute vec4 splatRotation;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vUV;
  
  uniform vec2 viewport;
  uniform float focal;
  uniform float splatScaleMult;
  
  void main() {
    vColor = splatColor;
    vOpacity = splatOpacity;
    vUV = position.xy;
    
    // Transform center to view space
    vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
    
    // Skip splats behind camera
    if (viewCenter.z > -0.1) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
    
    // Use max scale for splat size to ensure proper coverage
    float maxScale = max(splatScale.x, max(splatScale.y, splatScale.z));
    
    // Project size to screen space
    float projScale = focal * maxScale * splatScaleMult / (-viewCenter.z);
    
    // Clamp to reasonable size
    projScale = clamp(projScale, 2.0, 2048.0);
    
    // Billboard offset in screen space
    vec2 offset = position.xy * projScale;
    
    // Project center
    vec4 clipPos = projectionMatrix * viewCenter;
    
    // Add offset in clip space
    clipPos.xy += offset * clipPos.w / viewport;
    
    gl_Position = clipPos;
  }
`;

const fragmentShader = `
  precision highp float;
  
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vUV;
  
  void main() {
    // Gaussian falloff
    float d2 = dot(vUV, vUV);
    if (d2 > 1.0) discard;
    
    float alpha = exp(-d2 * 4.0) * vOpacity;
    if (alpha < 0.005) discard;
    
    gl_FragColor = vec4(vColor, alpha);
  }
`;

function GaussianSplatCloud({ url, rotation = [0, 0, 0], splatScale = 6.0 }) {
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
    geometry.setAttribute('position', baseGeometry.getAttribute('position'));
    
    // Instance attributes
    geometry.setAttribute('splatCenter', new THREE.InstancedBufferAttribute(splatData.positions, 3));
    geometry.setAttribute('splatColor', new THREE.InstancedBufferAttribute(splatData.colors, 3));
    geometry.setAttribute('splatOpacity', new THREE.InstancedBufferAttribute(splatData.opacities, 1));
    geometry.setAttribute('splatScale', new THREE.InstancedBufferAttribute(splatData.scales, 3));
    geometry.setAttribute('splatRotation', new THREE.InstancedBufferAttribute(splatData.rotations, 4));
    
    geometry.instanceCount = splatData.count;
    
    // Set bounding sphere for frustum culling
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        viewport: { value: new THREE.Vector2(size.width, size.height) },
        focal: { value: size.height / 2 },
        splatScaleMult: { value: splatScale }
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide
    });
    
    return { geometry, material };
  }, [splatData, size]);
  
  useFrame(() => {
    if (material && camera) {
      material.uniforms.viewport.value.set(size.width, size.height);
      material.uniforms.splatScaleMult.value = splatScale;
      
      // Compute focal length from camera
      const fovY = camera.fov * Math.PI / 180;
      const fy = size.height / (2 * Math.tan(fovY / 2));
      material.uniforms.focal.value = fy;
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
      scale={[-1, 1, 1]}
    />
  );
}

export default GaussianSplatCloud;
