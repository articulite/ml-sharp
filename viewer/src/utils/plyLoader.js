/**
 * PLY loader for Gaussian Splat binary files
 */

export async function loadPLY(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  
  // Parse header
  const textDecoder = new TextDecoder();
  const headerEnd = findHeaderEnd(buffer);
  const headerText = textDecoder.decode(new Uint8Array(buffer, 0, headerEnd));
  
  const header = parseHeader(headerText);
  const dataOffset = headerEnd + 1; // +1 for newline after end_header
  
  // Parse vertex data
  const vertexCount = header.vertexCount;
  const vertexSize = 14 * 4; // 14 floats per vertex
  
  const dataView = new DataView(buffer, dataOffset);
  
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);
  const scales = new Float32Array(vertexCount * 3);
  const rotations = new Float32Array(vertexCount * 4);
  
  for (let i = 0; i < vertexCount; i++) {
    const offset = i * vertexSize;
    
    // Position (x, y, z) - flip Y to correct for coordinate system
    positions[i * 3 + 0] = dataView.getFloat32(offset + 0, true);
    positions[i * 3 + 1] = -dataView.getFloat32(offset + 4, true);
    positions[i * 3 + 2] = dataView.getFloat32(offset + 8, true);
    
    // Color from spherical harmonics DC component (f_dc_0, f_dc_1, f_dc_2)
    // Convert from SH to RGB: color = 0.5 + SH_C0 * f_dc
    const SH_C0 = 0.28209479177387814;
    colors[i * 3 + 0] = Math.max(0, Math.min(1, 0.5 + SH_C0 * dataView.getFloat32(offset + 12, true)));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * dataView.getFloat32(offset + 16, true)));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * dataView.getFloat32(offset + 20, true)));
    
    // Opacity (sigmoid applied)
    const rawOpacity = dataView.getFloat32(offset + 24, true);
    opacities[i] = 1 / (1 + Math.exp(-rawOpacity));
    
    // Scale (exp applied)
    scales[i * 3 + 0] = Math.exp(dataView.getFloat32(offset + 28, true));
    scales[i * 3 + 1] = Math.exp(dataView.getFloat32(offset + 32, true));
    scales[i * 3 + 2] = Math.exp(dataView.getFloat32(offset + 36, true));
    
    // Rotation quaternion (w, x, y, z) - normalize and flip Y axis
    const qw = dataView.getFloat32(offset + 40, true);
    const qx = dataView.getFloat32(offset + 44, true);
    const qy = dataView.getFloat32(offset + 48, true);
    const qz = dataView.getFloat32(offset + 52, true);
    const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    // Flip Y axis: negate x and z components of quaternion
    rotations[i * 4 + 0] = qw / qlen;
    rotations[i * 4 + 1] = -qx / qlen;
    rotations[i * 4 + 2] = qy / qlen;
    rotations[i * 4 + 3] = -qz / qlen;
  }
  
  return {
    positions,
    colors,
    opacities,
    scales,
    rotations,
    count: vertexCount
  };
}

function findHeaderEnd(buffer) {
  const view = new Uint8Array(buffer);
  const endHeader = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // "end_header"
  
  for (let i = 0; i < Math.min(view.length, 4096); i++) {
    let match = true;
    for (let j = 0; j < endHeader.length; j++) {
      if (view[i + j] !== endHeader[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      return i + endHeader.length;
    }
  }
  throw new Error('Could not find end_header in PLY file');
}

function parseHeader(headerText) {
  const lines = headerText.split('\n');
  let vertexCount = 0;
  
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'element' && parts[1] === 'vertex') {
      vertexCount = parseInt(parts[2], 10);
    }
  }
  
  return { vertexCount };
}

