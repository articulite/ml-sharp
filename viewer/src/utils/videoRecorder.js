/**
 * Video Recorder for 360° Sphere Viewer
 * 
 * Strategy: Capture at native resolution, scale to 720p.
 * Uses chunked recording with context reset to avoid Chrome's memory bug.
 * 
 * Fixed specs: 60fps, 4s, 720p output
 */

import { Muxer, ArrayBufferTarget } from 'mp4-muxer';

const CONFIG = Object.freeze({
  fps: 60,
  duration: 4,
  width: 1280,
  height: 720,
  bitrate: 6_000_000,
  orbitRadiusH: Math.PI * 0.25,
  orbitRadiusV: Math.PI * 0.10,
  fov: 90,
  chunkSize: 100,
});

const TOTAL_FRAMES = CONFIG.fps * CONFIG.duration;

function calculateOrbitPosition(progress) {
  const angle = progress * Math.PI * 2 + Math.PI / 2;
  return {
    theta: Math.PI + Math.sin(angle) * CONFIG.orbitRadiusH,
    phi: Math.PI / 2 + Math.cos(angle) * CONFIG.orbitRadiusV,
  };
}

function sphericalToCartesian(theta, phi) {
  const sinPhi = Math.sin(phi);
  return {
    x: 50 * sinPhi * Math.sin(theta),
    y: 50 * Math.cos(phi),
    z: 50 * sinPhi * Math.cos(theta),
  };
}

export function createVideoRecorder() {
  let isRecording = false;
  let isCancelled = false;
  let frameCount = 0;
  let onProgress = null;
  let onComplete = null;
  let originalFov = null;
  let cameraRef = null;
  let controlsRef = null;

  function cleanup() {
    if (cameraRef && originalFov !== null) {
      cameraRef.fov = originalFov;
      cameraRef.updateProjectionMatrix();
    }
    
    if (controlsRef) {
      controlsRef.enabled = true;
    }
    
    originalFov = null;
    cameraRef = null;
    controlsRef = null;
    isRecording = false;
  }

  return {
    async start(canvas, controls, camera, renderFn, renderer) {
      if (isRecording) return;
      
      isRecording = true;
      isCancelled = false;
      frameCount = 0;
      cameraRef = camera;
      controlsRef = controls;
      
      if (controls) {
        controls.enabled = false;
        controls.autoRotate = false;
      }
      
      if (camera) {
        originalFov = camera.fov;
        camera.fov = CONFIG.fov;
        camera.updateProjectionMatrix();
      }
      
      // Get actual canvas dimensions (might be different due to pixel ratio)
      const srcWidth = canvas.width;
      const srcHeight = canvas.height;
      
      console.log(`Recording: ${CONFIG.fps}fps, ${CONFIG.duration}s`);
      console.log(`Source: ${srcWidth}x${srcHeight} → Output: ${CONFIG.width}x${CONFIG.height}`);
      
      // Create scaling canvas
      const scaleCanvas = document.createElement('canvas');
      scaleCanvas.width = CONFIG.width;
      scaleCanvas.height = CONFIG.height;
      const scaleCtx = scaleCanvas.getContext('2d', { alpha: false });
      
      // Setup encoder
      const muxer = new Muxer({
        target: new ArrayBufferTarget(),
        video: {
          codec: 'avc',
          width: CONFIG.width,
          height: CONFIG.height,
        },
        fastStart: 'in-memory',
      });
      
      const encoder = new VideoEncoder({
        output: (chunk, metadata) => muxer.addVideoChunk(chunk, metadata),
        error: (e) => {
          console.error('Encoder error:', e);
          isCancelled = true;
        },
      });
      
      encoder.configure({
        codec: 'avc1.42001f',
        width: CONFIG.width,
        height: CONFIG.height,
        bitrate: CONFIG.bitrate,
        framerate: CONFIG.fps,
        hardwareAcceleration: 'prefer-software',
      });
      
      const frameDurationUs = Math.round(1_000_000 / CONFIG.fps);
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      // Buffers for pixel reading at SOURCE resolution
      const pixelBuffer = new Uint8Array(srcWidth * srcHeight * 4);
      const rowSize = srcWidth * 4;
      const tempRow = new Uint8Array(rowSize);
      
      // Temp canvas for flipping (at source resolution)
      const flipCanvas = document.createElement('canvas');
      flipCanvas.width = srcWidth;
      flipCanvas.height = srcHeight;
      const flipCtx = flipCanvas.getContext('2d', { alpha: false });
      
      // Process in chunks
      for (let chunk = 0; chunk < Math.ceil(TOTAL_FRAMES / CONFIG.chunkSize) && !isCancelled; chunk++) {
        const chunkStart = chunk * CONFIG.chunkSize;
        const chunkEnd = Math.min(chunkStart + CONFIG.chunkSize, TOTAL_FRAMES);
        
        console.log(`Chunk ${chunk + 1}: frames ${chunkStart}-${chunkEnd - 1}`);
        
        for (let i = chunkStart; i < chunkEnd && !isCancelled; i++) {
          // Backpressure
          while (encoder.encodeQueueSize > 2) {
            await new Promise(r => setTimeout(r, 5));
          }
          
          // Update camera
          const progress = i / TOTAL_FRAMES;
          const { theta, phi } = calculateOrbitPosition(progress);
          const target = sphericalToCartesian(theta, phi);
          camera.lookAt(target.x, target.y, target.z);
          
          // Render
          renderFn();
          gl.finish();
          
          // Read pixels at FULL source resolution
          gl.readPixels(0, 0, srcWidth, srcHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);
          
          // Flip vertically (WebGL is bottom-up)
          const halfHeight = Math.floor(srcHeight / 2);
          for (let y = 0; y < halfHeight; y++) {
            const topOffset = y * rowSize;
            const bottomOffset = (srcHeight - 1 - y) * rowSize;
            tempRow.set(pixelBuffer.subarray(topOffset, topOffset + rowSize));
            pixelBuffer.set(pixelBuffer.subarray(bottomOffset, bottomOffset + rowSize), topOffset);
            pixelBuffer.set(tempRow, bottomOffset);
          }
          
          // Put to flip canvas
          const imageData = new ImageData(new Uint8ClampedArray(pixelBuffer.buffer), srcWidth, srcHeight);
          flipCtx.putImageData(imageData, 0, 0);
          
          // Scale to output resolution
          scaleCtx.drawImage(flipCanvas, 0, 0, CONFIG.width, CONFIG.height);
          
          // Encode from scaled canvas
          const frame = new VideoFrame(scaleCanvas, {
            timestamp: i * frameDurationUs,
            duration: frameDurationUs,
          });
          
          encoder.encode(frame, { keyFrame: i % CONFIG.fps === 0 });
          frame.close();
          
          frameCount = i + 1;
          onProgress?.((frameCount / TOTAL_FRAMES) * 100, frameCount, TOTAL_FRAMES);
        }
        
        // Between chunks: reset context
        if (chunk < Math.ceil(TOTAL_FRAMES / CONFIG.chunkSize) - 1 && !isCancelled) {
          console.log('Resetting WebGL context...');
          const loseContext = gl.getExtension('WEBGL_lose_context');
          if (loseContext) {
            loseContext.loseContext();
            await new Promise(r => setTimeout(r, 100));
            loseContext.restoreContext();
            await new Promise(r => setTimeout(r, 500));
            renderFn();
            await new Promise(r => requestAnimationFrame(r));
          }
        }
      }
      
      if (isCancelled) {
        try { encoder.close(); } catch {}
        cleanup();
        return;
      }
      
      await encoder.flush();
      encoder.close();
      
      muxer.finalize();
      const { buffer } = muxer.target;
      const blob = new Blob([buffer], { type: 'video/mp4' });
      
      console.log(`Video ready: ${(blob.size / 1024 / 1024).toFixed(2)} MB`);
      
      cleanup();
      onComplete?.(blob);
    },
    
    cancel() {
      isCancelled = true;
      cleanup();
    },
    
    onProgress(callback) {
      onProgress = callback;
      return this;
    },
    
    onComplete(callback) {
      onComplete = callback;
      return this;
    },
    
    get isRecording() {
      return isRecording;
    },
    
    get progress() {
      return (frameCount / TOTAL_FRAMES) * 100;
    },
    
    get config() {
      return CONFIG;
    },
  };
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export { CONFIG as VIDEO_CONFIG };
