/**
 * Video Recorder for 360° Sphere Viewer
 * 
 * Fixed specs: 60fps, 4s, 720p output
 * 
 * ## Chrome WebGL Canvas Read Memory Leak Bug
 * 
 * Chrome has a bug where reading from a WebGL canvas (via readPixels, drawImage,
 * createImageBitmap, captureStream, toBlob, etc.) accumulates GPU memory that
 * is never released. After ~128 reads, Chrome exhausts GPU memory and loses
 * the WebGL context (CONTEXT_LOST_WEBGL).
 * 
 * This affects ALL canvas read methods - the leak is in Chrome's compositor,
 * not in JavaScript code. Setting preserveDrawingBuffer doesn't help.
 * 
 * ## Workaround: Chunked Recording with Context Reset
 * 
 * The fix is to deliberately lose and restore the WebGL context every 100 frames
 * using the WEBGL_lose_context extension. This forces Chrome to release the
 * accumulated GPU resources.
 * 
 * Flow:
 * 1. Record 100 frames
 * 2. Call loseContext() - forces Chrome to release GPU resources
 * 3. Wait 100ms
 * 4. Call restoreContext() - Three.js automatically restores scene state
 * 5. Wait 500ms for restoration
 * 6. Continue with next chunk
 * 
 * The video encoder maintains state across context resets, so the output
 * is seamless.
 * 
 * ## Other Implementation Notes
 * 
 * - Captures at native canvas resolution, then scales to 720p
 * - Uses gl.readPixels() for synchronous GPU readback
 * - Flips pixels vertically (WebGL is bottom-up)
 * - Software H.264 encoding to avoid GPU contention
 * - Backpressure on encoder queue to prevent memory buildup
 */

import { Muxer, ArrayBufferTarget } from 'mp4-muxer';

const CONFIG = Object.freeze({
  fps: 60,
  duration: 6,
  width: 1280,
  height: 720,
  bitrate: 12_000_000,
  // Original spherical orbit + subtle horizontal parallax
  orbitRadiusH: Math.PI * 0.15,  // Horizontal look sweep (reduced from 0.25)
  orbitRadiusV: Math.PI * 0.06,  // Vertical look sweep (reduced from 0.10)
  parallaxShift: 0.1,            // Subtle left-right position shift (reduced from 0.15)
  fov: 75,
  chunkSize: 60,
});

const TOTAL_FRAMES = CONFIG.fps * CONFIG.duration;

/**
 * Original spherical orbit (O-shaped look path) + subtle horizontal parallax.
 * Camera looks in a smooth loop, with subtle left-right position shift.
 */
function calculateOrbitPosition(progress) {
  const angle = progress * Math.PI * 2 + Math.PI / 2;
  const phiCenter = Math.PI / 2;
  const thetaCenter = Math.PI;
  
  return {
    theta: thetaCenter + Math.sin(angle) * CONFIG.orbitRadiusH,
    phi: phiCenter + Math.cos(angle) * CONFIG.orbitRadiusV,
  };
}

function sphericalToCartesian(theta, phi, radius = 50) {
  return {
    x: radius * Math.sin(phi) * Math.sin(theta),
    y: radius * Math.cos(phi),
    z: radius * Math.sin(phi) * Math.cos(theta),
  };
}

function calculateCameraOrbit(progress, originalPosition) {
  // Subtle left-right position shift for parallax (sin loops seamlessly)
  const parallaxX = Math.sin(progress * Math.PI * 2) * CONFIG.parallaxShift;
  
  const cameraPos = {
    x: originalPosition.x + parallaxX,
    y: originalPosition.y,
    z: originalPosition.z,
  };
  
  // Original spherical orbit for look direction
  const { theta, phi } = calculateOrbitPosition(progress);
  const lookTarget = sphericalToCartesian(theta, phi);
  
  return { cameraPos, lookTarget };
}

export function createVideoRecorder() {
  let isRecording = false;
  let isCancelled = false;
  let frameCount = 0;
  let onProgress = null;
  let onComplete = null;
  let originalFov = null;
  let originalCameraPos = null;
  let cameraRef = null;
  let controlsRef = null;

  function cleanup() {
    if (cameraRef) {
      if (originalFov !== null) {
        cameraRef.fov = originalFov;
        cameraRef.updateProjectionMatrix();
      }
      if (originalCameraPos) {
        cameraRef.position.set(originalCameraPos.x, originalCameraPos.y, originalCameraPos.z);
        // Restore original forward look direction
        cameraRef.lookAt(
          originalCameraPos.x,
          originalCameraPos.y,
          originalCameraPos.z + CONFIG.lookDistance
        );
      }
    }
    
    if (controlsRef) {
      controlsRef.enabled = true;
    }
    
    originalFov = null;
    originalCameraPos = null;
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
        originalCameraPos = {
          x: camera.position.x,
          y: camera.position.y,
          z: camera.position.z,
        };
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
      
      // Try HEVC first, fall back to AVC
      const hevcCodec = 'hvc1.1.6.L93.B0';
      const avcCodec = 'avc1.42001f';
      
      let useHevc = false;
      try {
        const hevcSupport = await VideoEncoder.isConfigSupported({
          codec: hevcCodec,
          width: CONFIG.width,
          height: CONFIG.height,
          bitrate: CONFIG.bitrate,
          framerate: CONFIG.fps,
        });
        useHevc = hevcSupport.supported;
      } catch (e) {
        useHevc = false;
      }
      
      const codecString = useHevc ? hevcCodec : avcCodec;
      console.log(`Using codec: ${useHevc ? 'HEVC (H.265)' : 'AVC (H.264)'}`);
      
      // Setup encoder
      const muxer = new Muxer({
        target: new ArrayBufferTarget(),
        video: {
          codec: useHevc ? 'hevc' : 'avc',
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
        codec: codecString,
        width: CONFIG.width,
        height: CONFIG.height,
        bitrate: CONFIG.bitrate,
        framerate: CONFIG.fps,
        hardwareAcceleration: 'prefer-hardware',  // Use GPU encoder (MediaCodec)
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
          
          // Update camera for carousel orbit (facing outward)
          const progress = i / TOTAL_FRAMES;
          const { cameraPos, lookTarget } = calculateCameraOrbit(progress, originalCameraPos);
          camera.position.set(cameraPos.x, cameraPos.y, cameraPos.z);
          camera.lookAt(lookTarget.x, lookTarget.y, lookTarget.z);
          
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
          
          // Encode from scaled canvas - use explicit visibleRect for hardware encoder
          const frame = new VideoFrame(scaleCanvas, {
            timestamp: i * frameDurationUs,
            duration: frameDurationUs,
            visibleRect: { x: 0, y: 0, width: CONFIG.width, height: CONFIG.height },
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
