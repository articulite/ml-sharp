import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Suspense, useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import GaussianSplatCloud, { MergedGaussianSplats, PIPELINE_TYPES, getFaceList, getFaceRotations } from './components/GaussianSplats';
import { FrustumVisualizer, FrustumWireframe } from './components/FrustumVisualizer';
import GenerateDashboard from './components/GenerateDashboard';
import { createVideoRecorder, downloadBlob, VIDEO_CONFIG } from './utils/videoRecorder';
import './App.css';

// Get face rotations from shared definitions
const CUBE_FACE_ROTATIONS = getFaceRotations('cubemap_6');

const DEFAULT_SPLAT_PATH = '/splats/';
const DEFAULT_PIPELINE_TYPE = 'cubemap_6';

// Culling mode descriptions
const CULL_MODES = [
  { id: 0, name: 'None', desc: 'Render all gaussians' },
  { id: 1, name: 'CPU Pre-filter', desc: 'Filter on load (fastest runtime)' },
  { id: 2, name: 'GPU Vertex', desc: 'Discard in vertex shader' },
  { id: 3, name: 'GPU Fragment', desc: 'Discard in fragment shader' },
];

// Camera reset component
function CameraReset({ trigger, controlsRef }) {
  const { camera } = useThree();
  
  useEffect(() => {
    if (trigger > 0) {
      camera.position.set(0, 0, 0);
      if (controlsRef?.current) {
        controlsRef.current.target.set(0, 0, -0.001);
        controlsRef.current.minDistance = 0;
        controlsRef.current.update();
      }
    }
  }, [trigger, camera, controlsRef]);
  
  return null;
}

// Video recorder bridge - provides access to Three.js internals for recording
function VideoRecorderBridge({ recorderRef, controlsRef }) {
  const { gl, camera, scene } = useThree();
  
  useEffect(() => {
    if (recorderRef) {
      recorderRef.current = {
        camera,
        canvas: gl.domElement,
        controls: controlsRef?.current,
        render: () => gl.render(scene, camera),
        gl,
        scene,
      };
    }
  }, [gl, camera, scene, controlsRef, recorderRef]);
  
  return null;
}

// Camera FOV controller
function CameraFov({ fov }) {
  const { camera } = useThree();
  
  useEffect(() => {
    if (camera.fov !== fov) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  }, [fov, camera]);
  
  return null;
}

// Parallax animation - moves camera in tight circle while maintaining look direction
function ParallaxAnimation({ enabled, radius = 0.05, speed = 0.5, controlsRef }) {
  const { camera } = useThree();
  const prevOffset = useRef(new THREE.Vector3());
  
  useFrame(({ clock }) => {
    if (!enabled) {
      // Remove any remaining offset when disabled
      if (prevOffset.current.lengthSq() > 0) {
        camera.position.sub(prevOffset.current);
        if (controlsRef?.current) {
          controlsRef.current.target.sub(prevOffset.current);
          controlsRef.current.update();
        }
        prevOffset.current.set(0, 0, 0);
      }
      return;
    }
    
    const t = clock.getElapsedTime() * speed;
    const newOffset = new THREE.Vector3(
      Math.cos(t) * radius,
      Math.sin(t) * radius,
      0
    );
    
    // Remove previous offset, apply new offset
    camera.position.sub(prevOffset.current).add(newOffset);
    
    if (controlsRef?.current) {
      controlsRef.current.target.sub(prevOffset.current).add(newOffset);
    }
    
    prevOffset.current.copy(newOffset);
  });
  
  return null;
}

// WASD + QE camera controls
function WASDControls({ speed = 0.05, controlsRef }) {
  const { camera } = useThree();
  const keys = useRef({});
  
  useEffect(() => {
    const handleKeyDown = (e) => {
      keys.current[e.code] = true;
    };
    const handleKeyUp = (e) => {
      keys.current[e.code] = false;
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);
  
  useFrame(() => {
    const direction = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    
    // Get camera's forward and right vectors
    camera.getWorldDirection(direction);
    right.crossVectors(direction, up).normalize();
    
    let moved = false;
    const moveVector = new THREE.Vector3();
    
    // Forward/Back (W/S)
    if (keys.current['KeyW']) {
      moveVector.add(direction.clone().multiplyScalar(speed));
      moved = true;
    }
    if (keys.current['KeyS']) {
      moveVector.add(direction.clone().multiplyScalar(-speed));
      moved = true;
    }
    
    // Left/Right (A/D)
    if (keys.current['KeyA']) {
      moveVector.add(right.clone().multiplyScalar(-speed));
      moved = true;
    }
    if (keys.current['KeyD']) {
      moveVector.add(right.clone().multiplyScalar(speed));
      moved = true;
    }
    
    // Up/Down (Q/E)
    if (keys.current['KeyQ']) {
      moveVector.y -= speed;
      moved = true;
    }
    if (keys.current['KeyE']) {
      moveVector.y += speed;
      moved = true;
    }
    
    if (moved) {
      camera.position.add(moveVector);
      // Also move the OrbitControls target so we strafe instead of orbit
      if (controlsRef?.current) {
        controlsRef.current.target.add(moveVector);
      }
    }
  });
  
  return null;
}

function LoadingIndicator() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshBasicMaterial color="#4a9eff" wireframe />
    </mesh>
  );
}

function SplatViewer({ refreshTrigger, splatBasePath = DEFAULT_SPLAT_PATH, pipelineType = DEFAULT_PIPELINE_TYPE, fullscreenMode = false }) {
  const [availableFaces, setAvailableFaces] = useState([]);
  const [enabledFaces, setEnabledFaces] = useState({});
  const [loading, setLoading] = useState(true);
  const [splatScale, setSplatScale] = useState(0.6);  // 0.6 default scale
  const [cullMode, setCullMode] = useState(1);  // CPU Pre-filter by default
  const [showFrustums, setShowFrustums] = useState(false);
  const [frustumDepth, setFrustumDepth] = useState(2.0);
  const [panelOpen, setPanelOpen] = useState(true);
  const [cameraResetTrigger, setCameraResetTrigger] = useState(0);
  const [parallaxEnabled, setParallaxEnabled] = useState(true);  // Parallax on by default
  const [orientMode, setOrientMode] = useState(1);  // 0=stored, 1=to center, 2=billboard - default to center
  const [useMergedSplats, setUseMergedSplats] = useState(true);  // Use merged by default for correct depth
  const [viewFov, setViewFov] = useState(60);  // 60° FOV default
  const [faceDistance, setFaceDistance] = useState(0);  // Distance to push faces outward from center
  const [gaussianDropRate, setGaussianDropRate] = useState(0);  // 0% drop by default
  const [useHarmonized, setUseHarmonized] = useState(false);  // Depth harmonization toggle
  const [harmonizedAvailable, setHarmonizedAvailable] = useState(false);  // Track if harmonized files exist
  const [isHarmonizing, setIsHarmonizing] = useState(false);  // Harmonization in progress
  const controlsRef = useRef();
  const threeBridgeRef = useRef(null);
  
  // Video recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const videoRecorderRef = useRef(null);
  
  // Start video recording (fixed: 60fps, 4s, 720p for optimal performance)
  const startRecording = useCallback(() => {
    if (!threeBridgeRef.current || isRecording) return;
    
    const recorder = createVideoRecorder(); // Uses optimized fixed config
    videoRecorderRef.current = recorder;
    
    // Disable parallax during recording
    setParallaxEnabled(false);
    
    recorder
      .onProgress((percent) => {
        setRecordingProgress(percent);
      })
      .onComplete((blob) => {
        setIsRecording(false);
        setRecordingProgress(0);
        // Re-enable controls
        if (controlsRef.current) {
          controlsRef.current.enabled = true;
        }
        // Download the video
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const extension = blob.type.includes('mp4') ? 'mp4' : 'webm';
        downloadBlob(blob, `sharp-360-${timestamp}.${extension}`);
      });
    
    setIsRecording(true);
    
    const { canvas, controls, camera, gl, scene } = threeBridgeRef.current;
    recorder.start(canvas, controls, camera, () => {
      gl.render(scene, camera);
    });
  }, [isRecording]);
  
  // Cancel recording
  const cancelRecording = useCallback(() => {
    if (videoRecorderRef.current) {
      videoRecorderRef.current.cancel();
      videoRecorderRef.current = null;
    }
    setIsRecording(false);
    setRecordingProgress(0);
    // Re-enable controls
    if (controlsRef.current) {
      controlsRef.current.enabled = true;
    }
  }, []);

  // Get the face list for the current pipeline type
  const expectedFaces = getFaceList(pipelineType);
  const faceRotations = getFaceRotations(pipelineType);

  // Compute effective splat path based on harmonization toggle
  const effectiveSplatPath = useHarmonized 
    ? splatBasePath.replace('/splats/', '/splats_harmonized/')
    : splatBasePath;

  const checkFaces = async (basePath, faceList) => {
    const available = [];
    const enabled = {};

    for (const face of faceList) {
      try {
        const response = await fetch(`${basePath}input_${face}.ply`, { method: 'HEAD' });
        if (response.ok) {
          available.push(face);
          // Default: enable all faces except top and bottom (for cubemap only)
          if (pipelineType === 'cubemap_6') {
            enabled[face] = (face !== 'top' && face !== 'bottom');
          } else {
            enabled[face] = true;
          }
        }
      } catch {
        // File not available
      }
    }

    setAvailableFaces(available);
    setEnabledFaces(enabled);
    setLoading(false);
  };

  // Check if harmonized files exist
  const checkHarmonizedAvailable = async (basePath) => {
    const harmonizedPath = basePath.replace('/splats/', '/splats_harmonized/');
    try {
      const response = await fetch(`${harmonizedPath}input_front.ply`, { method: 'HEAD' });
      setHarmonizedAvailable(response.ok);
    } catch {
      setHarmonizedAvailable(false);
    }
  };

  // Run harmonization on current job
  const runHarmonization = async () => {
    // Extract job_id from splatBasePath (e.g., /generated/job_xxx/splats/)
    const match = splatBasePath.match(/\/generated\/(job_[^/]+)\//);
    if (!match) {
      alert('Harmonization only works for generated jobs');
      return;
    }
    
    const jobId = match[1];
    setIsHarmonizing(true);
    
    try {
      const response = await fetch(`/api/jobs/${jobId}/harmonize`, { method: 'POST' });
      const data = await response.json();
      
      if (data.success) {
        setHarmonizedAvailable(true);
        setUseHarmonized(true);
        // Trigger refresh to load new files
        checkFaces(effectiveSplatPath.replace('/splats/', '/splats_harmonized/'), expectedFaces);
      } else {
        alert(`Harmonization failed: ${data.error}`);
      }
    } catch (e) {
      alert(`Harmonization error: ${e.message}`);
    }
    
    setIsHarmonizing(false);
  };

  useEffect(() => {
    checkFaces(effectiveSplatPath, expectedFaces);
    checkHarmonizedAvailable(splatBasePath);
  }, [pipelineType, useHarmonized]);

  // Re-check faces when refreshTrigger, splatBasePath, or pipelineType changes
  useEffect(() => {
    if (refreshTrigger > 0) {
      setLoading(true);
      checkFaces(effectiveSplatPath, expectedFaces);
      checkHarmonizedAvailable(splatBasePath);
    }
  }, [refreshTrigger, splatBasePath, pipelineType, useHarmonized]);

  const toggleFace = (face) => {
    setEnabledFaces(prev => ({ ...prev, [face]: !prev[face] }));
  };

  // Key includes cullMode, orientMode, and basePath so components reinitialize when they change
  const splatKey = (face) => `${face}-cull-${cullMode}-orient-${orientMode}-${refreshTrigger}-${splatBasePath}`;

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <p>Detecting available splat faces...</p>
      </div>
    );
  }

  return (
    <div className="viewer-container">
      {!fullscreenMode && (
      <div className={`controls-panel ${panelOpen ? 'open' : 'closed'}`}>
        <button className="panel-toggle" onClick={() => setPanelOpen(!panelOpen)}>
          {panelOpen ? '◀' : '▶'}
        </button>
        
        {panelOpen && (
          <>
            <h2>{pipelineType === 'cylinder_8' ? 'Cylinder Faces' : 'Cube Faces'}</h2>
            <p className="pipeline-type-label">
              {pipelineType === 'cylinder_8' ? '8-face cylinder' : '6-face cubemap'}
            </p>
            <div className="face-toggles">
              {availableFaces.map(face => (
                <label key={face} className="face-toggle">
                  <input
                    type="checkbox"
                    checked={enabledFaces[face] || false}
                    onChange={() => toggleFace(face)}
                  />
                  <span className="toggle-label">{face.toUpperCase()}</span>
                </label>
              ))}
            </div>
            {availableFaces.length === 0 && (
              <p className="no-faces-message">No splat files found</p>
            )}
            
            <div className="scale-slider">
              <label>
                <span>Splat Scale: {splatScale.toFixed(1)}</span>
                <input
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={splatScale}
                  onChange={(e) => setSplatScale(parseFloat(e.target.value))}
                />
              </label>
            </div>

            <div className="fov-slider">
              <label>
                <span>View FOV: {viewFov}°</span>
                <input
                  type="range"
                  min="30"
                  max="120"
                  step="5"
                  value={viewFov}
                  onChange={(e) => setViewFov(parseInt(e.target.value))}
                />
              </label>
            </div>

            {pipelineType === 'cylinder_8' && (
              <div className="face-distance-slider">
                <label>
                  <span>Face Distance: {faceDistance.toFixed(2)}</span>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.05"
                    value={faceDistance}
                    onChange={(e) => setFaceDistance(parseFloat(e.target.value))}
                  />
                </label>
                <p className="slider-hint">Push faces outward from center</p>
              </div>
            )}

            <h2>Orientation</h2>
            <div className="orient-mode-selector">
              <label className="orient-mode-option">
                <input
                  type="radio"
                  name="orientMode"
                  checked={orientMode === 0}
                  onChange={() => setOrientMode(0)}
                />
                <span className="orient-mode-name">Stored</span>
              </label>
              <label className="orient-mode-option">
                <input
                  type="radio"
                  name="orientMode"
                  checked={orientMode === 1}
                  onChange={() => setOrientMode(1)}
                />
                <span className="orient-mode-name">To Center</span>
              </label>
              <label className="orient-mode-option">
                <input
                  type="radio"
                  name="orientMode"
                  checked={orientMode === 2}
                  onChange={() => setOrientMode(2)}
                />
                <span className="orient-mode-name">Billboard</span>
              </label>
            </div>
            <p className="orient-hint">Billboard = always face camera (fixes side-view stretching)</p>

            <div className="merge-toggle">
              <label className="face-toggle">
                <input
                  type="checkbox"
                  checked={useMergedSplats}
                  onChange={(e) => setUseMergedSplats(e.target.checked)}
                />
                <span className="toggle-label">Merged Depth Sort</span>
              </label>
              <p className="merge-hint">Sort all splats globally by distance from origin</p>
            </div>

            <h2>Depth Harmonization</h2>
            <div className="harmonize-controls">
              <label className="face-toggle">
                <input
                  type="checkbox"
                  checked={useHarmonized}
                  onChange={(e) => setUseHarmonized(e.target.checked)}
                  disabled={!harmonizedAvailable}
                />
                <span className="toggle-label">
                  Use Harmonized
                  {harmonizedAvailable ? ' ✓' : ' (not available)'}
                </span>
              </label>
              <p className="harmonize-hint">
                Corrects depth scale mismatches between cube faces
              </p>
              {!harmonizedAvailable && splatBasePath.includes('/generated/') && (
                <button 
                  className="harmonize-btn"
                  onClick={runHarmonization}
                  disabled={isHarmonizing}
                >
                  {isHarmonizing ? 'Harmonizing...' : 'Run Harmonization'}
                </button>
              )}
              {harmonizedAvailable && useHarmonized && (
                <p className="harmonize-active">✨ Viewing harmonized splats</p>
              )}
            </div>

            <h2>Performance</h2>
            <div className="gaussian-drop-slider">
              <label>
                <span>Drop Gaussians: {gaussianDropRate}%</span>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="10"
                  value={gaussianDropRate}
                  onChange={(e) => setGaussianDropRate(parseInt(e.target.value))}
                />
              </label>
              <p className="slider-hint">Randomly removes gaussians to improve performance</p>
            </div>

            <h2>Frustum Culling</h2>
            <div className="cull-mode-selector">
              {CULL_MODES.map(mode => (
                <label key={mode.id} className="cull-mode-option">
                  <input
                    type="radio"
                    name="cullMode"
                    checked={cullMode === mode.id}
                    onChange={() => setCullMode(mode.id)}
                  />
                  <div className="cull-mode-info">
                    <span className="cull-mode-name">{mode.name}</span>
                  </div>
                </label>
              ))}
            </div>

            <div className="frustum-controls">
              <label className="frustum-toggle">
                <input
                  type="checkbox"
                  checked={showFrustums}
                  onChange={(e) => setShowFrustums(e.target.checked)}
                />
                <span>Show Frustums</span>
              </label>
              
              <label className="frustum-depth-slider">
                <span>Depth: {frustumDepth.toFixed(1)}</span>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.1"
                  value={frustumDepth}
                  onChange={(e) => setFrustumDepth(parseFloat(e.target.value))}
                />
              </label>
            </div>

            <button 
              className="reset-camera-btn"
              onClick={() => setCameraResetTrigger(prev => prev + 1)}
            >
              Reset Camera to Origin
            </button>

            <button 
              className={`parallax-btn ${parallaxEnabled ? 'active' : ''}`}
              onClick={() => setParallaxEnabled(prev => !prev)}
            >
              {parallaxEnabled ? 'Stop Parallax' : 'Start Parallax'}
            </button>

            <h2>Video Recording</h2>
            <div className="recording-info">
              <p className="recording-specs">
                {VIDEO_CONFIG.fps}fps • {VIDEO_CONFIG.duration}s • {VIDEO_CONFIG.width}×{VIDEO_CONFIG.height}
              </p>
              <p className="recording-hint">
                Optimized H.264/MP4 export with circular orbit path
              </p>
            </div>
          </>
        )}
      </div>
      )}

      {/* Recording overlay */}
      {isRecording && (
        <div className="recording-overlay">
          <div className="recording-indicator">
            <span className="recording-dot"></span>
            <span>Recording...</span>
          </div>
          <div className="recording-progress-bar">
            <div 
              className="recording-progress-fill" 
              style={{ width: `${recordingProgress}%` }}
            />
          </div>
          <span className="recording-percent">{Math.round(recordingProgress)}%</span>
          <button className="cancel-recording-btn" onClick={cancelRecording}>
            Cancel
          </button>
        </div>
      )}

      {/* Record button (floating) */}
      {!isRecording && (
        <button 
          className="record-btn"
          onClick={startRecording}
          title="Record 360° orbit video"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="8" />
          </svg>
        </button>
      )}

      <Canvas
        camera={{ position: [0, 0, 0], fov: 45 }}
        gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
        className="splat-canvas"
      >
        <color attach="background" args={['#0a0a0f']} />
        <ambientLight intensity={0.5} />
        
        <Suspense fallback={<LoadingIndicator />}>
          {useMergedSplats ? (
            // Single merged mesh with all faces - correct depth sorting
            <MergedGaussianSplats
              key={`merged-${refreshTrigger}-${effectiveSplatPath}-${orientMode}-${pipelineType}-${faceDistance}-${cullMode}-drop${gaussianDropRate}-harm${useHarmonized}`}
              basePath={effectiveSplatPath}
              enabledFaces={enabledFaces}
              splatScale={splatScale}
              orientMode={orientMode}
              pipelineType={pipelineType}
              faceDistance={faceDistance}
              cullMode={cullMode}
              dropRate={gaussianDropRate / 100}
            />
          ) : (
            // Separate meshes per face (may have depth issues)
            availableFaces.map(face => (
              enabledFaces[face] && (
                <GaussianSplatCloud
                  key={splatKey(face)}
                  url={`${effectiveSplatPath}input_${face}.ply`}
                  rotation={faceRotations[face]}
                  splatScale={splatScale}
                  cullMode={cullMode}
                  face={face}
                  orientMode={orientMode}
                />
              )
            ))
          )}
        </Suspense>

        {/* Frustum visualizations for enabled faces */}
        {showFrustums && availableFaces.map(face => (
          enabledFaces[face] && (
            <group key={`frustum-${face}`}>
              <FrustumVisualizer 
                direction={face} 
                depth={frustumDepth} 
                visible={true} 
              />
              <FrustumWireframe 
                direction={face} 
                depth={frustumDepth} 
                visible={true} 
              />
            </group>
          )
        ))}
        
        <WASDControls speed={0.05} controlsRef={controlsRef} />
        <CameraReset trigger={cameraResetTrigger} controlsRef={controlsRef} />
        <CameraFov fov={viewFov} />
        <ParallaxAnimation enabled={parallaxEnabled && !isRecording} controlsRef={controlsRef} />
        <VideoRecorderBridge recorderRef={threeBridgeRef} controlsRef={controlsRef} />
        <OrbitControls 
          ref={controlsRef}
          enableDamping 
          dampingFactor={0.05}
          minDistance={0}
          maxDistance={50}
          target={[0, 0, -0.001]}
        />
      </Canvas>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('generate');
  const [splatRefreshTrigger, setSplatRefreshTrigger] = useState(0);
  const [splatBasePath, setSplatBasePath] = useState('/splats/');
  const [currentPipelineType, setCurrentPipelineType] = useState(DEFAULT_PIPELINE_TYPE);
  const [fullscreenMode, setFullscreenMode] = useState(false);

  // F11 fullscreen mode - hides navbar and viewer UI + browser fullscreen
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'F11') {
        e.preventDefault();
        
        if (!document.fullscreenElement) {
          // Enter fullscreen
          document.documentElement.requestFullscreen().catch(() => {});
          setFullscreenMode(true);
        } else {
          // Exit fullscreen
          document.exitFullscreen().catch(() => {});
          setFullscreenMode(false);
        }
      }
    };
    
    // Sync state when user exits fullscreen via Escape key
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        setFullscreenMode(false);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const handleSplatsGenerated = (splatsData) => {
    // If splatsData contains a job path, use that; otherwise use default
    if (splatsData?.jobId) {
      setSplatBasePath(`/generated/${splatsData.jobId}/splats/`);
    } else {
      setSplatBasePath('/splats/');
    }
    // Update pipeline type if provided
    if (splatsData?.pipelineType) {
      setCurrentPipelineType(splatsData.pipelineType);
    } else {
      setCurrentPipelineType(DEFAULT_PIPELINE_TYPE);
    }
    // Trigger refresh of the viewer when new splats are generated
    setSplatRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className={`app ${fullscreenMode ? 'fullscreen-mode' : ''}`}>
      {!fullscreenMode && (
      <header className="app-header">
        <h1>Sharp Viewer</h1>
        <nav className="app-tabs">
          <button 
            className={`tab-btn ${activeTab === 'generate' ? 'active' : ''}`}
            onClick={() => setActiveTab('generate')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            Generate
          </button>
          <button 
            className={`tab-btn ${activeTab === 'viewer' ? 'active' : ''}`}
            onClick={() => setActiveTab('viewer')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8-10-8-10-8z" />
            </svg>
            Viewer
          </button>
        </nav>
      </header>
      )}
      <main className="app-main">
        {activeTab === 'generate' && (
          <GenerateDashboard onSplatsGenerated={handleSplatsGenerated} />
        )}
        {activeTab === 'viewer' && (
          <SplatViewer 
            refreshTrigger={splatRefreshTrigger} 
            splatBasePath={splatBasePath} 
            pipelineType={currentPipelineType}
            fullscreenMode={fullscreenMode}
          />
        )}
      </main>
    </div>
  );
}

export default App;
