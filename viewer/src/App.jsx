import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Suspense, useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import GaussianSplatCloud from './components/GaussianSplats';
import { FrustumVisualizer, FrustumWireframe } from './components/FrustumVisualizer';
import GenerateDashboard from './components/GenerateDashboard';
import './App.css';

// Cube face orientations (Euler angles in radians)
// Base 180° X rotation is baked in to correct coordinate system
const CUBE_FACE_ROTATIONS = {
  front:  [Math.PI, 0, 0],
  back:   [Math.PI, Math.PI, 0],
  left:   [Math.PI, Math.PI / 2, 0],
  right:  [Math.PI, -Math.PI / 2, 0],
  top:    [Math.PI / 2, 0, 0],
  bottom: [-Math.PI / 2, 0, 0],
};

const SPLAT_BASE_PATH = '/splats/';

// Culling mode descriptions
const CULL_MODES = [
  { id: 0, name: 'None', desc: 'Render all gaussians' },
  { id: 1, name: 'CPU Pre-filter', desc: 'Filter on load (fastest runtime)' },
  { id: 2, name: 'GPU Vertex', desc: 'Discard in vertex shader' },
  { id: 3, name: 'GPU Fragment', desc: 'Discard in fragment shader' },
];

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

function SplatViewer({ refreshTrigger }) {
  const [availableFaces, setAvailableFaces] = useState([]);
  const [enabledFaces, setEnabledFaces] = useState({});
  const [loading, setLoading] = useState(true);
  const [splatScale, setSplatScale] = useState(1.0);
  const [cullMode, setCullMode] = useState(0);
  const [showFrustums, setShowFrustums] = useState(true);
  const [frustumDepth, setFrustumDepth] = useState(2.0);
  const [panelOpen, setPanelOpen] = useState(true);
  const controlsRef = useRef();

  const checkFaces = async () => {
    const faces = ['front', 'back', 'left', 'right', 'top', 'bottom'];
    const available = [];
    const enabled = {};

    for (const face of faces) {
      try {
        const response = await fetch(`${SPLAT_BASE_PATH}input_${face}.ply`, { method: 'HEAD' });
        if (response.ok) {
          available.push(face);
          // Default: only enable 'front' face on initial load
          enabled[face] = (face === 'front');
        }
      } catch {
        // File not available
      }
    }

    setAvailableFaces(available);
    setEnabledFaces(enabled);
    setLoading(false);
  };

  useEffect(() => {
    checkFaces();
  }, []);

  // Re-check faces when refreshTrigger changes (after generation)
  useEffect(() => {
    if (refreshTrigger > 0) {
      setLoading(true);
      checkFaces();
    }
  }, [refreshTrigger]);

  const toggleFace = (face) => {
    setEnabledFaces(prev => ({ ...prev, [face]: !prev[face] }));
  };

  // Key includes cullMode so components reinitialize when culling mode changes
  const splatKey = (face) => `${face}-cull-${cullMode}-${refreshTrigger}`;

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
      <div className={`controls-panel ${panelOpen ? 'open' : 'closed'}`}>
        <button className="panel-toggle" onClick={() => setPanelOpen(!panelOpen)}>
          {panelOpen ? '◀' : '▶'}
        </button>
        
        {panelOpen && (
          <>
            <h2>Cube Faces</h2>
            <div className="face-toggles">
              {availableFaces.map(face => (
                <label key={face} className="face-toggle">
                  <input
                    type="checkbox"
                    checked={enabledFaces[face] || false}
                    onChange={() => toggleFace(face)}
                  />
                  <span className="toggle-label">{face}</span>
                </label>
              ))}
            </div>
            {availableFaces.length === 0 && (
              <p className="no-faces-message">No splat files found in /splats/</p>
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
          </>
        )}
      </div>

      <Canvas
        camera={{ position: [0, 0, 3], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
        className="splat-canvas"
      >
        <color attach="background" args={['#0a0a0f']} />
        <ambientLight intensity={0.5} />
        
        <Suspense fallback={<LoadingIndicator />}>
          {availableFaces.map(face => (
            enabledFaces[face] && (
              <GaussianSplatCloud
                key={splatKey(face)}
                url={`${SPLAT_BASE_PATH}input_${face}.ply`}
                rotation={CUBE_FACE_ROTATIONS[face]}
                splatScale={splatScale}
                cullMode={cullMode}
                face={face}
              />
            )
          ))}
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
        <OrbitControls 
          ref={controlsRef}
          enableDamping 
          dampingFactor={0.05}
          minDistance={0.1}
          maxDistance={50}
        />
      </Canvas>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('generate');
  const [splatRefreshTrigger, setSplatRefreshTrigger] = useState(0);

  const handleSplatsGenerated = () => {
    // Trigger refresh of the viewer when new splats are generated
    setSplatRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="app">
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
      <main className="app-main">
        {activeTab === 'generate' && (
          <GenerateDashboard onSplatsGenerated={handleSplatsGenerated} />
        )}
        {activeTab === 'viewer' && (
          <SplatViewer refreshTrigger={splatRefreshTrigger} />
        )}
      </main>
    </div>
  );
}

export default App;
