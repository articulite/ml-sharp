import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Suspense, useState, useEffect } from 'react';
import GaussianSplatCloud from './components/GaussianSplats';
import './App.css';

// Cube face orientations (Euler angles in radians)
const CUBE_FACE_ROTATIONS = {
  front:  [0, 0, 0],
  back:   [0, Math.PI, 0],
  left:   [0, Math.PI / 2, 0],
  right:  [0, -Math.PI / 2, 0],
  top:    [-Math.PI / 2, 0, 0],
  bottom: [Math.PI / 2, 0, 0],
};

const SPLAT_BASE_PATH = '/splats/';

function LoadingIndicator() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshBasicMaterial color="#4a9eff" wireframe />
    </mesh>
  );
}

function SplatViewer() {
  const [availableFaces, setAvailableFaces] = useState([]);
  const [enabledFaces, setEnabledFaces] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check which splat files are available
    const checkFaces = async () => {
      const faces = ['front', 'back', 'left', 'right', 'top', 'bottom'];
      const available = [];
      const enabled = {};

      for (const face of faces) {
        try {
          const response = await fetch(`${SPLAT_BASE_PATH}input_${face}.ply`, { method: 'HEAD' });
          if (response.ok) {
            available.push(face);
            enabled[face] = true;
          }
        } catch {
          // File not available
        }
      }

      setAvailableFaces(available);
      setEnabledFaces(enabled);
      setLoading(false);
    };

    checkFaces();
  }, []);

  const toggleFace = (face) => {
    setEnabledFaces(prev => ({ ...prev, [face]: !prev[face] }));
  };

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
      <div className="controls-panel">
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
                key={face}
                url={`${SPLAT_BASE_PATH}input_${face}.ply`}
                rotation={CUBE_FACE_ROTATIONS[face]}
              />
            )
          ))}
        </Suspense>
        
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05}
          minDistance={0.5}
          maxDistance={20}
        />
      </Canvas>
    </div>
  );
}

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Gaussian Splat Viewer</h1>
      </header>
      <SplatViewer />
    </div>
  );
}

export default App;
