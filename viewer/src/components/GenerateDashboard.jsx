import { useState, useRef, useCallback, useEffect } from 'react';
import './GenerateDashboard.css';

// In development, Vite proxies /api and /ws to the backend
// In production, these would need to be configured appropriately
const API_BASE = '';
const WS_BASE = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

const CUBE_FACES = ['front', 'back', 'left', 'right', 'top', 'bottom'];

function generateJobId() {
  return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function ProgressBar({ percent, status, message }) {
  const getStatusColor = () => {
    switch (status) {
      case 'completed': return 'var(--accent-emerald)';
      case 'error': return 'var(--accent-coral)';
      case 'running': return 'var(--accent-cyan)';
      default: return 'var(--text-muted)';
    }
  };

  return (
    <div className="progress-container">
      <div className="progress-header">
        <span className="progress-status" style={{ color: getStatusColor() }}>
          {status === 'running' ? '◉' : status === 'completed' ? '✓' : status === 'error' ? '✗' : '○'} {status}
        </span>
        <span className="progress-percent">{percent.toFixed(0)}%</span>
      </div>
      <div className="progress-bar-track">
        <div 
          className="progress-bar-fill" 
          style={{ 
            width: `${percent}%`,
            background: `linear-gradient(90deg, ${getStatusColor()}, ${getStatusColor()}88)`
          }}
        />
      </div>
      <div className="progress-message">{message}</div>
    </div>
  );
}

function ImagePreview({ url, label }) {
  return (
    <div className="image-preview">
      <img src={url} alt={label} />
      <span className="image-label">{label}</span>
    </div>
  );
}

function GenerateDashboard({ onSplatsGenerated }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fov, setFov] = useState(110);
  const [outputSize, setOutputSize] = useState(0);
  const [selectedFaces, setSelectedFaces] = useState(
    CUBE_FACES.reduce((acc, face) => ({ ...acc, [face]: true }), {})
  );
  const [generateSplats, setGenerateSplats] = useState(true);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, status: 'idle', message: '' });
  const [results, setResults] = useState(null);
  const [serverOnline, setServerOnline] = useState(false);
  
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);

  // Check server health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        setServerOnline(response.ok);
      } catch {
        setServerOnline(false);
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback((e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(selectedFile);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      setResults(null);
      
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(droppedFile);
    }
  }, []);

  const toggleFace = (face) => {
    setSelectedFaces(prev => ({ ...prev, [face]: !prev[face] }));
  };

  const selectAllFaces = () => {
    setSelectedFaces(CUBE_FACES.reduce((acc, face) => ({ ...acc, [face]: true }), {}));
  };

  const selectNoFaces = () => {
    setSelectedFaces(CUBE_FACES.reduce((acc, face) => ({ ...acc, [face]: false }), {}));
  };

  const handleProcess = async () => {
    if (!file) return;
    
    const jobId = generateJobId();
    setIsProcessing(true);
    setProgress({ percent: 0, status: 'running', message: 'Starting pipeline...' });
    setResults(null);
    
    // Connect WebSocket for progress updates
    const ws = new WebSocket(`${WS_BASE}/ws/${jobId}`);
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setProgress({
          percent: data.percent,
          status: data.status,
          message: data.message,
        });
        
        if (data.status === 'completed' || data.status === 'error') {
          setIsProcessing(false);
          if (data.results) {
            setResults(data.results);
            if (data.results.splats && onSplatsGenerated) {
              onSplatsGenerated(data.results.splats);
            }
          }
          ws.close();
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
    
    ws.onerror = () => {
      setProgress({ percent: 0, status: 'error', message: 'WebSocket connection failed' });
      setIsProcessing(false);
    };
    
    ws.onopen = async () => {
      // Send the processing request
      const formData = new FormData();
      formData.append('file', file);
      formData.append('fov', fov.toString());
      formData.append('output_size', outputSize.toString());
      formData.append('faces', Object.entries(selectedFaces)
        .filter(([_, enabled]) => enabled)
        .map(([face]) => face)
        .join(',') || 'all');
      formData.append('generate_splats', generateSplats.toString());
      formData.append('job_id', jobId);
      
      try {
        const response = await fetch(`${API_BASE}/api/process`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const error = await response.text();
          setProgress({ percent: 0, status: 'error', message: `Request failed: ${error}` });
          setIsProcessing(false);
        }
      } catch (error) {
        setProgress({ percent: 0, status: 'error', message: `Network error: ${error.message}` });
        setIsProcessing(false);
      }
    };
  };

  const selectedFaceCount = Object.values(selectedFaces).filter(Boolean).length;

  return (
    <div className="generate-dashboard">
      <div className="dashboard-header">
        <div className="header-content">
          <h2>Generate Cubemap & Splats</h2>
          <p className="header-subtitle">Convert equirectangular panoramas to cube faces and Gaussian splats</p>
        </div>
        <div className={`server-status ${serverOnline ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {serverOnline ? 'Server Online' : 'Server Offline'}
        </div>
      </div>

      <div className="dashboard-content">
        <div className="dashboard-left">
          {/* Upload Section */}
          <div className="section upload-section">
            <h3>Input Image</h3>
            <div 
              className={`drop-zone ${preview ? 'has-image' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
            >
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="upload-preview" />
                  <div className="preview-overlay">
                    <span>Click to change</span>
                  </div>
                </div>
              ) : (
                <div className="drop-prompt">
                  <div className="drop-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                    </svg>
                  </div>
                  <span className="drop-text">Drop equirectangular image here</span>
                  <span className="drop-subtext">or click to browse</span>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </div>
            {file && (
              <div className="file-info">
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
          </div>

          {/* Progress Section */}
          {(isProcessing || progress.status !== 'idle') && (
            <div className="section progress-section">
              <h3>Progress</h3>
              <ProgressBar {...progress} />
            </div>
          )}

          {/* Results Section */}
          {results && (
            <div className="section results-section">
              <h3>Generated Cube Faces</h3>
              <div className="results-grid">
                {results.cubefaces?.faces?.map((face) => (
                  <ImagePreview 
                    key={face.name} 
                    url={`${API_BASE}${face.url}`} 
                    label={face.name}
                  />
                ))}
              </div>
              
              {results.cubefaces?.metadata && (
                <div className="metadata-panel">
                  <div className="metadata-item">
                    <span>Source Size</span>
                    <strong>{results.cubefaces.metadata.source_size}</strong>
                  </div>
                  <div className="metadata-item">
                    <span>Face Size</span>
                    <strong>{results.cubefaces.metadata.face_size}</strong>
                  </div>
                  <div className="metadata-item">
                    <span>FOV</span>
                    <strong>{results.cubefaces.metadata.fov}°</strong>
                  </div>
                  <div className="metadata-item">
                    <span>Focal Length</span>
                    <strong>{results.cubefaces.metadata.focal_length_px}px</strong>
                  </div>
                </div>
              )}

              {results.splats?.splats?.length > 0 && (
                <>
                  <h3 className="splats-header">Generated Splats</h3>
                  <div className="splats-list">
                    {results.splats.splats.map((splat) => (
                      <div key={splat.name} className="splat-item">
                        <span className="splat-icon">◆</span>
                        <span className="splat-name">{splat.filename}</span>
                        <a 
                          href={`${API_BASE}${splat.url}`} 
                          download 
                          className="splat-download"
                        >
                          Download
                        </a>
                      </div>
                    ))}
                  </div>
                  <p className="splats-hint">
                    ✨ Splats have been copied to the viewer. Switch to the Viewer tab to see them!
                  </p>
                </>
              )}
            </div>
          )}
        </div>

        <div className="dashboard-right">
          {/* Configuration Section */}
          <div className="section config-section">
            <h3>Configuration</h3>
            
            <div className="config-group">
              <label className="config-label">
                <span>Field of View</span>
                <span className="config-value">{fov}°</span>
              </label>
              <input
                type="range"
                min="60"
                max="140"
                step="5"
                value={fov}
                onChange={(e) => setFov(Number(e.target.value))}
                className="config-slider"
              />
              <div className="slider-labels">
                <span>60°</span>
                <span>140°</span>
              </div>
            </div>

            <div className="config-group">
              <label className="config-label">
                <span>Output Size</span>
                <span className="config-value">{outputSize === 0 ? 'Auto' : `${outputSize}px`}</span>
              </label>
              <input
                type="range"
                min="0"
                max="2048"
                step="256"
                value={outputSize}
                onChange={(e) => setOutputSize(Number(e.target.value))}
                className="config-slider"
              />
              <div className="slider-labels">
                <span>Auto</span>
                <span>2048px</span>
              </div>
            </div>

            <div className="config-group faces-config">
              <div className="config-label-row">
                <span>Cube Faces ({selectedFaceCount}/6)</span>
                <div className="face-actions">
                  <button onClick={selectAllFaces} className="face-action-btn">All</button>
                  <button onClick={selectNoFaces} className="face-action-btn">None</button>
                </div>
              </div>
              <div className="faces-grid">
                {CUBE_FACES.map((face) => (
                  <label key={face} className="face-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedFaces[face]}
                      onChange={() => toggleFace(face)}
                    />
                    <span className="face-name">{face}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="config-group">
              <label className="toggle-option">
                <input
                  type="checkbox"
                  checked={generateSplats}
                  onChange={(e) => setGenerateSplats(e.target.checked)}
                />
                <span className="toggle-label">Generate Gaussian Splats</span>
              </label>
              <p className="toggle-hint">
                Creates .ply files for each cube face using the Sharp model
              </p>
            </div>
          </div>

          {/* Process Button */}
          <button 
            className="process-btn"
            onClick={handleProcess}
            disabled={!file || !serverOnline || isProcessing || selectedFaceCount === 0}
          >
            {isProcessing ? (
              <>
                <span className="spinner" />
                Processing...
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Generate {generateSplats ? 'Cubemap & Splats' : 'Cubemap'}
              </>
            )}
          </button>

          {!serverOnline && (
            <div className="server-warning">
              <p>⚠️ Backend server is not running.</p>
              <code>cd viewer/server && uv run uvicorn main:app --port 8765</code>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default GenerateDashboard;

