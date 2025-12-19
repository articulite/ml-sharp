# Sharp Viewer Dashboard

A web-based dashboard for generating cubemaps and Gaussian splats from equirectangular panoramas using the Sharp model.

## Overview

The Sharp Viewer Dashboard provides an intuitive interface for the equirectangular-to-cubemap-to-splats workflow:

1. **Upload** an equirectangular (360°) panorama image
2. **Configure** FOV, output size, and which cube faces to generate
3. **Generate** cube face images and optionally Gaussian splats
4. **View** the generated splats in the integrated 3D viewer

## Quick Start

### Prerequisites

- Python 3.11+ with `uv` package manager
- Node.js 18+ with npm
- Sharp dependencies installed (see main README)

### Running the Dashboard

You need to run **two servers** - the Python backend and the Vite frontend.

#### 1. Start the Backend Server

```bash
cd viewer/server
uv run uvicorn main:app --reload --port 8765
```

The backend server handles:
- Image upload and processing
- Equirectangular to cube face conversion
- Gaussian splat generation via Sharp model
- WebSocket progress updates

#### 2. Start the Frontend

```bash
cd viewer
npm install  # First time only
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### Generate Tab

1. **Drop or select** an equirectangular image (2:1 aspect ratio panorama)
2. **Adjust settings**:
   - **Field of View**: 60° - 140° (default: 110°). Higher FOV creates more overlap between faces
   - **Output Size**: Auto or fixed pixel size (256 - 2048px)
   - **Cube Faces**: Select which faces to generate (front, back, left, right, top, bottom)
   - **Generate Gaussian Splats**: Enable to run Sharp model on each face
3. **Click "Generate"** and watch the progress bar
4. **Results** show generated cube face previews and downloadable splat files

### Viewer Tab

- Generated splats are automatically copied to `/public/splats/`
- Toggle individual cube faces on/off
- Adjust splat scale and frustum visualization
- Navigate with mouse (orbit) and WASD+QE keys

## Architecture

```
viewer/
├── server/
│   ├── main.py           # FastAPI backend with WebSocket progress
│   └── requirements.txt  # Python dependencies
├── src/
│   ├── App.jsx           # Main app with tab navigation
│   └── components/
│       ├── GenerateDashboard.jsx  # Upload & generate UI
│       ├── GenerateDashboard.css  # Dashboard styles
│       ├── GaussianSplats.jsx     # 3D splat renderer
│       └── FrustumVisualizer.jsx  # Frustum visualization
└── public/
    ├── generated/        # Job outputs (cube faces, splats)
    └── splats/           # Active splats for viewer
```

## API Endpoints

### REST

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/process` | POST | Start pipeline (multipart form) |
| `/api/jobs/{job_id}/files` | GET | List generated files |
| `/generated/{job_id}/{subdir}/{filename}` | GET | Serve generated files |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{job_id}` | Progress updates for job |

Progress messages:
```json
{
  "job_id": "job_...",
  "percent": 45.5,
  "status": "running",
  "message": "Extracting front face (1/6)",
  "results": {}
}
```

## Configuration Options

### Field of View (FOV)

The FOV controls how much of the panorama is captured in each cube face:

- **90°**: Standard cubemap with no overlap
- **110°** (default): ~10% overlap between adjacent faces, good for splat merging
- **140°**: Maximum overlap, useful for seamless transitions

### Output Size

- **Auto**: Calculated from input resolution (input height / 2, rounded to multiple of 16)
- **Fixed**: Specify exact pixel dimensions (256 - 2048px)

### Generate Gaussian Splats

When enabled, runs the Sharp model on each cube face to generate `.ply` splat files. This requires:
- Sufficient GPU memory (CUDA recommended)
- First run downloads the model (~500MB)

## Troubleshooting

### "Server Offline" Status

Make sure the backend is running:
```bash
cd viewer/server
uv run uvicorn main:app --port 8765
```

### Model Not Found

The dashboard requires the Sharp model to be downloaded first. Run any prediction to cache it:
```bash
sharp predict -i <any_image.jpg> -o /tmp/test
```
This downloads and caches the model for future dashboard use.

### Out of Memory

- Reduce output size
- Process fewer faces at once
- Use `--device cpu` (much slower)

### Splats Not Showing in Viewer

- Check that files exist in `viewer/public/splats/`
- Refresh the page or switch tabs
- Check browser console for errors

## Development

### Backend Development

```bash
cd viewer/server
uv run uvicorn main:app --reload --port 8765
```

The `--reload` flag enables hot reloading on code changes.

### Frontend Development

```bash
cd viewer
npm run dev
```

Vite provides hot module replacement for instant updates.

## License

See main repository [LICENSE](../LICENSE) and [LICENSE_MODEL](../LICENSE_MODEL).

