# Sharp Viewer Backend Server

FastAPI backend server for the Sharp Viewer Dashboard.

## Prerequisites

- Python 3.11+
- The main `sharp` package must be installed (from the project root)

## Running the Server

The server imports from the `sharp` package which requires PyTorch and other dependencies. To avoid DLL/dependency conflicts on Windows, **run the server from the main project root**:

```bash
cd C:\Users\kaika\BP\gitprojects\ml-sharp
uv run uvicorn viewer.server.main:app --reload --port 8765
```

Or on Linux/macOS:

```bash
cd /path/to/ml-sharp
uv run uvicorn viewer.server.main:app --reload --port 8765
```

The server will be available at `http://localhost:8765`.

## API Endpoints

See the auto-generated API docs at:
- Swagger UI: `http://localhost:8765/docs`
- ReDoc: `http://localhost:8765/redoc`

## Development

If you need to add Python dependencies, add them to `pyproject.toml` in this directory. However, for dependencies already provided by the main `sharp` package (like `torch`, `numpy`, etc.), rely on the main project environment instead.

