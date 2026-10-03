# Photo to 3D (fully offline)

Scan with your camera or upload a picture -> optimized 3D model (GLB + STL).
Uses TripoSR (MIT license). No accounts, no limits, nothing leaves your PC.

## One-time setup (needs internet once)
1. Install Python (3.11 is the safest; 3.12-3.14 should also work) and Git.
2. Install PyTorch for your hardware (see https://pytorch.org). NVIDIA GPU on Windows:
   `pip install torch --index-url https://download.pytorch.org/whl/cu128`
   (NVIDIA GPU: pick the CUDA build. No GPU: the CPU build works but is slower.)
3. `pip install -r requirements.txt`
4. Build torchmcubes against the PyTorch you installed (it must NOT use build isolation):
       pip install scikit-build-core pybind11 cmake
       pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
   On Windows this needs "Build Tools for Visual Studio 2022" with the "Desktop development with C++"
   workload. The CUDA Toolkit is optional (without it a CPU-only marching-cubes build is made, which is fine).
5. `python download_models.py`  <- downloads TripoSR code + weights + background remover

## Every time (no internet needed)
    python server.py
Open http://127.0.0.1:8000

## Notes
- Output is centered, scaled to 1 unit, Y-up, decimated, with vertex colors. Choose Light / Balanced / Detailed.
- One photo gives a guessed back side. For a truly measured scan you would add a multi-photo
  photogrammetry mode (COLMAP/OpenMVS) later.
- The viewer's three.js is bundled in static/vendor, so it works offline.

## Tested-for hardware notes
- NVIDIA RTX 3050 6 GB: use the CUDA 12.1 PyTorch build. The server lowers its memory use automatically on cards under 8 GB.
  You do not need the CUDA Toolkit; `torchmcubes` will build its CPU version, which is fine.
- If you get "CUDA out of memory", choose Light or Balanced detail and close other GPU apps (games, browsers with many tabs).
