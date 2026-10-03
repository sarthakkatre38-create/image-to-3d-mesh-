from __future__ import annotations

import io
import os
import sys
import time
import uuid
import threading
import traceback
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import psutil
import torch
import trimesh

from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRIPOSR_DIR = ROOT / "TripoSR"
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OFFLINE SETTINGS
# ============================================================

os.environ.setdefault(
    "U2NET_HOME",
    str(WEIGHTS_DIR / "u2net"),
)

os.environ.setdefault(
    "HF_HUB_OFFLINE",
    "1",
)

if str(TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(TRIPOSR_DIR))


# ============================================================
# TRIPOSR
# ============================================================

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
except Exception as exc:
    raise RuntimeError(
        "Failed to import TripoSR.\n"
        f"TripoSR directory: {TRIPOSR_DIR}\n"
        f"Error: {exc}"
    ) from exc


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Scan3D server")
print("=" * 60)
print(f"Project : {ROOT}")
print(f"TripoSR : {TRIPOSR_DIR}")
print(f"Weights : {WEIGHTS_DIR}")
print(f"Device  : {DEVICE}")

if DEVICE == "cuda":
    try:
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
else:
    print("WARNING: CUDA is not available.")

print("=" * 60)


# ============================================================
# QUALITY
# ============================================================

QUALITY = {
    "light": {
        "resolution":  64,
        "target_faces": 10000,
    },
    "balanced": {
        "resolution": 96,
        "target_faces": 40000,
    },
    "detailed": {
        "resolution": 128,
        "target_faces": 120000,
    },
}


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

_model = None
_rembg_session = None

_model_lock = threading.Lock()
_generation_lock = threading.Lock()

jobs = {}
jobs_lock = threading.Lock()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Scan3D",
    version="1.0.0",
)


# ============================================================
# JOB HELPERS
# ============================================================

def create_job():

    job_id = uuid.uuid4().hex[:12]

    job = {
        "id": job_id,
        "status": "queued",
        "stage": "Waiting",
        "progress": 0,
        "elapsed": 0,
        "eta": None,
        "message": "Waiting to start...",
        "result": None,
        "error": None,
        "started_at": None,
    }

    with jobs_lock:
        jobs[job_id] = job

    return job_id


def update_job(
    job_id,
    *,
    status=None,
    stage=None,
    progress=None,
    message=None,
    eta=None,
):

    with jobs_lock:

        job = jobs.get(job_id)

        if job is None:
            return

        if status is not None:
            job["status"] = status

        if stage is not None:
            job["stage"] = stage

        if progress is not None:
            job["progress"] = max(
                0,
                min(100, int(progress)),
            )

        if message is not None:
            job["message"] = message

        if eta is not None:
            job["eta"] = eta

        if job["started_at"] is not None:
            job["elapsed"] = round(
                time.perf_counter()
                - job["started_at"],
                1,
            )


def finish_job(
    job_id,
    result,
):

    with jobs_lock:

        job = jobs.get(job_id)

        if job is None:
            return

        job["status"] = "completed"
        job["stage"] = "Complete"
        job["progress"] = 100
        job["eta"] = 0
        job["message"] = "Model generated successfully."
        job["result"] = result

        if job["started_at"] is not None:
            job["elapsed"] = round(
                time.perf_counter()
                - job["started_at"],
                1,
            )


def fail_job(
    job_id,
    error,
):

    with jobs_lock:

        job = jobs.get(job_id)

        if job is None:
            return

        job["status"] = "failed"
        job["stage"] = "Failed"
        job["message"] = str(error)
        job["error"] = str(error)

        if job["started_at"] is not None:
            job["elapsed"] = round(
                time.perf_counter()
                - job["started_at"],
                1,
            )


def get_job(job_id):

    with jobs_lock:

        job = jobs.get(job_id)

        if job is None:
            return None

        return dict(job)


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    global _model
    global _rembg_session

    with _model_lock:

        if _model is None:

            triposr_weights_dir = WEIGHTS_DIR / "triposr"

            config_file = triposr_weights_dir / "config.yaml"
            checkpoint_file = triposr_weights_dir / "model.ckpt"

            if not config_file.exists():
                raise FileNotFoundError(
                    f"Missing TripoSR config:\n{config_file}"
                )

            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Missing TripoSR checkpoint:\n{checkpoint_file}"
                )

            print()
            print("Loading TripoSR...")
            print(f"Config     : {config_file}")
            print(f"Checkpoint : {checkpoint_file}")

            _model = TSR.from_pretrained(
                str(triposr_weights_dir),
                config_name="config.yaml",
                weight_name="model.ckpt",
            )

            _model.to(DEVICE)
            _model.eval()
            _model.renderer.set_chunk_size(8192)

            print("✓ TripoSR loaded")

        if _rembg_session is None:

            print("Loading background-removal model...")

            try:

                from rembg import new_session

                _rembg_session = new_session(
                    "u2net"
                )

                print("✓ rembg loaded")

            except Exception as exc:

                print(
                    "WARNING: rembg could not be loaded:"
                )
                print(exc)

                _rembg_session = False


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    # Start with RGBA so background removal can use transparency.
    image = image.convert("RGBA")

    if _rembg_session is not False:

        try:

            image = remove_background(
                image,
                session=_rembg_session,
            )

        except TypeError:

            image = remove_background(
                image
            )

        except Exception as exc:

            print(
                "WARNING: background removal failed:"
            )
            print(exc)

    try:

        image = resize_foreground(
            image,
            0.85,
        )

    except TypeError:

        image = resize_foreground(
            image
        )

    # --------------------------------------------------------
    # TripoSR expects RGB, not RGBA.
    # Composite transparent areas onto a white background.
    # --------------------------------------------------------

    if image.mode == "RGBA":

        background = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255, 255),
        )

        image = Image.alpha_composite(
            background,
            image,
        )

        image = image.convert("RGB")

    else:

        image = image.convert("RGB")

    return image


# ============================================================
# MESH CLEANUP
# ============================================================

def clean_mesh(mesh):

    if mesh is None:
        raise RuntimeError(
            "TripoSR returned no mesh."
        )

    if len(mesh.vertices) == 0:
        raise RuntimeError(
            "Generated mesh has no vertices."
        )

    if len(mesh.faces) == 0:
        raise RuntimeError(
            "Generated mesh has no faces."
        )

    try:
        mesh.remove_infinite_values()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


# ============================================================
# FACE REDUCTION
# ============================================================

def reduce_faces(
    mesh,
    target_faces,
):

    target_faces = int(target_faces)

    if target_faces <= 0:
        return mesh

    if len(mesh.faces) <= target_faces:
        return mesh

    try:

        simplified = mesh.simplify_quadric_decimation(
            face_count=target_faces
        )

        if simplified is not None:
            return simplified

    except TypeError:

        try:

            simplified = mesh.simplify_quadric_decimation(
                target_faces
            )

            if simplified is not None:
                return simplified

        except Exception as exc:

            print(
                "Face reduction unavailable:"
            )
            print(exc)

    except Exception as exc:

        print(
            "Face reduction unavailable:"
        )
        print(exc)

    print(
        "Keeping original mesh."
    )

    return mesh


# ============================================================
# EXPORT
# ============================================================

def export_mesh(
    mesh,
    output_base,
):

    glb_path = output_base.with_suffix(
        ".glb"
    )

    stl_path = output_base.with_suffix(
        ".stl"
    )

    glb_data = mesh.export(
        file_type="glb"
    )

    if isinstance(glb_data, str):
        glb_data = glb_data.encode()

    glb_path.write_bytes(
        glb_data
    )

    stl_data = mesh.export(
        file_type="stl"
    )

    if isinstance(stl_data, str):
        stl_data = stl_data.encode()

    stl_path.write_bytes(
        stl_data
    )

    return glb_path, stl_path


# ============================================================
# GENERATION WORKER
# ============================================================

def generate_worker(
    job_id,
    raw_data,
    quality,
):

    start = time.perf_counter()

    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["started_at"] = start

    settings = QUALITY[quality]

    resolution = settings["resolution"]
    target_faces = settings["target_faces"]

    try:

        # ----------------------------------------------------
        # 0 → 5
        # ----------------------------------------------------

        update_job(
            job_id,
            status="running",
            stage="Preparing image",
            progress=5,
            message="Preparing your image...",
        )

        pil_image = Image.open(
            io.BytesIO(raw_data)
        )

        pil_image.load()

        pil_image = pil_image.convert(
            "RGBA"
        )

        # ----------------------------------------------------
        # 5 → 15
        # ----------------------------------------------------

        update_job(
            job_id,
            stage="Loading AI model",
            progress=10,
            message="Loading local AI model...",
        )

        load_models()

        update_job(
            job_id,
            stage="Removing background",
            progress=15,
            message="Removing background...",
        )

        prepared = prepare_image(
            pil_image
        )

        # ----------------------------------------------------
        # 15 → 55
        # ----------------------------------------------------

        update_job(
            job_id,
            stage="Generating 3D scene",
            progress=20,
            message="AI is analyzing the image...",
        )

        inference_start = time.perf_counter()

        with torch.no_grad():

            scene_codes = _model(
                [prepared],
                device=DEVICE,
            )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        # ----------------------------------------------------
        # 55 → 85
        # ----------------------------------------------------

        update_job(
            job_id,
            stage="Generating 3D geometry",
            progress=60,
            message="Creating 3D geometry...",
        )

        mesh_start = time.perf_counter()

        with torch.no_grad():

            meshes = _model.extract_mesh(
                scene_codes,
                resolution=resolution,
            )

        mesh_time = (
            time.perf_counter()
            - mesh_start
        )

        if not meshes:
            raise RuntimeError(
                "TripoSR returned an empty mesh list."
            )

        mesh = meshes[0]

        print(
            f"Raw mesh: "
            f"{len(mesh.vertices):,} vertices / "
            f"{len(mesh.faces):,} faces"
        )

        update_job(
            job_id,
            stage="Optimizing mesh",
            progress=85,
            message="Cleaning and optimizing mesh...",
        )

        mesh = clean_mesh(
            mesh
        )

        mesh = reduce_faces(
            mesh,
            target_faces,
        )

        mesh = clean_mesh(
            mesh
        )

        # ----------------------------------------------------
        # 93 → 100
        # ----------------------------------------------------

        update_job(
            job_id,
            stage="Exporting model",
            progress=94,
            message="Creating GLB and STL files...",
        )

        job_id_short = job_id

        output_base = (
            OUTPUT_DIR / job_id_short
        )

        glb_path, stl_path = export_mesh(
            mesh,
            output_base,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        triangles = len(mesh.faces)

        result = {
            "ok": True,
            "id": job_id,
            "glb": f"/outputs/{glb_path.name}",
            "stl": f"/outputs/{stl_path.name}",
            "triangles": triangles,
            "vertices": len(mesh.vertices),
            "seconds": round(
                elapsed,
                2,
            ),
            "quality": quality,
            "resolution": resolution,
            "inference_seconds": round(
                inference_time,
                2,
            ),
            "mesh_seconds": round(
                mesh_time,
                2,
            ),
        }

        finish_job(
            job_id,
            result,
        )

        print()
        print("=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(
            f"Time      : {elapsed:.2f}s"
        )
        print(
            f"Triangles : {triangles:,}"
        )
        print(
            f"Vertices  : {len(mesh.vertices):,}"
        )
        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print("GENERATION FAILED")
        print("=" * 60)

        traceback.print_exc()

        print("=" * 60)

        fail_job(
            job_id,
            exc,
        )

    finally:

        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================
# START GENERATION
# ============================================================

@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    quality: str = Form("balanced"),
):

    if quality not in QUALITY:

        raise HTTPException(
            status_code=400,
            detail="Invalid quality.",
        )

    if not image.content_type:

        raise HTTPException(
            status_code=400,
            detail="No image content type.",
        )

    if not image.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="File must be an image.",
        )

    try:

        raw_data = await image.read()

        if not raw_data:
            raise ValueError(
                "Image is empty."
            )

        # Validate image before creating job.
        test_image = Image.open(
            io.BytesIO(raw_data)
        )

        test_image.verify()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    job_id = create_job()

    worker = threading.Thread(
        target=generate_worker,
        args=(
            job_id,
            raw_data,
            quality,
        ),
        daemon=True,
    )

    worker.start()

    return {
        "ok": True,
        "job_id": job_id,
    }


# ============================================================
# PROGRESS API
# ============================================================

@app.get("/api/progress/{job_id}")
def progress(
    job_id: str,
):

    job = get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return job


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    memory = psutil.virtual_memory()

    result = {
        "ok": True,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "model_loaded": _model is not None,
        "weights": (
            (WEIGHTS_DIR /"triposr"/ "model.ckpt").exists()
            and
            (WEIGHTS_DIR / "triposr"/ "config.yaml").exists()
        ),
        "ram_used_percent": memory.percent,
        "ram_available_gb": round(
            memory.available / (1024 ** 3),
            2,
        ),
    }

    if torch.cuda.is_available():

        try:

            result["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )

            result["gpu_memory_allocated_mb"] = round(
                torch.cuda.memory_allocated(0)
                / (1024 ** 2),
                1,
            )

        except Exception:
            pass

    return result


# ============================================================
# OUTPUT FILE
# ============================================================

@app.get("/outputs/{filename}")
def output_file(
    filename: str,
):

    safe_name = Path(
        filename
    ).name

    path = OUTPUT_DIR / safe_name

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Output file not found.",
        )

    return FileResponse(
        path
    )


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def index():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="static/index.html not found.",
        )

    return FileResponse(
        index_file
    )


# ============================================================
# STATIC
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static",
)

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print("=" * 60)
    print("Starting Scan3D")
    print("=" * 60)
    print(
        "Open: http://127.0.0.1:8000/"
    )
    print(
        "API:  http://127.0.0.1:8000/api/status"
    )
    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )