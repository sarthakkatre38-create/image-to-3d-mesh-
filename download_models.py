"""Run ONCE while online. After this, the app never needs the internet."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
os.environ["U2NET_HOME"] = str(ROOT / "weights" / "u2net")

# 1) TripoSR code (MIT license)
if not (ROOT / "TripoSR").exists():
    subprocess.check_call(["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", str(ROOT / "TripoSR")])

# 2) TripoSR weights
from huggingface_hub import hf_hub_download

dest = ROOT / "weights" / "triposr"
dest.mkdir(parents=True, exist_ok=True)
for fname in ("config.yaml", "model.ckpt"):
    shutil.copy(hf_hub_download("stabilityai/TripoSR", fname), dest / fname)

# 2b) Small config file TripoSR reads at startup (cached for offline use)
from transformers import ViTConfig

ViTConfig.from_pretrained("facebook/dino-vitb16")

# 3) Background-removal model (rembg / u2net)
from rembg import new_session

new_session()

print("Done. Start the app with: python server.py")
