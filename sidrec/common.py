import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager

import torch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPARED = DATA / "processed" / "beauty"
WIDE_PREPARED = PREPARED.with_name("beauty_wide")
PREPARED_256 = PREPARED.with_name("beauty_256")
RUNS = ROOT / "runs"
METHODS = ("sequential", "global_pop", "zcr", "neighbors", "neighbors_zero")
CODEBOOK_SIZES = (32, 16, 8, 64)
WIDE_CODEBOOK_SIZES = (64, 32, 16, 64)
CODEBOOK_SIZES_256 = (256, 256, 256, 64)
NEIGHBOR_SWEEP = {
    "neighbor_weight_025": {"neighbor_weight": 0.25, "neighbors": 3, "sweeps": 4},
    "neighbor_weight_100": {"neighbor_weight": 1.0, "neighbors": 3, "sweeps": 4},
    "neighbor_k_1": {"neighbor_weight": 0.5, "neighbors": 1, "sweeps": 4},
}
HISTORY_LENGTH = 10
PAD = 0
BOS = 1


def prepared_for(preset):
    if preset in NEIGHBOR_SWEEP:
        return PREPARED_256 / "neighbor_sweep" / preset
    return {
        "course_wide": WIDE_PREPARED,
        "course_256": PREPARED_256,
        "course_256_long": PREPARED_256,
    }.get(preset, PREPARED)


def device_for(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
