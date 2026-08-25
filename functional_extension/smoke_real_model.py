from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
HF_CACHE = HERE / ".venv" / "hf_cache"

for extra in (str(SRC), str(HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from PIL import Image as PILImage

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest
from eagent_functional.transformers_provider import RealTransformersVisionLanguageModel

PROMPT = "Describe the image in one sentence."
FIXTURE_PATH = HF_CACHE / "deterministic_fixture.png"
MAX_NEW_TOKENS = 40


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
_KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
_PSAPI.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_MemoryCounters),
    wintypes.DWORD,
]
_PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def _memory_snapshot() -> tuple[int, int, int]:
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(_MemoryCounters)
    handle = _KERNEL32.GetCurrentProcess()
    ok = _PSAPI.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return (0, 0, 0)
    return (
        counters.WorkingSetSize,
        counters.PeakWorkingSetSize,
        counters.PrivateUsage,
    )


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def _mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _write_deterministic_fixture() -> None:
    canvas = PILImage.new("RGB", (128, 128), (30, 60, 160))
    for x in range(32, 96):
        for y in range(32, 96):
            canvas.putpixel((x, y), (200, 40, 40))
    canvas.save(FIXTURE_PATH, format="PNG")


def main() -> int:
    print("M1 STEP 2A standalone real-model provider smoke test")
    print("deterministic development/functional substitute; CPU only; no stub fallback")
    print()

    _write_deterministic_fixture()
    fixture_bytes = FIXTURE_PATH.stat().st_size
    print(f"fixture_path: {FIXTURE_PATH}")
    print(f"fixture_size_bytes: {fixture_bytes}")

    baseline_ws, _baseline_peak, baseline_private = _memory_snapshot()
    cache_before = _dir_size_bytes(HF_CACHE)

    model = RealTransformersVisionLanguageModel()
    image = Image(path=str(FIXTURE_PATH))
    request = ModelRequest(prompt=PROMPT, images=[image], max_tokens=MAX_NEW_TOKENS)

    load_start = time.perf_counter()
    loaded_model, processor = model._ensure_loaded()
    load_seconds = time.perf_counter() - load_start

    device_attr = getattr(loaded_model, "device", None)
    print()
    print(f"model_loaded: {loaded_model is not None}")
    print(f"model_device: {device_attr}")
    print(f"processor_type: {type(processor).__name__}")
    print(f"model_load_seconds: {load_seconds:.2f}")

    pil_probe = model._pil_image(image)
    probe_messages = model._chat_messages(request)
    probe_templated = processor.apply_chat_template(
        probe_messages, add_generation_prompt=True
    )
    probe_inputs = processor(
        text=probe_templated, images=[pil_probe], return_tensors="pt"
    )
    has_pixel_values = "pixel_values" in probe_inputs
    pixel_shape = (
        tuple(probe_inputs["pixel_values"].shape) if has_pixel_values else None
    )
    print()
    print(f"chat_template_has_image_placeholder: {probe_messages[0]['content'][0]}")
    print(f"processor_produced_pixel_values: {has_pixel_values}")
    print(f"pixel_values_shape: {pixel_shape}")

    infer_start = time.perf_counter()
    response = model.generate(request)
    infer_seconds = time.perf_counter() - infer_start

    after_ws, peak_ws, after_private = _memory_snapshot()
    cache_after = _dir_size_bytes(HF_CACHE)
    download_bytes = cache_after - cache_before

    print()
    print(f"inference_seconds: {infer_seconds:.2f}")
    print(f"response_model_name: {response.model_name}")
    print(f"response_model_name_matches_selected_id: {response.model_name == RealTransformersVisionLanguageModel.DEFAULT_MODEL_ID}")
    print(f"response_usage: {response.usage}")
    print(f"response_text_nonempty: {bool(response.text and response.text.strip())}")
    print(f"response_text_length: {len(response.text)}")
    print("response_text_begin>>>")
    print(response.text)
    print("<<<response_text_end")

    print()
    print(f"model_download_size_bytes: {download_bytes}")
    print(f"model_download_size_mib: {_mib(download_bytes):.1f}")
    print(f"hf_cache_total_mib: {_mib(cache_after):.1f}")
    print(f"working_set_baseline_mib: {_mib(baseline_ws):.1f}")
    print(f"working_set_after_mib: {_mib(after_ws):.1f}")
    print(f"working_set_peak_mib: {_mib(peak_ws):.1f}")
    print(f"private_usage_baseline_mib: {_mib(baseline_private):.1f}")
    print(f"private_usage_after_mib: {_mib(after_private):.1f}")

    image_processed = (
        has_pixel_values
        and response.usage is not None
        and response.usage.get("images") == 1
    )
    text_ok = bool(response.text and response.text.strip())
    name_ok = response.model_name == RealTransformersVisionLanguageModel.DEFAULT_MODEL_ID
    success = (
        loaded_model is not None and image_processed and text_ok and name_ok
    )

    print()
    print(f"image_input_successfully_processed: {image_processed}")
    print(f"SMOKE_TEST_SUCCESS: {success}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
