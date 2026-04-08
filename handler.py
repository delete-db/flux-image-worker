"""
RunPod Serverless handler for Flux Kontext Dev image generation.
Pipeline loaded once at startup, kept in GPU memory.
Supports text-to-image and reference-based (character consistency) generation.
"""

import base64
import io
import os
import time
from typing import Any

import runpod
import torch
from PIL import Image

# ── Configuration ───────────────────────────────────────────

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/runpod-volume/ComfyUI/models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_ROOT, "flux-kontext-dev"))
USE_FP8 = os.environ.get("USE_FP8", "true").lower() == "true"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
WORKER_VERSION = "flux-kontext-v3"

# ── Load Pipeline Once ──────────────────────────────────────

print(f"Worker version: {WORKER_VERSION}")
print(f"Loading Flux Kontext Dev pipeline...")
print(f"  Model path: {MODEL_PATH}")
print(f"  FP8: {USE_FP8}")

load_start = time.time()

from diffusers import FluxKontextPipeline, AutoModel

if USE_FP8:
    transformer = AutoModel.from_pretrained(
        MODEL_PATH,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )
    PIPELINE = FluxKontextPipeline.from_pretrained(
        MODEL_PATH,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
else:
    PIPELINE = FluxKontextPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

PIPELINE.to("cuda")

load_elapsed = time.time() - load_start
print(f"Pipeline loaded in {load_elapsed:.1f}s")


# ── Helpers ─────────────────────────────────────────────────

def decode_input_image(image_input: str) -> Image.Image:
    """Decode base64 or download URL image."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        import requests
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        payload = image_input.split(",", 1)[1] if "," in image_input else image_input
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def encode_output_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Handler ─────────────────────────────────────────────────

@torch.inference_mode()
def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required input: prompt"}

    width = int(job_input.get("width", 1080))
    height = int(job_input.get("height", 1920))
    # Snap to multiples of 32
    width = max(64, (width // 32) * 32)
    height = max(64, (height // 32) * 32)

    seed = int(job_input.get("seed", 42))
    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    num_steps = int(job_input.get("num_inference_steps", 28))
    output_format = job_input.get("output_format", "png").upper()
    if output_format not in ("PNG", "JPEG"):
        output_format = "PNG"

    # Handle reference image (for character consistency)
    input_image = None
    image_input = job_input.get("image")
    if image_input:
        try:
            input_image = decode_input_image(image_input)
        except Exception as exc:
            return {"error": f"Failed to decode input image: {exc}"}

    mode = "i2i" if input_image else "t2i"
    print(f"Generating {mode}: {width}x{height}, seed={seed}, guidance={guidance_scale}, steps={num_steps}")
    gen_start = time.time()

    try:
        generator = torch.Generator("cuda").manual_seed(seed)

        result = PIPELINE(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            generator=generator,
            max_sequence_length=512,
        )

        output_image = result.images[0]
        image_base64 = encode_output_image(output_image, output_format)

        gen_elapsed = time.time() - gen_start
        print(f"Generation complete in {gen_elapsed:.1f}s")

        return {
            "image_base64": image_base64,
            "mode": mode,
            "width": output_image.width,
            "height": output_image.height,
            "generation_time_seconds": round(gen_elapsed, 1),
        }

    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
