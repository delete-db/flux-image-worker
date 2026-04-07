"""
RunPod Serverless handler for Flux Kontext Dev image generation.
Pipeline loaded once at startup, kept in GPU memory.
Supports text-to-image and reference-based (character consistency) generation.
Strength parameter controls reference image influence on composition.
"""

import base64
import io
import math
import os
import time
from typing import Any

import numpy as np
import runpod
import torch
from PIL import Image

# ── Configuration ───────────────────────────────────────────

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/runpod-volume/ComfyUI/models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_ROOT, "flux-kontext-dev"))
USE_FP8 = os.environ.get("USE_FP8", "true").lower() == "true"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
WORKER_VERSION = "flux-kontext-v2"

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


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """Compute mu for dynamic shifting based on latent sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def build_strength_sigmas(num_steps, strength, height, width):
    """
    Build truncated sigma schedule for img2img strength control.
    Lower strength = fewer steps = less reference image influence on composition.
    Returns (sigmas_list, mu) for passing to the pipeline.
    """
    # Compute latent sequence length (after VAE 8x downscale + 2x2 packing)
    image_seq_len = (height // 16) * (width // 16)

    # Compute mu for dynamic shifting
    scheduler_config = PIPELINE.scheduler.config
    mu = calculate_shift(
        image_seq_len,
        scheduler_config.get("base_image_seq_len", 256),
        scheduler_config.get("max_image_seq_len", 4096),
        scheduler_config.get("base_shift", 0.5),
        scheduler_config.get("max_shift", 1.15),
    )

    # Build unshifted sigmas (the scheduler will apply dynamic shift)
    full_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

    # Truncate: skip early steps to reduce reference influence
    start_idx = int(round(num_steps * (1.0 - strength)))
    start_idx = max(0, min(start_idx, num_steps - 1))
    truncated = full_sigmas[start_idx:]

    return truncated.tolist(), mu


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

    # Strength: 1.0 = full creative freedom, 0.0 = keep reference exactly
    # Default 1.0 (no truncation). Use 0.4-0.6 for wide scenes with character refs.
    strength = float(job_input.get("strength", 1.0))
    strength = max(0.01, min(1.0, strength))

    mode = "i2i" if input_image else "t2i"
    print(f"Generating {mode}: {width}x{height}, seed={seed}, guidance={guidance_scale}, steps={num_steps}, strength={strength}")
    gen_start = time.time()

    try:
        generator = torch.Generator("cuda").manual_seed(seed)

        # Strength control: set timesteps on scheduler manually, then pass sigmas
        if input_image and strength < 0.95:
            sigmas, mu = build_strength_sigmas(num_steps, strength, height, width)
            # Set timesteps with mu on the scheduler directly
            PIPELINE.scheduler.set_timesteps(sigmas=sigmas, device="cuda", mu=mu)
            actual_steps = len(sigmas)
            print(f"  Strength {strength}: running {actual_steps}/{num_steps} steps (mu={mu:.3f})")

            result = PIPELINE(
                image=input_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                sigmas=sigmas,
                height=height,
                width=width,
                generator=generator,
            )
        else:
            result = PIPELINE(
                image=input_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                generator=generator,
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
