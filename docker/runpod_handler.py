"""RunPod serverless handler for StreamDiffusion API.

This allows the API to run as a RunPod serverless endpoint.
For persistent streaming, use the regular API deployment instead.
"""

import asyncio
import base64
import io
import logging
import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, "/app")

import runpod
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Global pipeline instance for reuse
_pipeline = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global _pipeline

    if _pipeline is None:
        from src.pipeline.wrapper import StreamDiffusionWrapper

        # Default configuration
        default_params = {
            "model_id": os.environ.get("MODEL_ID", "stabilityai/sdxl-turbo"),
            "width": int(os.environ.get("WIDTH", 768)),
            "height": int(os.environ.get("HEIGHT", 448)),
            "num_inference_steps": 50,
            "t_index_list": [11],
            "guidance_scale": 1.0,
            "delta": 1.0,
            "acceleration": "tensorrt",
            "use_lcm_lora": True,
            "use_denoising_batch": True,
        }

        _pipeline = StreamDiffusionWrapper(
            pipeline_type="streamdiffusion-sdxl",
            params=default_params,
        )

        # Initialize synchronously for RunPod
        asyncio.get_event_loop().run_until_complete(_pipeline.initialize())

    return _pipeline


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image data."""
    if image_data.startswith("data:"):
        # Remove data URL prefix
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image: Image.Image | np.ndarray, format: str = "JPEG") -> str:
    """Encode image to base64."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def process_request(job: dict) -> dict[str, Any]:
    """Process a single inference request."""
    job_input = job.get("input", {})

    # Get pipeline
    pipeline = get_pipeline()

    # Extract parameters
    image_data = job_input.get("image")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "")
    guidance_scale = job_input.get("guidance_scale", 1.0)
    seed = job_input.get("seed")

    # Update prompt if provided
    if prompt:
        await pipeline.update_params({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
        })

    # Update seed if provided
    if seed is not None:
        await pipeline.update_params({"seed": seed})

    # Process image
    if image_data:
        input_image = decode_image(image_data)
        output = await pipeline.process_frame(input_image)
        output_b64 = encode_image(output)
        return {"image": output_b64}
    else:
        # Text-to-image mode
        # Create a blank input for txt2img
        width = pipeline.params.get("width", 768)
        height = pipeline.params.get("height", 448)
        blank = Image.new("RGB", (width, height), color="black")
        output = await pipeline.process_frame(blank)
        output_b64 = encode_image(output)
        return {"image": output_b64}


def handler(job: dict) -> dict[str, Any]:
    """RunPod handler function."""
    try:
        result = asyncio.get_event_loop().run_until_complete(process_request(job))
        return result
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Start RunPod serverless handler
    runpod.serverless.start({"handler": handler})
