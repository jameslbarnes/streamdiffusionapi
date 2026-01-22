"""Pytest configuration and fixtures."""

import asyncio
import sys

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_stream_config():
    """Sample stream configuration matching Daydream format."""
    return {
        "pipeline": "streamdiffusion-sdxl",
        "name": "Test Stream",
        "params": {
            "model_id": "stabilityai/sdxl-turbo",
            "prompt": "beautiful nature, minimalistic, vibrant, colorful",
            "negative_prompt": "blurry, low quality, flat, 2d",
            "num_inference_steps": 50,
            "seed": 704475,
            "t_index_list": [11],
            "width": 768,
            "height": 448,
            "acceleration": "tensorrt",
            "lora_dict": None,
            "ip_adapter": {
                "scale": 0.5,
                "enabled": False,
            },
            "controlnets": [
                {
                    "enabled": True,
                    "model_id": "xinsir/controlnet-depth-sdxl-1.0",
                    "preprocessor": "depth_tensorrt",
                    "conditioning_scale": 0,
                    "preprocessor_params": {},
                },
                {
                    "enabled": True,
                    "model_id": "xinsir/controlnet-canny-sdxl-1.0",
                    "preprocessor": "canny",
                    "conditioning_scale": 0,
                    "preprocessor_params": {
                        "low_threshold": 100,
                        "high_threshold": 200,
                    },
                },
            ],
            "use_lcm_lora": True,
            "guidance_scale": 1,
            "delta": 1,
            "do_add_noise": True,
            "use_denoising_batch": True,
            "use_safety_checker": False,
            "enable_similar_image_filter": False,
            "video_codec": "h264",
            "video_profile": "high",
            "video_preset": "medium",
        },
        "output_rtmp_url": None,
    }


@pytest.fixture
def sample_controlnet_config():
    """Sample ControlNet configuration."""
    return {
        "enabled": True,
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "preprocessor": "depth_tensorrt",
        "conditioning_scale": 0.5,
        "preprocessor_params": {},
    }
