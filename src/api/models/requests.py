"""Request models for the StreamDiffusion API."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ControlNetConfig(BaseModel):
    """Configuration for a single ControlNet."""

    enabled: bool = True
    model_id: str = Field(..., description="HuggingFace model ID for the ControlNet")
    preprocessor: str = Field(
        ...,
        description="Preprocessor type: depth_tensorrt, canny, feedback, openpose, etc.",
    )
    conditioning_scale: float = Field(
        0.5, ge=0.0, le=2.0, description="Conditioning scale for this ControlNet"
    )
    preprocessor_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the preprocessor",
    )


class IPAdapterConfig(BaseModel):
    """Configuration for IP-Adapter."""

    enabled: bool = False
    scale: float = Field(0.5, ge=0.0, le=1.0, description="IP-Adapter influence scale")
    image_url: str | None = Field(None, description="URL to style reference image")


class StreamParams(BaseModel):
    """Parameters for the StreamDiffusion pipeline."""

    # Model configuration
    model_id: str = Field(
        "stabilityai/sdxl-turbo",
        description="HuggingFace model ID",
    )
    acceleration: Literal["none", "xformers", "tensorrt"] = Field(
        "tensorrt",
        description="Acceleration method",
    )

    # Prompt configuration
    prompt: str = Field("", description="Positive prompt for generation")
    negative_prompt: str = Field("", description="Negative prompt")

    # Generation parameters
    num_inference_steps: int = Field(50, ge=1, le=100, description="Total denoising steps")
    t_index_list: list[int] = Field(
        [11],
        description="Timestep indices for actual inference (1-4 values)",
    )
    guidance_scale: float = Field(1.0, ge=0.0, description="CFG scale")
    delta: float = Field(1.0, ge=0.0, le=1.0, description="RCFG delta parameter")
    seed: int = Field(42, description="Random seed for generation")

    # Resolution
    width: int = Field(768, ge=256, le=2048, description="Output width")
    height: int = Field(448, ge=256, le=2048, description="Output height")

    # LoRA configuration
    lora_dict: dict[str, float] | None = Field(
        None,
        description="Dictionary of LoRA model paths to weights",
    )
    use_lcm_lora: bool = Field(True, description="Whether to use LCM-LoRA for acceleration")

    # ControlNet configuration
    controlnets: list[ControlNetConfig] = Field(
        default_factory=list,
        description="List of ControlNet configurations",
    )

    # IP-Adapter configuration
    ip_adapter: IPAdapterConfig = Field(
        default_factory=IPAdapterConfig,
        description="IP-Adapter configuration",
    )

    # Advanced generation settings
    do_add_noise: bool = Field(True, description="Whether to add noise during denoising")
    use_denoising_batch: bool = Field(True, description="Use batch denoising optimization")
    use_safety_checker: bool = Field(False, description="Enable safety checker")

    # Interpolation settings
    normalize_seed_weights: bool = Field(True, description="Normalize seed blend weights")
    normalize_prompt_weights: bool = Field(True, description="Normalize prompt blend weights")
    seed_interpolation_method: Literal["linear", "slerp"] = Field(
        "linear",
        description="Seed interpolation method",
    )
    prompt_interpolation_method: Literal["linear", "slerp"] = Field(
        "linear",
        description="Prompt interpolation method",
    )

    # Similar image filter (skip redundant frames)
    enable_similar_image_filter: bool = Field(
        False,
        description="Skip frames similar to previous output",
    )
    similar_image_filter_threshold: float = Field(
        0.98,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for skipping",
    )
    similar_image_filter_max_skip_frame: int = Field(
        10,
        ge=1,
        description="Maximum consecutive frames to skip",
    )

    # Video encoding settings
    video_codec: Literal["h264", "h265", "vp9"] = Field("h264", description="Video codec")
    video_profile: Literal["baseline", "main", "high"] = Field(
        "high",
        description="H.264 profile",
    )
    video_tune: Literal["film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"] = Field(
        "film",
        description="x264 tune preset",
    )
    video_preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] = Field(
        "medium",
        description="Encoding speed preset",
    )
    video_bframes: int = Field(2, ge=0, le=16, description="Number of B-frames")
    gop_size: int = Field(30, ge=1, le=300, description="GOP size (keyframe interval)")


class CreateStreamRequest(BaseModel):
    """Request to create a new stream."""

    pipeline: Literal["streamdiffusion", "streamdiffusion-sdxl", "streamdiffusion-sd15"] = Field(
        "streamdiffusion-sdxl",
        description="Pipeline type to use",
    )
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable stream name")
    params: StreamParams = Field(..., description="Pipeline parameters")
    output_rtmp_url: str | None = Field(
        None,
        description="Optional RTMP URL to restream output",
    )


class UpdateStreamRequest(BaseModel):
    """Request to update an existing stream.

    Only include fields you want to update. Fields set to None are not updated.
    """

    pipeline: Literal["streamdiffusion", "streamdiffusion-sdxl", "streamdiffusion-sd15"] | None = None
    params: StreamParams | None = None
    output_rtmp_url: str | None = None


class PartialStreamParams(BaseModel):
    """Partial parameters for hot-reload updates.

    These parameters can be updated without restarting the pipeline.
    """

    prompt: str | None = None
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    delta: float | None = None
    num_inference_steps: int | None = None
    t_index_list: list[int] | None = None
    seed: int | None = None
    # ControlNet scales can be updated without reload
    controlnet_scales: dict[int, float] | None = Field(
        None,
        description="Map of controlnet index to new conditioning_scale",
    )
