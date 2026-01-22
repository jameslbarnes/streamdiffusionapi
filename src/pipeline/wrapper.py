"""StreamDiffusion pipeline wrapper."""

import asyncio
import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class StreamDiffusionWrapper:
    """Wrapper around StreamDiffusion for the API.

    This class handles:
    - Pipeline initialization and configuration
    - Model loading with optional TensorRT acceleration
    - ControlNet and IP-Adapter setup
    - Frame processing
    - Hot parameter updates
    """

    # Parameters that can be updated without pipeline restart
    HOT_PARAMS = {
        "prompt",
        "negative_prompt",
        "guidance_scale",
        "delta",
        "num_inference_steps",
        "t_index_list",
        "seed",
    }

    def __init__(self, pipeline_type: str, params: dict[str, Any]):
        self.pipeline_type = pipeline_type
        self.params = params
        self.stream = None
        self.pipe = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # Determine model type from pipeline
        self.is_sdxl = "sdxl" in pipeline_type.lower()

    async def initialize(self) -> None:
        """Initialize the StreamDiffusion pipeline."""
        if self._initialized:
            return

        async with self._lock:
            logger.info(f"Initializing {self.pipeline_type} pipeline...")

            # Run initialization in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_sync)

            self._initialized = True
            logger.info("Pipeline initialized")

    def _init_sync(self) -> None:
        """Synchronous initialization (runs in thread pool)."""
        import torch
        from diffusers import AutoencoderTiny

        # Import StreamDiffusion components
        try:
            from streamdiffusion import StreamDiffusion
            from streamdiffusion.image_utils import postprocess_image
        except ImportError as e:
            raise RuntimeError(
                "StreamDiffusion not installed. Run: "
                "pip install 'git+https://github.com/daydreamlive/StreamDiffusion.git@main"
                "#egg=streamdiffusion[tensorrt,controlnet,ipadapter]'"
            ) from e

        model_id = self.params.get("model_id", "stabilityai/sdxl-turbo")
        width = self.params.get("width", 768)
        height = self.params.get("height", 448)
        acceleration = self.params.get("acceleration", "tensorrt")

        logger.info(f"Loading model: {model_id}")

        # Load the base pipeline
        if self.is_sdxl:
            from diffusers import StableDiffusionXLPipeline

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to("cuda")
        else:
            from diffusers import StableDiffusionPipeline

            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to("cuda")

        # Get t_index_list for StreamDiffusion
        t_index_list = self.params.get("t_index_list", [11])
        num_inference_steps = self.params.get("num_inference_steps", 50)

        # Determine cfg_type based on guidance_scale
        guidance_scale = self.params.get("guidance_scale", 1.0)
        cfg_type = "none" if guidance_scale <= 1.0 else "self"

        # Create StreamDiffusion wrapper
        self.stream = StreamDiffusion(
            self.pipe,
            t_index_list=t_index_list,
            torch_dtype=torch.float16,
            cfg_type=cfg_type,
            width=width,
            height=height,
            use_denoising_batch=self.params.get("use_denoising_batch", True),
        )

        # Use Tiny VAE for faster decoding
        try:
            if self.is_sdxl:
                self.stream.vae = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesdxl"
                ).to("cuda", dtype=torch.float16)
            else:
                self.stream.vae = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesd"
                ).to("cuda", dtype=torch.float16)
        except Exception as e:
            logger.warning(f"Could not load Tiny VAE: {e}")

        # Load LCM-LoRA if enabled
        if self.params.get("use_lcm_lora", False):
            self._load_lcm_lora()

        # Load custom LoRAs
        lora_dict = self.params.get("lora_dict")
        if lora_dict:
            self._load_loras(lora_dict)

        # Setup ControlNets
        controlnets = self.params.get("controlnets", [])
        if controlnets:
            self._setup_controlnets(controlnets)

        # Setup IP-Adapter
        ip_adapter = self.params.get("ip_adapter", {})
        if ip_adapter.get("enabled", False):
            self._setup_ip_adapter(ip_adapter)

        # Prepare with initial prompt
        prompt = self.params.get("prompt", "")
        negative_prompt = self.params.get("negative_prompt", "")
        self.stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=self.params.get("delta", 1.0),
        )

        # Apply TensorRT acceleration if requested
        if acceleration == "tensorrt":
            self._apply_tensorrt()

        # Enable similar image filter if configured
        if self.params.get("enable_similar_image_filter", False):
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold=self.params.get(
                    "similar_image_filter_threshold", 0.98
                ),
                similar_image_filter_max_skip_frame=self.params.get(
                    "similar_image_filter_max_skip_frame", 10
                ),
            )

        # Warmup
        logger.info("Warming up pipeline...")
        dummy_image = Image.new("RGB", (width, height), color="black")
        for _ in range(3):
            self.stream(dummy_image)

        logger.info("Pipeline warmup complete")

    def _load_lcm_lora(self) -> None:
        """Load LCM-LoRA for the model."""
        try:
            if self.is_sdxl:
                self.stream.load_lcm_lora(
                    pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdxl"
                )
            else:
                self.stream.load_lcm_lora(
                    pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5"
                )
            self.stream.fuse_lora()
            logger.info("LCM-LoRA loaded and fused")
        except Exception as e:
            logger.warning(f"Could not load LCM-LoRA: {e}")

    def _load_loras(self, lora_dict: dict[str, float]) -> None:
        """Load custom LoRAs."""
        for lora_path, weight in lora_dict.items():
            try:
                self.pipe.load_lora_weights(lora_path)
                logger.info(f"Loaded LoRA: {lora_path} (weight: {weight})")
            except Exception as e:
                logger.warning(f"Could not load LoRA {lora_path}: {e}")

    def _setup_controlnets(self, controlnets: list[dict]) -> None:
        """Setup ControlNet models."""
        import torch
        from diffusers import ControlNetModel

        enabled_controlnets = [cn for cn in controlnets if cn.get("enabled", True)]
        if not enabled_controlnets:
            return

        logger.info(f"Setting up {len(enabled_controlnets)} ControlNets...")

        for i, cn_config in enumerate(enabled_controlnets):
            try:
                model_id = cn_config["model_id"]
                logger.info(f"Loading ControlNet: {model_id}")

                # Load ControlNet model
                if self.is_sdxl:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    ).to("cuda")
                else:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                    ).to("cuda")

                # Add to stream
                self.stream.add_controlnet(
                    controlnet,
                    conditioning_scale=cn_config.get("conditioning_scale", 0.5),
                )

                logger.info(f"ControlNet {i} loaded: {model_id}")
            except Exception as e:
                logger.warning(f"Could not load ControlNet {cn_config['model_id']}: {e}")

    def _setup_ip_adapter(self, ip_adapter_config: dict) -> None:
        """Setup IP-Adapter."""
        try:
            scale = ip_adapter_config.get("scale", 0.5)

            if self.is_sdxl:
                self.stream.load_ip_adapter(
                    pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                )
            else:
                self.stream.load_ip_adapter(
                    pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                )

            self.stream.set_ip_adapter_scale(scale)
            logger.info(f"IP-Adapter loaded (scale: {scale})")
        except Exception as e:
            logger.warning(f"Could not load IP-Adapter: {e}")

    def _apply_tensorrt(self) -> None:
        """Apply TensorRT acceleration."""
        try:
            from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

            logger.info("Applying TensorRT acceleration...")
            self.stream = accelerate_with_tensorrt(
                self.stream,
                "engines",  # Engine cache directory
                max_batch_size=2,
            )
            logger.info("TensorRT acceleration applied")
        except Exception as e:
            logger.warning(f"Could not apply TensorRT: {e}. Running without acceleration.")

    async def process_frame(self, frame: np.ndarray | Image.Image) -> np.ndarray:
        """Process a single frame through the pipeline."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")

        # Convert numpy to PIL if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._process_sync, frame)

        return output

    def _process_sync(self, frame: Image.Image) -> np.ndarray:
        """Synchronous frame processing."""
        # Process through StreamDiffusion
        output_image = self.stream(frame)

        # Convert to numpy array
        if isinstance(output_image, Image.Image):
            return np.array(output_image)
        return output_image

    async def update_params(self, updates: dict[str, Any]) -> None:
        """Update parameters that support hot reload."""
        if not self._initialized:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_params_sync, updates)

    def _update_params_sync(self, updates: dict[str, Any]) -> None:
        """Synchronous parameter update."""
        # Update prompt
        if "prompt" in updates or "negative_prompt" in updates:
            prompt = updates.get("prompt", self.params.get("prompt", ""))
            negative_prompt = updates.get(
                "negative_prompt", self.params.get("negative_prompt", "")
            )
            self.stream.update_prompt(prompt, negative_prompt)
            logger.info("Updated prompts")

        # Update guidance_scale
        if "guidance_scale" in updates:
            self.stream.guidance_scale = updates["guidance_scale"]
            logger.info(f"Updated guidance_scale: {updates['guidance_scale']}")

        # Update delta
        if "delta" in updates:
            self.stream.delta = updates["delta"]
            logger.info(f"Updated delta: {updates['delta']}")

        # Update seed
        if "seed" in updates:
            import torch
            torch.manual_seed(updates["seed"])
            logger.info(f"Updated seed: {updates['seed']}")

        # Update t_index_list (if supported)
        if "t_index_list" in updates:
            try:
                self.stream.t_index_list = updates["t_index_list"]
                logger.info(f"Updated t_index_list: {updates['t_index_list']}")
            except AttributeError:
                logger.warning("t_index_list update not supported on this stream")

        # Update params dict
        for key, value in updates.items():
            self.params[key] = value

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return

        logger.info("Cleaning up pipeline...")

        try:
            import torch

            # Clear stream
            self.stream = None
            self.pipe = None

            # Free GPU memory
            torch.cuda.empty_cache()

            logger.info("Pipeline cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

        self._initialized = False
