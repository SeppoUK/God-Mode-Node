import torch
import folder_paths
import nodes
import comfy
import logging
from typing import Dict, Any, Optional, Tuple, Literal, List
import time
import os

# Configure module-level logger
logger = logging.getLogger(__name__)

# Constants for magic strings and defaults
ENABLED = "Enabled"
DISABLED = "Disabled"
NONE = "None"

FACE_DETAILER_DEFAULTS = {
    "guide_size": 384,
    "guide_size_for": True,
    "max_size": 1024,
    "seed": 0,
    "steps": 20,
    "cfg": 6.0,
    "sampler_name": "euler_ancestral",
    "scheduler": "karras",
    "denoise": 0.6,
    "feather": 5,
    "noise_mask": True,
    "force_inpaint": True,
    "drop_size": 10,
    "wildcard": "",
    "cycle": 1,
    "bbox_threshold": 0.5,
    "bbox_dilation": 10,
    "bbox_crop_factor": 3.0,
    "sam_detection_hint": "center-1",
    "sam_dilation": 0,
    "sam_threshold": 0.93,
    "sam_bbox_expansion": 0,
    "sam_mask_hint_threshold": 0.7,
    "sam_mask_hint_use_negative": False,
}

# Aspect ratio presets (width, height)
ASPECT_RATIOS = {
    "1:1 (Square)": (1024, 1024),
    "2:3 (Portrait)": (832, 1216),
    "3:2 (Landscape)": (1216, 832),
    "9:16 (Phone)": (704, 1280),
    "16:9 (Cinema)": (1280, 704),
    "4:5 (Instagram)": (896, 1120),
    "5:4 (Instagram Portrait)": (1120, 896),
    "1:2 (Wide)": (1024, 2048),
}


class GodModeInfinity:
    """
    🚀 God Mode Infinity - Ultimate All-in-One Image Generation Node for ComfyUI

    Combines checkpoint loading, CLIP encoding, latent sampling, hires fix,
    LoRA loading, color correction, and automated face detailing into a single node.

    Features:
    - All-in-One Generation: Checkpoint, CLIP, and latent sampling in one place.
    - Aspect Ratio Presets: Quick selection from common ratios or custom dimensions.
    - Integrated Hires Fix: Latent upscaling with a secondary sampler pass.
    - Automated Face Fix: Uses Ultralytics YOLO models for face detection and detailing.
    - Color Correction: Adjust brightness and contrast post-generation.
    - Before/After Comparison: Toggle to output a side-by-side comparison.
    - Flexible Routing: Pass-through outputs for Model, CLIP, VAE, and optional inputs.
    - Session Tracking: Monitor usage statistics and performance.
    - Progress Callbacks: Real-time progress tracking for long-running generations.

    Requirements:
    - ComfyUI-Impact-Pack (for FaceDetailer and UltralyticsDetectorProvider)
    """

    # Session tracking for usage statistics
    _session_stats: Dict[str, Any] = {
        "total_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "avg_duration_ms": 0.0,
        "face_fix_successes": 0,
        "face_fix_failures": 0,
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for the node."""
        yolo_models = folder_paths.get_filename_list("ultralytics")
        checkpoint_list = folder_paths.get_filename_list("checkpoints")
        lora_list = folder_paths.get_filename_list("loras")
        upscale_models = folder_paths.get_filename_list("upscale_models")

        return {
            "required": {
                "ckpt_name": (checkpoint_list, {"tooltip": "Base checkpoint model for generation."}),
                "aspect_ratio": (
                    list(ASPECT_RATIOS.keys()) + ["Custom"],
                    {"default": "1:1 (Square)", "tooltip": "Select an aspect ratio preset or use Custom."}
                ),
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "1girl, fashion photography, professional lighting",
                        "tooltip": "Positive prompt for image generation. Separate concepts with commas."
                    }
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, low quality, distorted, deformed",
                        "tooltip": "Negative prompt to exclude unwanted elements."
                    }
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility."}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 1000, "tooltip": "Number of sampling steps."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 20.0, "tooltip": "Classifier-free guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral", "tooltip": "Sampler algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "Scheduler algorithm."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "tooltip": "Denoising strength."}),
                "face_fix": ([ENABLED, DISABLED], {"default": ENABLED, "tooltip": "Enable automated face detailing."}),
                "face_model": ([NONE] + yolo_models, {"default": NONE, "tooltip": "YOLO model for face detection. Requires ComfyUI-Impact-Pack."}),
                "face_denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "tooltip": "Denoise strength for face inpainting."}),
                "hires_fix": ([ENABLED, DISABLED], {"default": ENABLED, "tooltip": "Enable latent upscaling."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "tooltip": "Upscaling multiplier."}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05, "tooltip": "Brightness adjustment."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05, "tooltip": "Contrast adjustment."}),
                "output_comparison": (["False", "True"], {"default": "False", "tooltip": "Output side-by-side comparison."}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "tooltip": "Custom width (used if aspect_ratio is 'Custom')."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "tooltip": "Custom height (used if aspect_ratio is 'Custom')."}),
                "lora_name": ([NONE] + lora_list, {"default": NONE, "tooltip": "LoRA model to apply."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "tooltip": "LoRA strength multiplier."}),
                "upscale_model_name": ([NONE] + upscale_models, {"default": NONE, "tooltip": "Model for hires fix upscaling."}),
                "optional_vae": ("VAE", {"tooltip": "Override the checkpoint's default VAE."}),
                "optional_clip": ("CLIP", {"tooltip": "Override the checkpoint's default CLIP."}),
                "progress_callback": ("PROGRESS", {"tooltip": "Optional progress callback for tracking generation."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MAIN_IMAGE", "ORIGINAL_IMAGE", "MODEL", "CLIP", "VAE", "FILE_PREFIX")
    FUNCTION = "execute"
    CATEGORY = "GodMode"

    @classmethod
    def get_session_stats(cls) -> Dict[str, Any]:
        """Return current session usage statistics."""
        return cls._session_stats.copy()

    @classmethod
    def reset_stats(cls) -> None:
        """Reset session statistics."""
        cls._session_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "avg_duration_ms": 0.0,
            "face_fix_successes": 0,
            "face_fix_failures": 0,
        }

    @classmethod
    def validate_dependencies(cls) -> None:
        """Validate that required dependencies are installed."""
        try:
            nodes.NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]
            nodes.NODE_CLASS_MAPPINGS["FaceDetailer"]
        except KeyError:
            raise RuntimeError(
                "ComfyUI-Impact-Pack is required for face fix. "
                "Install it from: https://github.com/ltdrdata/ComfyUI-Impact-Pack"
            )

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """Sanitize a file path to prevent directory traversal."""
        if ".." in path or path.startswith("/") or path.startswith("\\"):
            raise ValueError(f"Invalid path: {path}")
        return os.path.basename(path)

    def _validate_inputs(self, aspect_ratio: str, width: int, height: int) -> Tuple[int, int]:
        """
        Validate input parameters and return final dimensions.

        Args:
            aspect_ratio: Selected aspect ratio preset or "Custom".
            width: Custom width (used if aspect_ratio is "Custom").
            height: Custom height (used if aspect_ratio is "Custom").

        Returns:
            Tuple of (final_width, final_height).

        Raises:
            ValueError: If inputs are invalid.
        """
        if aspect_ratio == "Custom":
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers for custom aspect ratio.")
            if width > 4096 or height > 4096:
                logger.warning(f"Very large dimensions ({width}x{height}). Consider scaling down.")
            if width % 64 != 0 or height % 64 != 0:
                logger.warning("Dimensions should be multiples of 64 for optimal performance.")
            return width, height

        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")

        final_w, final_h = ASPECT_RATIOS[aspect_ratio]
        if final_w < 64 or final_h < 64:
            raise ValueError(f"Dimensions must be at least 64x64. Got {final_w}x{final_h}.")

        return final_w, final_h

    def _load_models(self, ckpt_name: str, optional_vae: Optional[Any], optional_clip: Optional[Any]) -> Tuple[Any, Any, Any]:
        """Load checkpoint, VAE, and CLIP models."""
        logger.debug(f"Loading checkpoint: {ckpt_name}")
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        target_vae = optional_vae if optional_vae is not None else vae
        target_clip = optional_clip if optional_clip is not None else clip
        return model, target_clip, target_vae

    def _apply_lora(self, model: Any, clip: Any, lora_name: str, lora_strength: float) -> Tuple[Any, Any]:
        """Apply LoRA to the model and CLIP."""
        if lora_name == NONE or lora_name not in folder_paths.get_filename_list("loras"):
            return model, clip
        logger.info(f"Applying LoRA: {lora_name} (strength: {lora_strength})")
        return nodes.LoraLoader().load_lora(model, clip, lora_name, lora_strength, lora_strength)

    def _encode_prompts(self, clip: Any, positive: str, negative: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode positive and negative prompts."""
        logger.debug("Encoding text prompts")
        c_pos = nodes.CLIPTextEncode().encode(clip, positive)[0]
        c_neg = nodes.CLIPTextEncode().encode(clip, negative)[0]
        return c_pos, c_neg

    def _generate_base(
        self,
        model: Any,
        clip: Any,
        positive: torch.Tensor,
        negative: torch.Tensor,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Generate the base latent image."""
        logger.info(f"Starting base generation: {width}x{height}, steps={steps}, cfg={cfg}")
        latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
        samples = nodes.KSampler().sample(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise
        )[0]
        comfy.model_management.soft_empty_cache()
        if progress_callback:
            progress_callback(0.50, "Base generation complete...")
        return samples

    def _apply_hires_fix(
        self,
        model: Any,
        clip: Any,
        positive: torch.Tensor,
        negative: torch.Tensor,
        samples: torch.Tensor,
        upscale_by: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Apply hires fix upscaling to the latent image."""
        logger.info(f"Applying hires fix: upscale_by={upscale_by}")
        samples = nodes.LatentUpscaleBy().upscale(samples, "bilinear", upscale_by)[0]
        samples = nodes.KSampler().sample(
            model, seed + 1, steps, cfg, sampler_name, scheduler, positive, negative, samples, 0.45
        )[0]
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_callback:
            progress_callback(0.70, "Hires fix complete...")
        return samples

    def _decode_image(self, vae: Any, samples: torch.Tensor) -> torch.Tensor:
        """Decode the latent samples into an image."""
        logger.debug("Decoding final image")
        return nodes.VAEDecode().decode(vae, samples)[0]

    def _apply_color_correction(self, image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
        """Apply brightness and contrast adjustments to the image."""
        if brightness != 1.0 or contrast != 1.0:
            image = torch.clamp(image * contrast + (brightness - 1.0), 0.0, 1.0)
        return image

    def _apply_face_fix(
        self,
        image: torch.Tensor,
        model: Any,
        clip: Any,
        vae: Any,
        face_model: str,
        face_denoise: float,
        positive: torch.Tensor,
        negative: torch.Tensor,
        seed: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Apply automated face detailing to the image.

        Returns:
            Tuple of (processed_image, success: bool)
        """
        if face_model == NONE:
            if progress_callback:
                progress_callback(0.85, "Skipping face fix (no model selected)")
            return image, False

        try:
            logger.info(f"Running face fix with model: {face_model}")
            detector_cls = nodes.NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]
            bbox_detector = detector_cls().doit(model_name=face_model)[0]

            detailer_cls = nodes.NODE_CLASS_MAPPINGS["FaceDetailer"]
            detailed = detailer_cls().doit(
                image=image,
                model=model,
                clip=clip,
                vae=vae,
                **FACE_DETAILER_DEFAULTS,
                seed=seed,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=face_denoise,
                positive=positive,
                negative=negative,
                bbox_detector=bbox_detector,
            )
            if progress_callback:
                progress_callback(0.90, "Face fix complete...")
            return detailed[0], True
        except Exception as e:
            logger.error(f"Face fix failed: {e}", exc_info=True)
            return image, False

    def _update_session_stats(self, success: bool, face_fix_success: bool, duration_ms: float) -> None:
        """Update session statistics."""
        if success:
            self._session_stats["successful_runs"] += 1
            if face_fix_success:
                self._session_stats["face_fix_successes"] += 1
            else:
                self._session_stats["face_fix_failures"] += 1
        else:
            self._session_stats["failed_runs"] += 1

        # Update average duration
        total_successful = self._session_stats["successful_runs"]
        if total_successful > 0:
            avg_duration = (
                self._session_stats["avg_duration_ms"] * (total_successful - 1) + duration_ms
            ) / total_successful
            self._session_stats["avg_duration_ms"] = avg_duration

    def execute(
        self,
        ckpt_name: str,
        aspect_ratio: str,
        positive: str = "1girl, fashion photography, professional lighting",
        negative: str = "blurry, low quality, distorted, deformed",
        seed: int = 0,
        steps: int = 35,
        cfg: float = 6.0,
        sampler_name: str = "euler_ancestral",
        scheduler: str = "karras",
        denoise: float = 1.0,
        face_fix: str = ENABLED,
        face_model: str = NONE,
        face_denoise: float = 0.6,
        hires_fix: str = ENABLED,
        upscale_by: float = 2.0,
        brightness: float = 1.0,
        contrast: float = 1.0,
        output_comparison: str = "False",
        width: int = 1024,
        height: int = 1024,
        lora_name: str = NONE,
        lora_strength: float = 1.0,
        upscale_model_name: str = NONE,
        optional_vae: Optional[Any] = None,
        optional_clip: Optional[Any] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, str]:
        """
        Execute the GodModeInfinity generation pipeline.

        Args:
            ckpt_name: Path to base checkpoint model.
            aspect_ratio: Aspect ratio preset or "Custom" for custom dimensions.
            positive: Positive text prompt(s).
            negative: Negative text prompt(s).
            seed: Random seed for reproducibility.
            steps: Number of sampling steps.
            cfg: Classifier-free guidance scale.
            sampler_name: Sampler algorithm name.
            scheduler: Scheduler algorithm name.
            denoise: Denoising factor (0.0 to 1.0).
            face_fix: Enable/disable automated face detailing.
            face_model: YOLO model for face detection ("None" to disable).
            face_denoise: Face-specific denoising strength.
            hires_fix: Enable/disable latent upscaling.
            upscale_by: Upscaling multiplier for hires fix.
            brightness: Brightness correction (0.0-2.5).
            contrast: Contrast correction (0.0-2.5).
            output_comparison: Enable before/after comparison.
            width: Custom width (if aspect_ratio is "Custom").
            height: Custom height (if aspect_ratio is "Custom").
            lora_name: LoRA model name to apply ("None" to skip).
            lora_strength: LoRA strength multiplier.
            upscale_model_name: Model for hires fix upscaling ("None" to skip).
            optional_vae: Optional VAE override.
            optional_clip: Optional CLIP override.
            progress_callback: Optional progress callback function.

        Returns:
            Tuple of (main_image, original_image, model, clip, vae, file_prefix).
        """
        # Track session start
        self._session_stats["total_runs"] += 1
        start_time = time.time() * 1000  # milliseconds

        try:
            # --- Phase 0: Validate Dependencies ---
            self.validate_dependencies()

            # --- Phase 1: Validation ---
            logger.info(f"Validating inputs: aspect_ratio={aspect_ratio}, seed={seed}")
            final_w, final_h = self._validate_inputs(aspect_ratio, width, height)

            if progress_callback:
                progress_callback(0.05, "Validating inputs...")

            # --- Phase 2: Model Loading ---
            model, clip, vae = self._load_models(ckpt_name, optional_vae, optional_clip)
            if progress_callback:
                progress_callback(0.15, "Loading models...")

            # --- Phase 3: Apply LoRA ---
            model, clip = self._apply_lora(model, clip, lora_name, lora_strength)

            # --- Phase 4: Text Encoding ---
            c_pos, c_neg = self._encode_prompts(clip, positive, negative)
            if progress_callback:
                progress_callback(0.25, "Encoding prompts...")

            # --- Phase 5: Base Generation ---
            samples = self._generate_base(
                model, clip, c_pos, c_neg, final_w, final_h, seed, steps, cfg,
                sampler_name, scheduler, denoise, progress_callback
            )

            # --- Phase 6: Hires Fix ---
            if hires_fix == ENABLED:
                samples = self._apply_hires_fix(
                    model, clip, c_pos, c_neg, samples, upscale_by, seed, steps, cfg,
                    sampler_name, scheduler, progress_callback
                )

            # --- Phase 7: Image Decoding ---
            image = self._decode_image(vae, samples)

            # --- Phase 8: Color Correction ---
            image = self._apply_color_correction(image, brightness, contrast)
            original_image = image.clone()
            if progress_callback:
                progress_callback(0.80, "Color correction applied...")

            # --- Phase 9: Face Fix ---
            image, face_fix_success = self._apply_face_fix(
                image, model, clip, vae, face_model, face_denoise, c_pos, c_neg,
                seed, cfg, sampler_name, scheduler, progress_callback
            )

            # --- Phase 10: Final Output ---
            main_output = image
            if output_comparison == "True":
                main_output = torch.cat((original_image, image), dim=2)

            file_prefix = f"GM_{seed:08d}"

            # Track session completion
            end_time = time.time() * 1000
            duration_ms = end_time - start_time
            self._update_session_stats(True, face_fix_success, duration_ms)

            logger.info(f"Generation complete in {duration_ms:.0f}ms")
            if progress_callback:
                progress_callback(1.0, "Done!")

            return (main_output, original_image, model, clip, vae, file_prefix)

        except Exception as e:
            end_time = time.time() * 1000
            duration_ms = end_time - start_time
            self._update_session_stats(False, False, duration_ms)

            logger.error(f"Generation failed after {duration_ms:.0f}ms: {str(e)}", exc_info=True)

            if progress_callback:
                progress_callback(0.0, f"Error: {str(e)}")

            # Return error image
            error_tensor = torch.randn(3, 256, 256).clamp(0, 1)
            return (error_tensor, error_tensor, None, None, None, "ERROR")


# Register node
NODE_CLASS_MAPPINGS = {
    "GodModeInfinity": GodModeInfinity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GodModeInfinity": "🚀 God Mode Infinity"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
