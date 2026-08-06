import torch
import folder_paths
import nodes
import comfy
import logging
from typing import Dict, Any, Optional, Tuple
import time

# Configure module-level logger
logger = logging.getLogger(__name__)


class GodModeInfinity:
    """
    🚀 God Mode Infinity - Ultimate Image Generation Node for ComfyUI

    An all-in-one generation node that condenses an entire complex image
    generation workflow into a single, easy-to-use node. Handles base
    generation, hires fixing, LoRA loading, color correction, and automated
    face detailing with minimal setup.

    Features:
    - All-in-One Generation: Checkpoint loading, CLIP encoding, latent sampling
    - Aspect Ratio Presets: Quick selection from common aspect ratios or custom dimensions
    - Integrated Hires Fix: Built-in latent upscaling with secondary sampler pass
    - Automated Face Fix: Integrates FaceDetailer capabilities using Ultralytics YOLO
    - Color Correction: Brightness/contrast controls for final image adjustment
    - Before/After Comparison: Side-by-side comparison output toggle
    - Flexible Routing: Pass-through outputs for Model, CLIP, VAE, plus optional inputs
    - Session Tracking: Usage statistics and performance monitoring
    - Progress Callbacks: Optional progress tracking for long-running generations

    Requirements:
    - ComfyUI-Impact-Pack (for FaceDetailer and UltralyticsDetectorProvider)
    """

    # Aspect ratio presets (width, height)
    ASPECT_RATIOS = {
        "1:1 (Square)": (1024, 1024),
        "2:3 (Portrait)": (832, 1216),
        "3:2 (Landscape)": (1216, 832),
        "9:16 (Phone)": (704, 1280),
        "16:9 (Cinema)": (1280, 704),
    }

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
    def INPUT_TYPES(cls):
        """Return input types for the node."""
        yolo_models = folder_paths.get_filename_list("ultralytics")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "aspect_ratio": (
                    list(cls.ASPECT_RATIOS.keys()) + ["Custom"],
                    {"default": "1:1 (Square)"}
                ),
                "positive": ("STRING", {"multiline": True, "default": "1girl, fashion photography, professional lighting"}),
                "negative": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted, deformed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 20.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "face_fix": (["Enabled", "Disabled"], {"default": "Enabled"}),
                "face_model": (["None"] + yolo_models, {"default": "None"}),
                "face_denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5}),
                "hires_fix": (["Enabled", "Disabled"], {"default": "Enabled"}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05}),
                "output_comparison": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0}),
                "upscale_model_name": (["None"] + folder_paths.get_filename_list("upscale_models"), {"default": "None"}),
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "progress_callback": ("PROGRESS",),
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

    def _validate_inputs(self, aspect_ratio: str, width: int, height: int) -> Tuple[int, int]:
        """
        Validate input parameters and return final dimensions.

        Args:
            aspect_ratio: Selected aspect ratio preset or "Custom"
            width: Custom width (used if aspect_ratio is "Custom")
            height: Custom height (used if aspect_ratio is "Custom")

        Returns:
            Tuple of (final_width, final_height)

        Raises:
            ValueError: If inputs are invalid
        """
        if aspect_ratio == "Custom":
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers for custom aspect ratio")
            if width > 4096 or height > 4096:
                logger.warning(f"Very large dimensions ({width}x{height}). Consider scaling down.")
            return width, height

        if aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")

        final_w, final_h = self.ASPECT_RATIOS[aspect_ratio]

        if final_w < 64 or final_h < 64:
            raise ValueError(f"Dimensions must be at least 64x64. Got {final_w}x{final_h}")

        return final_w, final_h

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
        face_fix: str = "Enabled",
        face_model: str = "None",
        face_denoise: float = 0.6,
        hires_fix: str = "Enabled",
        upscale_by: float = 2.0,
        brightness: float = 1.0,
        contrast: float = 1.0,
        output_comparison: str = "False",
        width: int = 1024,
        height: int = 1024,
        lora_name: str = "None",
        lora_strength: float = 1.0,
        upscale_model_name: str = "None",
        optional_vae=None,
        optional_clip=None,
        progress_callback=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, str]:
        """
        Execute the GodModeInfinity generation pipeline.

        Args:
            ckpt_name: Path to base checkpoint model
            aspect_ratio: Aspect ratio preset or "Custom" for custom dimensions
            positive: Positive text prompt(s)
            negative: Negative text prompt(s)
            seed: Random seed for reproducibility
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Sampler algorithm name
            scheduler: Scheduler algorithm name
            denoise: Denoising factor (0.0 to 1.0)
            face_fix: Enable/disable automated face detailing
            face_model: YOLO model for face detection ("None" to disable)
            face_denoise: Face-specific denoising strength
            hires_fix: Enable/disable latent upscaling
            upscale_by: Upscaling multiplier for hires fix
            brightness: Brightness correction (0.0-2.5)
            contrast: Contrast correction (0.0-2.5)
            output_comparison: Enable before/after comparison
            width: Custom width (if aspect_ratio is "Custom")
            height: Custom height (if aspect_ratio is "Custom")
            lora_name: LoRA model name to apply ("None" to skip)
            lora_strength: LoRA strength multiplier
            upscale_model_name: Model for hires fix upscaling ("None" to skip)
            optional_vae: Optional VAE override
            optional_clip: Optional CLIP override
            progress_callback: Optional progress callback function

        Returns:
            Tuple of (main_image, original_image, model, clip, vae, file_prefix)
        """
        # Track session start
        self._session_stats["total_runs"] += 1
        start_time = time.time() * 1000  # milliseconds

        try:
            # --- Phase 1: Validation ---
            logger.info(f"Validating inputs: aspect_ratio={aspect_ratio}, seed={seed}")
            final_w, final_h = self._validate_inputs(aspect_ratio, width, height)

            if progress_callback:
                progress_callback(0.05, "Validating inputs...")

            # --- Phase 2: Model Loading ---
            logger.info(f"Loading checkpoint: {ckpt_name}")
            model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)

            target_vae = optional_vae if optional_vae is not None else vae
            target_clip = optional_clip if optional_clip is not None else clip

            if progress_callback:
                progress_callback(0.15, "Loading models...")

            # Apply LoRA if specified
            if lora_name != "None" and lora_name in folder_paths.get_filename_list("loras"):
                logger.info(f"Applying LoRA: {lora_name} (strength: {lora_strength})")
                model, target_clip = nodes.LoraLoader().load_lora(
                    model, target_clip, lora_name, lora_strength, lora_strength
                )

            # --- Phase 3: Text Encoding ---
            logger.info("Encoding text prompts")
            c_pos = nodes.CLIPTextEncode().encode(target_clip, positive)[0]
            c_neg = nodes.CLIPTextEncode().encode(target_clip, negative)[0]

            if progress_callback:
                progress_callback(0.25, "Encoding prompts...")

            # --- Phase 4: Base Generation ---
            logger.info(f"Starting base generation: {final_w}x{final_h}, steps={steps}, cfg={cfg}")
            latent = nodes.EmptyLatentImage().generate(final_w, final_h, 1)[0]

            samples = nodes.KSampler().sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                c_pos, c_neg, latent, denoise
            )[0]

            # Clear memory after base sampling
            comfy.model_management.soft_empty_cache()

            if progress_callback:
                progress_callback(0.50, "Base generation complete...")

            # --- Phase 5: Hires Fix ---
            if hires_fix == "Enabled":
                logger.info(f"Applying hires fix: upscale_by={upscale_by}")
                samples = nodes.LatentUpscaleBy().upscale(samples, "bilinear", upscale_by)[0]
                samples = nodes.KSampler().sample(
                    model, seed + 1, steps, cfg, sampler_name, scheduler,
                    c_pos, c_neg, samples, 0.45
                )[0]
                comfy.model_management.soft_empty_cache()

                if progress_callback:
                    progress_callback(0.70, "Hires fix complete...")

            # --- Phase 6: Image Decoding ---
            logger.info("Decoding final image")
            image = nodes.VAEDecode().decode(target_vae, samples)[0]

            # Apply brightness/contrast correction
            if brightness != 1.0 or contrast != 1.0:
                image = torch.clamp(image * contrast + (brightness - 1.0), 0.0, 1.0)

            original_image = image.clone()

            if progress_callback:
                progress_callback(0.80, "Color correction applied...")

            # --- Phase 7: Face Fix ---
            face_fix_success = False
            if face_fix == "Enabled" and face_model != "None":
                try:
                    logger.info(f"Running face fix with model: {face_model}")
                    detector_cls = nodes.NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]
                    bbox_detector = detector_cls().doit(model_name=face_model)[0]

                    detailer_cls = nodes.NODE_CLASS_MAPPINGS["FaceDetailer"]
                    detailed = detailer_cls().doit(
                        image=image, model=model, clip=target_clip, vae=target_vae,
                        guide_size=384, guide_size_for=True, max_size=1024, seed=seed,
                        steps=20, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                        denoise=face_denoise, feather=5, noise_mask=True,
                        force_inpaint=True, drop_size=10, wildcard="", cycle=1,
                        bbox_detector=bbox_detector,
                        positive=c_pos, negative=c_neg,
                        bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0,
                        sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
                        sam_bbox_expansion=0, sam_mask_hint_threshold=0.7,
                        sam_mask_hint_use_negative="False"
                    )
                    image = detailed[0]
                    face_fix_success = True

                    if progress_callback:
                        progress_callback(0.90, "Face fix complete...")

                except Exception as e:
                    logger.warning(f"Face fix failed with model {face_model}: {str(e)}")
                    # Fallback to original image
                    image = original_image
                    face_fix_success = False

            # --- Phase 8: Final Output ---
            main_output = image
            if output_comparison == "True":
                main_output = torch.cat((original_image, image), dim=2)

            file_prefix = f"GM_{seed:08d}"

            # Track session completion
            end_time = time.time() * 1000
            duration_ms = end_time - start_time

            self._session_stats["successful_runs"] += 1
            if face_fix_success:
                self._session_stats["face_fix_successes"] += 1
            else:
                self._session_stats["face_fix_failures"] += 1

            # Update running average for duration
            avg_duration = (
                self._session_stats["avg_duration_ms"] *
                (self._session_stats["successful_runs"] - 1) + duration_ms
            ) / self._session_stats["successful_runs"]
            self._session_stats["avg_duration_ms"] = avg_duration

            logger.info(f"Generation complete in {duration_ms:.0f}ms")

            if progress_callback:
                progress_callback(1.0, "Done!")

            return (main_output, original_image, model, target_clip, target_vae, file_prefix)

        except Exception as e:
            end_time = time.time() * 1000
            duration_ms = end_time - start_time

            self._session_stats["failed_runs"] += 1

            # Update running average for duration
            avg_duration = (
                self._session_stats["avg_duration_ms"] *
                self._session_stats["successful_runs"] + duration_ms
            ) / max(1, self._session_stats["successful_runs"])
            self._session_stats["avg_duration_ms"] = avg_duration

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
    "GodModeInfinity": "🚀 God Mode Infinity V6.1"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
