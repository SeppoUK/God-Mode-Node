GodModeInfinity is an ultimate, all-in-one generation node for ComfyUI. It condenses an entire complex image generation workflow—including base generation, hires fixing, LoRA loading, color correction, and automated face detailing—into a single, easy-to-use node.

Whether you are looking to declutter your workspace or want a powerful "plug-and-play" solution for high-quality character generation, GodModeInfinity handles the heavy lifting while still giving you access to the models and outputs for downstream piping.

✨ Key Features
All-in-One Generation: Handles Checkpoint loading, CLIP text encoding, and latent generation in one place.

Aspect Ratio Presets: Quickly select from common aspect ratios (Square, Portrait, Landscape, Phone, Cinema) or use custom dimensions.

Integrated Hires Fix: Built-in latent upscaling (bilinear) with a secondary sampler pass to add detail and scale up your initial generation.

Automated Face Fix: Integrates FaceDetailer capabilities using Ultralytics YOLO models to automatically detect and refine faces in your generations.

Color Correction: Built-in brightness and contrast sliders to tweak your final decoded image without needing extra post-processing nodes.

Before/After Comparison: A handy toggle that outputs a side-by-side comparison of your original generation alongside the final detailed/color-corrected version.

Flexible Routing: Pass-through outputs for your Model, CLIP, and VAE, plus optional inputs for custom VAE or CLIP models, allowing you to easily hook this node into larger, more complex workflows.

📥 Inputs
Required:

ckpt_name: Your base checkpoint model.

aspect_ratio: Quick presets for image dimensions.

positive / negative: Multiline text prompts.

seed, steps, cfg, sampler_name, scheduler, denoise: Standard sampling controls.

face_fix: Toggle the automated face detailing.

face_model: Select your Ultralytics YOLO model for face detection.

face_denoise: Denoise strength specifically for the face inpainting.

hires_fix: Toggle the latent upscaler.

upscale_by: The multiplier for the hires fix upscaling.

brightness / contrast: Image adjustment sliders.

output_comparison: Outputs a side-by-side image if set to True.

Optional:

width / height: Used if Aspect Ratio is set to "Custom".

lora_name / lora_strength: Easily inject a LoRA directly into the node.

optional_vae / optional_clip: Override the checkpoint's default VAE or CLIP with external nodes.

📤 Outputs
MAIN_IMAGE: The final, processed image (or the side-by-side comparison if enabled).

ORIGINAL_IMAGE: The raw decoded image before Face Fix or Color Correction.

MODEL / CLIP / VAE: The loaded models, ready to be piped to other nodes.

FILE_PREFIX: A generated string (e.g., GM_12345) for easy file saving.

⚠️ Requirements
Because this node integrates advanced face detailing, you must have the following custom node suite installed in your ComfyUI environment for the face fix feature to work:

ComfyUI-Impact-Pack (Provides the FaceDetailer and UltralyticsDetectorProvider classes utilized by this node).
