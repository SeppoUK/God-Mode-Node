import torch
import folder_paths
import nodes
import comfy

class GodModeInfinity:
    @classmethod
    def INPUT_TYPES(cls):
        yolo_models = folder_paths.get_filename_list("ultralytics")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "aspect_ratio": (["1:1 (Square)", "2:3 (Portrait)", "3:2 (Landscape)", "9:16 (Phone)", "16:9 (Cinema)", "Custom"], {"default": "1:1 (Square)"}),
                "positive": ("STRING", {"multiline": True, "default": "1girl, fashion photography"}),
                "negative": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 35}),
                "cfg": ("FLOAT", {"default": 6.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0}),
                "face_fix": (["Enabled", "Disabled"], {"default": "Enabled"}),
                "face_model": (yolo_models, ),
                "face_denoise": ("FLOAT", {"default": 0.6}),
                "hires_fix": (["Enabled", "Disabled"], {"default": "Enabled"}),
                "upscale_by": ("FLOAT", {"default": 1.5}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "output_comparison": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora_strength": ("FLOAT", {"default": 1.0}),
                "upscale_model_name": (["None"] + folder_paths.get_filename_list("upscale_models"), ),
                # New Optional Inputs
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
            }
        }

    # Added more return types for external piping
    RETURN_TYPES = ("IMAGE", "IMAGE", "MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MAIN_IMAGE", "ORIGINAL_IMAGE", "MODEL", "CLIP", "VAE", "FILE_PREFIX")
    FUNCTION = "execute"
    CATEGORY = "GodMode"

    def execute(self, ckpt_name, aspect_ratio, positive, negative, seed, steps, cfg, sampler_name, scheduler, 
                denoise, face_fix, face_model, face_denoise, hires_fix, upscale_by, brightness, contrast,
                output_comparison, width=1024, height=1024, lora_name="None", lora_strength=1.0, 
                upscale_model_name="None", optional_vae=None, optional_clip=None):

        # 1. Dimensions logic
        ratio_map = {"1:1 (Square)": (1024, 1024), "2:3 (Portrait)": (832, 1216), "3:2 (Landscape)": (1216, 832), "9:16 (Phone)": (704, 1280), "16:9 (Cinema)": (1280, 704)}
        final_w, final_h = ratio_map.get(aspect_ratio, (width, height))

        # 2. Setup Models (Use optional inputs if provided)
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        
        target_vae = optional_vae if optional_vae is not None else vae
        target_clip = optional_clip if optional_clip is not None else clip

        if lora_name != "None":
            model, target_clip = nodes.LoraLoader().load_lora(model, target_clip, lora_name, lora_strength, lora_strength)

        # 3. Base Generation
        c_pos = nodes.CLIPTextEncode().encode(target_clip, positive)[0]
        c_neg = nodes.CLIPTextEncode().encode(target_clip, negative)[0]
        latent = nodes.EmptyLatentImage().generate(final_w, final_h, 1)[0]
        
        samples = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, c_pos, c_neg, latent, denoise)[0]
        comfy.model_management.soft_empty_cache()

        # 4. Hires Fix
        if hires_fix == "Enabled":
            samples = nodes.LatentUpscaleBy().upscale(samples, "bilinear", upscale_by)[0]
            samples = nodes.KSampler().sample(model, seed + 1, steps, cfg, sampler_name, scheduler, c_pos, c_neg, samples, 0.45)[0]

        image = nodes.VAEDecode().decode(target_vae, samples)[0]
        
        if brightness != 1.0 or contrast != 1.0:
            image = torch.clamp(image * contrast + (brightness - 1.0), 0.0, 1.0)

        original_image = image.clone()

        # 5. Face Fix
        if face_fix == "Enabled" and face_model != "None":
            try:
                detector_cls = nodes.NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]
                bbox_detector = detector_cls().doit(model_name=face_model)[0]
                
                detailer_cls = nodes.NODE_CLASS_MAPPINGS["FaceDetailer"]
                detailed = detailer_cls().doit(
                    image=image, model=model, clip=target_clip, vae=target_vae, 
                    guide_size=384, guide_size_for=True, max_size=1024, seed=seed, 
                    steps=20, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, 
                    denoise=face_denoise, feather=5, noise_mask=True, force_inpaint=True, 
                    drop_size=10, wildcard="", cycle=1, bbox_detector=bbox_detector,
                    positive=c_pos, negative=c_neg,
                    bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0,
                    sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
                    sam_bbox_expansion=0, sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative="False"
                )
                image = detailed[0]
            except Exception as e:
                print(f"!!! GodMode Face Fix Error: {str(e)}")

        # 6. Final Logic
        main_output = image
        if output_comparison == "True":
            main_output = torch.cat((original_image, image), dim=2)

        return (main_output, original_image, model, target_clip, target_vae, f"GM_{seed}")