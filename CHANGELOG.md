# Changelog

All notable changes to the **GodModeInfinity** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [1.2.0] - 2026-08-07

### Added
- **Refactored Code Structure**: 
  - Split `execute` method into smaller, modular methods (`_load_models`, `_apply_lora`, `_generate_base`, etc.) for better maintainability.
  - Added constants for magic strings (`ENABLED`, `DISABLED`, `NONE`) and FaceDetailer defaults.
- **Enhanced Input Validation**:
  - Added path sanitization to prevent directory traversal attacks.
  - Added warnings for non-optimal dimensions (not multiples of 64).
- **Improved Error Handling**:
  - Narrowed exception handling to catch specific errors (e.g., `ValueError`, `RuntimeError`).
  - Added dependency validation for `ComfyUI-Impact-Pack` at runtime.
  - User-friendly error messages for missing dependencies or invalid inputs.
- **New Aspect Ratio Presets**:
  - Added `4:5 (Instagram)`, `5:4 (Instagram Portrait)`, and `1:2 (Wide)`.
- **Tooltips for All Inputs**:
  - Added descriptive tooltips for every input parameter in the node UI.
- **GPU Memory Management**:
  - Added `torch.cuda.empty_cache()` calls to clear GPU memory aggressively.
- **Logging Improvements**:
  - Used appropriate log levels (`debug`, `info`, `warning`, `error`) for better log filtering.
- **Type Safety**:
  - Added `Optional` and `Literal` type hints for better IDE support.

### Changed
- **Code Organization**:
  - Extracted hardcoded values (e.g., FaceDetailer parameters) into class-level constants.
  - Improved docstrings with examples and clearer descriptions.
- **Default Values**:
  - Set `face_fix` and `hires_fix` to `"Enabled"` by default for better out-of-the-box experience.
  - Set `face_model` to `"None"` by default to avoid errors when face fix is disabled.
- **Session Tracking**:
  - Refactored session stats updates into a helper method (`_update_session_stats`).

### Fixed
- **Memory Leaks**:
  - Added `torch.cuda.empty_cache()` to complement `soft_empty_cache()` for better GPU memory management.
- **Silent Failures**:
  - Face fix failures now log errors and return the original image with a `False` success flag.
- **Input Sanitization**:
  - Added `_sanitize_path` to prevent path traversal in file inputs.

### Deprecated
- None

### Removed
- None

---

## [1.1.0] - 2026-08-06

### Added
- Session Tracking System: Comprehensive usage statistics tracking.
- Progress Callbacks: Optional `progress_callback` input for real-time progress tracking.
- Input Validation: Enhanced parameter validation with explicit min/max bounds.
- Type Hints: Comprehensive type annotations throughout the codebase.
- Comprehensive Docstrings: Full documentation for class, methods, and key variables.
- Logging System: Replaced all `print()` statements with proper Python logging.

### Changed
- Code Structure: Refactored `execute()` method into well-organized sections.
- Parameter Defaults: Adjusted several default values for better out-of-box experience.
- Cache Management: Optimized memory management with strategic `soft_empty_cache()` calls.

### Fixed
- Face Fix Error Handling: Added try/except block to prevent crashes.
- Input Validation: Fixed potential issues with out-of-range parameter values.

---

## [1.0.0] - 2026-04-08

### Added
- Initial release of GodModeInfinity node.
- All-in-one image generation workflow.
- Aspect ratio presets (1:1, 2:3, 3:2, 9:16, 16:9).
- Built-in hires fix with latent upscaling.
- Automated face detailing with Ultralytics YOLO.
- Color correction (brightness/contrast).

---

## Migration Guide

### Breaking Changes
- None – Full backward compatibility maintained.

### New Features to Utilize
1. **Progress Tracking**: Connect the `progress_callback` input to a progress node for real-time updates.
2. **Session Statistics**: Use the `_session_stats` class variable to monitor usage patterns.
3. **Logging**: Check ComfyUI logs for detailed generation information.
4. **Validation**: Enjoy safer parameter ranges that prevent invalid configurations.

### Recommended Settings
- **Steps**: 35-50 for most use cases (range: 1-1000).
- **CFG**: 6.0-8.0 for balanced quality/speed (range: 0.1-20.0).
- **Upscale by**: 2.0 for good detail addition (range: 1.0-8.0).
- **Face denoise**: 0.6 for natural-looking faces (range: 0.0-1.5).