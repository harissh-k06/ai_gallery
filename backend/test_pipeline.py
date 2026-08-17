import asyncio
import base64
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure backend root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    pillow_heif.register_avif_opener()
except Exception:
    pass

VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".gif"
}

from app.services.modal_runner import app, run_gallery_pipeline


def load_test_images(folder_path: str = "app/test-imgs") -> list[dict[str, Any]]:
    """Scan and load real images from test directory, downscaling large images to max 1600px."""
    dir_path = Path(folder_path)
    if not dir_path.is_absolute() and not dir_path.exists():
        backend_dir = Path(__file__).resolve().parent
        candidate = backend_dir / folder_path
        if candidate.exists() or (backend_dir / "app").exists():
            dir_path = candidate

    dir_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"[TestRunner] No images found in {folder_path}. Drop photos into that folder to test.")
        return []

    images_payload: list[dict[str, Any]] = []
    print(f"[TestRunner] Found {len(image_files)} image(s) in '{dir_path}'. Ingesting...")

    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    rgba_img = img.convert("RGBA")
                    rgb_img = Image.new("RGB", rgba_img.size, (255, 255, 255))
                    rgb_img.paste(rgba_img, mask=rgba_img.split()[3])
                elif img.mode != "RGB":
                    rgb_img = img.convert("RGB")
                else:
                    rgb_img = img.copy()

                # Downscale large images so max dimension <= 1600px (preserving aspect ratio)
                rgb_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=90)
                base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

                images_payload.append({
                    "id": file_path.stem,
                    "base64": base64_str,
                    "metadata": {
                        "folder": file_path.parent.name,
                        "filename": file_path.name,
                        "timestamp": mod_time,
                    },
                })
                print(f"  + Loaded: {file_path.name} ({rgb_img.width}x{rgb_img.height})")
        except Exception as e:
            print(f"  - Warning: Failed to load {file_path.name}: {e}")

    return images_payload


async def _async_main():
    print("=" * 70)
    print(" AI GALLERY: DIRECT MODAL GPU TEST RUNNER")
    print("=" * 70)

    # 1. Ingest local test images
    test_images = load_test_images("app/test-imgs")
    if not test_images:
        return

    # 2. Pipeline Configuration
    default_config = {
        "dedup_cosine_threshold": 0.96,
        "cluster_eps": 0.28,
        "vlm_max_dynamic_pixels": 784 * 28 * 28,
        "dense_caption_density": "concise",
        "extract_ocr_bboxes": False,
    }

    print(f"\n[TestRunner] Dispatching {len(test_images)} image(s) to Modal GPU Vision Pipeline...")
    pipeline_result = await run_gallery_pipeline(
        images=test_images,
        config=default_config,
    )

    # 3. Output results
    print("\n" + "=" * 70)
    print(" MODAL PIPELINE EXECUTION RESULT")
    print("=" * 70)
    print(json.dumps(pipeline_result, indent=2))
    print("=" * 70)
    print(f"\nTest execution finished in {pipeline_result.get('metrics', {}).get('execution_time_sec', 'N/A')}s.")


@app.local_entrypoint()
def main():
    """CLI entrypoint for `modal run test_pipeline.py` or direct python execution."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
