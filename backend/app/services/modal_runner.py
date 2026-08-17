import base64
import concurrent.futures
import io
import json
import re
import time
from typing import Any

import modal
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

try:
    from app.schemas.payload import VectorEngineConfig
except ImportError:
    # Fallback definition for container / standalone execution environments
    from pydantic import BaseModel, Field

    class VectorEngineConfig(BaseModel):  # type: ignore[no-redef]
        dedup_cosine_threshold: float = Field(default=0.96, ge=0.80, le=0.99)
        cluster_eps: float = Field(default=0.28, ge=0.05, le=0.60)
        temporal_window_hours: int = Field(default=6, ge=1, le=72)
        hnsw_m: int = Field(default=16, ge=8, le=64)
        hnsw_ef_construction: int = Field(default=64, ge=16, le=256)
        vlm_max_dynamic_pixels: int = Field(default=784 * 28 * 28)
        dense_caption_density: str = Field(default="concise")
        extract_ocr_bboxes: bool = Field(default=False)
        hybrid_search_alpha: float = Field(default=0.7, ge=0.0, le=1.0)


# ============================================================================
# Modal App & Zero Cold-Start Image Specification
# ============================================================================

app = modal.App("ai-gallery-engine")


def _download_models_to_image():
    """Download and cache pipeline weights into the Modal image layer at build time."""
    from transformers import (
        AutoModel,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    siglip_id = "google/siglip-so400m-patch14-384"
    qwen_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"[Build Step] Pre-baking SigLIP weights: {siglip_id}")
    AutoProcessor.from_pretrained(siglip_id)
    AutoModel.from_pretrained(siglip_id)

    print(f"[Build Step] Pre-baking Qwen2.5-VL weights: {qwen_id}")
    AutoProcessor.from_pretrained(qwen_id)
    Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_id)


modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "torchvision",
        "pillow",
        "pillow-heif",
        "pillow-avif-plugin",
        "scikit-learn",
        "numpy",
        "qwen-vl-utils",
        "pydantic",
        "sentencepiece",
        "protobuf",
    )
    .run_function(
        _download_models_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)


# Modal 1.0+ renamed container_idle_timeout to scaledown_window
_cls_kwargs: dict[str, Any] = {
    "gpu": "A10G",
    "image": modal_image,
    "secrets": [modal.Secret.from_name("huggingface-secret")],
}
try:
    import inspect
    if "scaledown_window" in inspect.signature(app.cls).parameters:
        _cls_kwargs["scaledown_window"] = 180
    else:
        _cls_kwargs["container_idle_timeout"] = 180
except Exception:
    _cls_kwargs["scaledown_window"] = 180


@app.cls(**_cls_kwargs)
class GalleryVisionEngine:
    """
    Two-Tier GPU Vision Processing Pipeline:
      - Tier 1: High-throughput SigLIP batch embedding (google/siglip-so400m-patch14-384)
                for vector clustering, burst deduplication, and exemplar election.
      - Tier 2: Selective High-Fidelity VLM (Qwen/Qwen2.5-VL-7B-Instruct)
                for semantic scene understanding, dense tagging, and OCR extraction.
    """

    @modal.enter()
    def setup(self):
        import torch
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GalleryVisionEngine] Initializing models on device: {self.device}")

        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Tier 1: SigLIP Embedder
        self.siglip_model_id = "google/siglip-so400m-patch14-384"
        print(f"[GalleryVisionEngine] Loading Tier 1 SigLIP: {self.siglip_model_id}")
        siglip_config = AutoConfig.from_pretrained(self.siglip_model_id)
        if hasattr(siglip_config, "text_config"):
            siglip_config.text_config.bos_token_id = None
            siglip_config.text_config.eos_token_id = None
        siglip_config.bos_token_id = None
        siglip_config.eos_token_id = None

        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_id)
        self.siglip_model = AutoModel.from_pretrained(
            self.siglip_model_id,
            config=siglip_config,
            dtype=target_dtype,
        ).to(self.device).eval()

        # Tier 2: Qwen2.5-VL Vision-Language Model with Native PyTorch SDPA Attention
        self.qwen_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"[GalleryVisionEngine] Loading Tier 2 Qwen2.5-VL with SDPA: {self.qwen_model_id}")
        self.qwen_processor = AutoProcessor.from_pretrained(self.qwen_model_id)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_id,
            dtype=target_dtype,
            attn_implementation="sdpa",
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

        print("[GalleryVisionEngine] All pipeline models initialized successfully.")

    @modal.method()
    def process_gallery(self, images_payload: list[dict[str, Any]], config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the full two-tier vision analysis pipeline on a batch of gallery images.

        Args:
            images_payload: List of dicts containing image data (base64/bytes), id, and metadata.
            config_dict: Engine parameters (dedup_cosine_threshold, cluster_eps, etc.).

        Returns:
            Structured JSON dict with clusters, deduplicated groups, documents/OCR, and metrics.
        """
        import numpy as np
        import torch
        from sklearn.cluster import DBSCAN

        start_time = time.time()
        total_ingested = len(images_payload)

        if total_ingested == 0:
            return {
                "clusters": [],
                "duplicates": [],
                "documents": [],
                "metrics": {
                    "total_ingested": 0,
                    "exemplars_processed_vlm": 0,
                    "documents_processed_vlm": 0,
                    "execution_time_sec": 0.0,
                },
            }

        # ----------------------------------------------------------------------
        # Step A: Parallel Image Ingestion & Multi-Thread Base64 Decoding
        # ----------------------------------------------------------------------
        def _decode_item(idx_item: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
            idx, item = idx_item
            image_id = str(item.get("id") or item.get("image_id") or item.get("key") or f"img_{idx}")
            meta = item.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}

            # Populate metadata fields if passed at top level
            for k in ("folder", "folder_name", "timestamp", "filename", "path"):
                if k in item and k not in meta:
                    meta[k] = item[k]

            raw_data = item.get("base64") or item.get("data") or item.get("image_bytes") or item.get("bytes")
            pil_img = self._decode_image(raw_data)

            if pil_img is not None:
                return {
                    "id": image_id,
                    "image": pil_img,
                    "metadata": meta,
                    "original_index": idx,
                }
            else:
                print(f"[GalleryVisionEngine] Warning: Failed to decode image for ID: {image_id}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            decode_results = list(executor.map(_decode_item, enumerate(images_payload)))

        decoded_records: list[dict[str, Any]] = [r for r in decode_results if r is not None]

        if not decoded_records:
            return {
                "clusters": [],
                "duplicates": [],
                "documents": [],
                "metrics": {
                    "total_ingested": total_ingested,
                    "exemplars_processed_vlm": 0,
                    "documents_processed_vlm": 0,
                    "execution_time_sec": round(time.time() - start_time, 2),
                },
            }

        num_valid = len(decoded_records)

        # ----------------------------------------------------------------------
        # Step B: Tier 1 - SigLIP Batch Embedding & Cosine Similarity (Batch Size 64)
        # ----------------------------------------------------------------------
        batch_size = 64
        embeddings_list: list[np.ndarray] = []

        for i in range(0, num_valid, batch_size):
            batch_slice = decoded_records[i : i + batch_size]
            batch_images = [r["image"] for r in batch_slice]

            inputs = self.siglip_processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.siglip_model.get_image_features(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                elif isinstance(outputs, (tuple, list)):
                    features = outputs[0]
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    features = outputs

                # Normalize embeddings to unit length (L2 norm) -> 1152-dim vectors
                norm_features = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings_list.append(norm_features.cpu().to(torch.float32).numpy())

        embeddings = np.concatenate(embeddings_list, axis=0)  # Shape: (num_valid, 1152)

        # Compute full Cosine Similarity matrix (dot product of L2-normalized vectors)
        sim_matrix = np.dot(embeddings, embeddings.T)
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

        # ----------------------------------------------------------------------
        # Step C: Deduplication & DBSCAN Clustering
        # ----------------------------------------------------------------------
        dedup_threshold = float(config_dict.get("dedup_cosine_threshold", 0.96))
        cluster_eps = float(config_dict.get("cluster_eps", 0.28))

        is_duplicate = [False] * num_valid
        duplicate_map: dict[int, list[tuple[int, float]]] = {}  # primary_idx -> list of (dup_idx, sim)

        # Identify burst shots and near-duplicates
        for i in range(num_valid):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, num_valid):
                if is_duplicate[j]:
                    continue
                score = float(sim_matrix[i, j])
                if score >= dedup_threshold:
                    is_duplicate[j] = True
                    duplicate_map.setdefault(i, []).append((j, score))

        duplicates_output: list[dict[str, Any]] = []
        for prim_idx, dup_list in duplicate_map.items():
            duplicates_output.append({
                "primary_id": decoded_records[prim_idx]["id"],
                "duplicate_ids": [decoded_records[d_idx]["id"] for d_idx, _ in dup_list],
                "similarity_score": round(float(np.mean([s for _, s in dup_list])), 4),
            })

        # Retain non-duplicate indices for visual clustering
        non_dup_indices = [i for i in range(num_valid) if not is_duplicate[i]]

        clusters_output: list[dict[str, Any]] = []
        exemplar_indices: list[int] = []

        if non_dup_indices:
            non_dup_embeddings = embeddings[non_dup_indices]

            # DBSCAN clustering on cosine distance (1.0 - cosine_similarity)
            dbscan = DBSCAN(eps=cluster_eps, min_samples=1, metric="cosine").fit(non_dup_embeddings)
            labels = dbscan.labels_

            # Group indices by cluster
            clusters_map: dict[int, list[int]] = {}
            for pos, record_idx in enumerate(non_dup_indices):
                label = int(labels[pos])
                clusters_map.setdefault(label, []).append(record_idx)

            # Elect 1 representative exemplar per cluster
            for cluster_id, member_record_indices in clusters_map.items():
                if len(member_record_indices) == 1:
                    exemplar_idx = member_record_indices[0]
                else:
                    # Select member with highest average cosine similarity to cluster peers
                    best_score = -1.0
                    best_idx = member_record_indices[0]
                    for candidate_idx in member_record_indices:
                        peer_sims = [sim_matrix[candidate_idx, peer_idx] for peer_idx in member_record_indices]
                        avg_peer_sim = float(np.mean(peer_sims))
                        if avg_peer_sim > best_score:
                            best_score = avg_peer_sim
                            best_idx = candidate_idx
                    exemplar_idx = best_idx

                exemplar_indices.append(exemplar_idx)

                # Assemble full member list including duplicates attached to cluster primaries
                full_member_ids: list[str] = []
                for p_idx in member_record_indices:
                    full_member_ids.append(decoded_records[p_idx]["id"])
                    if p_idx in duplicate_map:
                        for d_idx, _ in duplicate_map[p_idx]:
                            full_member_ids.append(decoded_records[d_idx]["id"])

                clusters_output.append({
                    "cluster_id": cluster_id,
                    "representative_id": decoded_records[exemplar_idx]["id"],
                    "representative_index": exemplar_idx,
                    "member_ids": full_member_ids,
                    "summary": "",
                    "vlm_metadata": {},
                })

        # ----------------------------------------------------------------------
        # Step D: Tier 2 - Adaptive Qwen2.5-VL Pass
        # ----------------------------------------------------------------------
        # Identify documents / receipts via folder metadata or filename keywords
        document_indices: list[int] = []
        doc_keywords = {"receipt", "document", "doc", "invoice", "bill", "scan", "tax", "note", "ticket", "statement"}

        for idx, rec in enumerate(decoded_records):
            folder_str = str(rec["metadata"].get("folder") or rec["metadata"].get("folder_name") or "").lower()
            filename_str = str(rec["metadata"].get("filename") or "").lower()
            is_doc_metadata = any(kw in folder_str or kw in filename_str for kw in doc_keywords)
            if is_doc_metadata and idx not in exemplar_indices:
                document_indices.append(idx)

        # Total selective set for Qwen2.5-VL
        target_vlm_indices = list(dict.fromkeys(exemplar_indices + document_indices))
        vlm_results: dict[int, dict[str, Any]] = {}

        vlm_max_dynamic_pixels = int(config_dict.get("vlm_max_dynamic_pixels", 784 * 28 * 28))
        dense_density = str(config_dict.get("dense_caption_density", "concise"))
        extract_ocr_bboxes = bool(config_dict.get("extract_ocr_bboxes", False))

        for target_idx in target_vlm_indices:
            target_record = decoded_records[target_idx]
            is_doc = (target_idx in document_indices) or extract_ocr_bboxes

            if is_doc:
                # Full dynamic resolution & higher token budget for dense OCR / document layout
                curr_max_pixels = vlm_max_dynamic_pixels
                curr_max_tokens = 2048  # Increased from 768 to prevent JSON cutoff
            else:
                # Non-document scenic clusters: Clamp resolution to 512*512 and cap max_new_tokens to 192
                curr_max_pixels = min(vlm_max_dynamic_pixels, 512 * 512)
                curr_max_tokens = 192

            vlm_metadata = self._run_qwen_vl(
                image=target_record["image"],
                density=dense_density,
                extract_bboxes=extract_ocr_bboxes,
                max_pixels=curr_max_pixels,
                max_new_tokens=curr_max_tokens,
            )
            vlm_results[target_idx] = vlm_metadata

        # Merge VLM metadata into cluster outputs
        for cluster in clusters_output:
            rep_idx = cluster.pop("representative_index", None)
            if rep_idx is not None and rep_idx in vlm_results:
                vlm_meta = vlm_results[rep_idx]
                cluster["summary"] = vlm_meta.get("summary", "")
                cluster["vlm_metadata"] = vlm_meta

        # Assemble document / OCR extractions
        documents_output: list[dict[str, Any]] = []
        for idx in target_vlm_indices:
            meta = vlm_results.get(idx, {})
            is_doc_result = meta.get("is_document", False) or idx in document_indices
            ocr_text = meta.get("ocr_text", "")
            if is_doc_result or (ocr_text and len(ocr_text.strip()) > 5):
                documents_output.append({
                    "image_id": decoded_records[idx]["id"],
                    "ocr_text": ocr_text,
                    "bboxes": meta.get("bboxes", []),
                    "categories": meta.get("categories", ["document"]),
                })

        # ----------------------------------------------------------------------
        # Step E: Response Assembly
        # ----------------------------------------------------------------------
        exec_time = round(time.time() - start_time, 2)
        response = {
            "clusters": clusters_output,
            "duplicates": duplicates_output,
            "documents": documents_output,
            "metrics": {
                "total_ingested": total_ingested,
                "exemplars_processed_vlm": len(exemplar_indices),
                "documents_processed_vlm": len(document_indices),
                "execution_time_sec": exec_time,
            },
        }

        print(
            f"[GalleryVisionEngine] Completed processing {total_ingested} images "
            f"({len(clusters_output)} clusters, {len(duplicates_output)} duplicate sets, "
            f"{len(documents_output)} documents) in {exec_time}s."
        )
        return response

    def _decode_image(self, raw_data: Any) -> Image.Image | None:
        """Decode base64 string, bytearray, or raw bytes into an RGB PIL Image with safe color normalization."""
        if raw_data is None:
            return None
        try:
            if isinstance(raw_data, str):
                # Remove optional data URL header (e.g. data:image/png;base64,...)
                if "," in raw_data and "base64" in raw_data[:30]:
                    raw_data = raw_data.split(",", 1)[1]
                image_bytes = base64.b64decode(raw_data)
            elif isinstance(raw_data, (bytes, bytearray)):
                image_bytes = bytes(raw_data)
            else:
                return None

            img = Image.open(io.BytesIO(image_bytes))

            # Normalize alpha channels or transparency palettes to RGB on a white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba_img = img.convert("RGBA")
                rgb_img = Image.new("RGB", rgba_img.size, (255, 255, 255))
                rgb_img.paste(rgba_img, mask=rgba_img.split()[3])
                return rgb_img
            elif img.mode != "RGB":
                return img.convert("RGB")
            else:
                return img
        except Exception as e:
            print(f"[GalleryVisionEngine] Image decoding error: {e}")
            return None

    def _run_qwen_vl(
        self,
        image: Image.Image,
        density: str,
        extract_bboxes: bool,
        max_pixels: int,
        max_new_tokens: int = 768,
    ) -> dict[str, Any]:
        """Execute Qwen2.5-VL conditional generation for structured vision metadata."""
        import torch

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        caption_guide = {
            "minimal": "Provide a concise 1-sentence caption and 3-5 core visual tags.",
            "concise": "Provide a clear 2-3 sentence descriptive summary, key visual tags, detected categories, and notable objects/people/settings.",
            "exhaustive": "Provide a comprehensive, highly detailed description covering all visual elements, background, atmosphere, subjects, colors, and contextual details.",
        }.get(density, "Provide a clear descriptive summary and tags.")

        ocr_guide = (
            "Extract all readable text with precise bounding boxes in format [ymin, xmin, ymax, xmax] (normalized 0-1000) and line text."
            if extract_bboxes
            else "Extract key readable text (OCR) if present in the image, otherwise leave empty string."
        )

        prompt_text = (
            f"Analyze this image and return ONLY a strict, valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "summary": "Short scene headline",\n'
            f'  "description": "{caption_guide}",\n'
            f'  "tags": ["tag1", "tag2", "tag3"],\n'
            f'  "categories": ["category1", "category2"],\n'
            f'  "is_document": true/false,\n'
            f'  "ocr_text": "{ocr_guide}",\n'
            f'  "bboxes": [{{"label": "text", "bbox_2d": [ymin, xmin, ymax, xmax], "text": "..."}}]\n'
            f"}}\n"
            f"Output strictly valid JSON with no markdown formatting or surrounding commentary."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            if process_vision_info is not None:
                text_prompt = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.qwen_processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                # Direct processor fallback
                text_prompt = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.qwen_processor(
                    text=[text_prompt],
                    images=[image],
                    return_tensors="pt",
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            return self._parse_vlm_json(output_text)
        except Exception as err:
            print(f"[GalleryVisionEngine] Qwen2.5-VL generation error: {err}")
            return {
                "summary": "Visual gallery scene",
                "description": "",
                "tags": [],
                "categories": ["general"],
                "is_document": False,
                "ocr_text": "",
                "bboxes": [],
            }

    @staticmethod
    def _parse_vlm_json(raw_text: str) -> dict[str, Any]:
        """Safely parse JSON response from VLM output with truncation recovery."""
        import json
        import re

        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            start_idx = cleaned.find("{")
            if start_idx != -1:
                cleaned = cleaned[start_idx:]

        # 1. Try standard parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 2. Try closing truncated JSON strings and brackets
        fixed = cleaned
        if fixed.count('"') % 2 != 0:
            fixed += '"'
        fixed = re.sub(r',\s*$', '', fixed)  # remove trailing comma
        
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        if open_brackets > 0:
            fixed += ']' * open_brackets
        if open_braces > 0:
            fixed += '}' * open_braces

        try:
            data = json.loads(fixed)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 3. Regex Fallback for catastrophic truncation
        def _extract_str(key: str, text: str) -> str:
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
            return m.group(1) if m else ""

        def _extract_list(key: str, text: str) -> list[str]:
            m = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            return re.findall(r'"([^"]+)"', m.group(1)) if m else []

        is_doc = '"is_document": true' in cleaned.lower()
        return {
            "summary": _extract_str("summary", cleaned) or "Visual scene",
            "description": _extract_str("description", cleaned) or cleaned[:120].replace("\n", " "),
            "tags": _extract_list("tags", cleaned),
            "categories": _extract_list("categories", cleaned),
            "is_document": is_doc,
            "ocr_text": _extract_str("ocr_text", cleaned),
            "bboxes": [],  # Bboxes are too complex for regex recovery, default to empty
        }


# ============================================================================
# Direct Python Invocation Bridge
# ============================================================================

async def run_gallery_pipeline(
    images: list[dict[str, Any]],
    config: VectorEngineConfig | dict[str, Any],
) -> dict[str, Any]:
    """
    Direct asynchronous invocation bridge from FastAPI or client services
    to the remote Modal Vision Engine.

    Args:
        images: List of image dicts containing payload data.
        config: VectorEngineConfig pydantic model or dict of configuration parameters.

    Returns:
        Structured response dictionary containing clusters, duplicates, documents, and metrics.
    """
    engine = GalleryVisionEngine()
    config_payload = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    result = await engine.process_gallery.remote.aio(images, config_payload)
    return result


# ============================================================================
# Local CLI Remote Execution Test & Local Image Loader
# ============================================================================

def _load_local_test_images(folder_path: str = "app/test-imgs") -> list[dict[str, Any]]:
    """
    Load real local images from folder, downscale large dimensions to <= 1600px,
    and encode as base64 JPEG payload for remote pipeline execution.
    """
    from datetime import datetime
    from pathlib import Path

    dir_path = Path(folder_path)
    if not dir_path.is_absolute() and not dir_path.exists():
        # Check relative to backend directory if called from different subpath
        backend_dir = Path(__file__).resolve().parent.parent.parent
        candidate = backend_dir / folder_path
        if candidate.exists() or (backend_dir / "app").exists():
            dir_path = candidate

    dir_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"[TestRunner] No images found in {folder_path}. Please drop real images into that directory.")
        return []

    test_payload: list[dict[str, Any]] = []
    print(f"[TestRunner] Found {len(image_files)} image(s) in {dir_path}. Loading...")

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

                # Downscale large photos so max dimension <= 1600px
                rgb_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=90)
                base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

                test_payload.append({
                    "id": file_path.stem,
                    "base64": base64_str,
                    "metadata": {
                        "folder": file_path.parent.name,
                        "filename": file_path.name,
                        "timestamp": mod_time,
                    },
                })
        except Exception as e:
            print(f"[TestRunner] Warning: Could not process {file_path.name}: {e}")

    return test_payload


@app.local_entrypoint()
def run_vision_benchmark():
    """Run local entrypoint test against the remote Modal GPU container using local images."""
    print("[Local Entrypoint] Loading local test images from 'app/test-imgs'...")
    test_images = _load_local_test_images("app/test-imgs")
    if not test_images:
        print("[Local Entrypoint] Aborting benchmark: No images found.")
        return

    print(f"[Local Entrypoint] Ingested {len(test_images)} local test images.")

    config = {
        "dedup_cosine_threshold": 0.96,
        "cluster_eps": 0.28,
        "vlm_max_dynamic_pixels": 784 * 28 * 28,
        "dense_caption_density": "concise",
        "extract_ocr_bboxes": True,
    }

    print("[Local Entrypoint] Dispatching remote execution to Modal GalleryVisionEngine...")
    engine = GalleryVisionEngine()
    result = engine.process_gallery.remote(test_images, config)

    print("\n" + "=" * 60)
    print("MODAL REMOTE PIPELINE EXECUTION RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    run_vision_benchmark()
