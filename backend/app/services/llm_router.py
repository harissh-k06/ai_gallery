import json
# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI
from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.schemas.payload import GalleryFingerprint, IngestionResponse, ModeOption, VectorEngineConfig

PRESET_CATALOG: list[ModeOption] = [
    ModeOption(
        mode_id="smart_all_in_one",
        title="Smart All-in-One",
        description="Full holistic indexing: balanced clustering, standard OCR extraction, and moderate deduplication.",
        config=VectorEngineConfig(
            dedup_cosine_threshold=0.96,
            cluster_eps=0.28,
            temporal_window_hours=6,
            hnsw_m=16,
            hnsw_ef_construction=64,
            vlm_max_dynamic_pixels=784 * 28 * 28,
            dense_caption_density="concise",
            extract_ocr_bboxes=False,
            hybrid_search_alpha=0.7,
        ),
    ),
    ModeOption(
        mode_id="memories_focus",
        title="Memories & Highlights",
        description="Prioritizes event narrative, fine-grained visual clustering, and rich dense captions for highlight reels.",
        config=VectorEngineConfig(
            dedup_cosine_threshold=0.98,
            cluster_eps=0.22,
            temporal_window_hours=12,
            hnsw_m=32,
            hnsw_ef_construction=128,
            vlm_max_dynamic_pixels=1024 * 28 * 28,
            dense_caption_density="exhaustive",
            extract_ocr_bboxes=False,
            hybrid_search_alpha=0.85,
        ),
    ),
    ModeOption(
        mode_id="productivity_focus",
        title="Productivity & Documents",
        description="Prioritizes high-density OCR extraction, bounding box detection, and lexical hybrid search across receipts and notes.",
        config=VectorEngineConfig(
            dedup_cosine_threshold=0.95,
            cluster_eps=0.35,
            temporal_window_hours=2,
            hnsw_m=16,
            hnsw_ef_construction=64,
            vlm_max_dynamic_pixels=1280 * 28 * 28,
            dense_caption_density="concise",
            extract_ocr_bboxes=True,
            hybrid_search_alpha=0.4,
        ),
    ),
    ModeOption(
        mode_id="storage_saver",
        title="Storage Declutter",
        description="Aggressive near-duplicate pruning with lower cosine threshold and fast coarse clustering.",
        config=VectorEngineConfig(
            dedup_cosine_threshold=0.90,
            cluster_eps=0.40,
            temporal_window_hours=1,
            hnsw_m=12,
            hnsw_ef_construction=32,
            vlm_max_dynamic_pixels=512 * 28 * 28,
            dense_caption_density="minimal",
            extract_ocr_bboxes=False,
            hybrid_search_alpha=0.6,
        ),
    ),
]

PRESET_MAP: dict[str, ModeOption] = {preset.mode_id: preset for preset in PRESET_CATALOG}


def _deterministic_fallback(fingerprint: GalleryFingerprint) -> tuple[str, str]:
    if fingerprint.total_photos <= 0:
        return "smart_all_in_one", "Default holistic vector indexing strategy selected for an empty gallery."

    total = fingerprint.total_photos
    burst_ratio = fingerprint.burst_detected_count / total if total > 0 else 0.0

    breakdown_lower = {k.lower(): v for k, v in fingerprint.folder_breakdown.items()}
    doc_count = sum(v for k, v in breakdown_lower.items() if any(w in k for w in ["screenshot", "download", "document", "receipt", "invoice"]))
    camera_count = sum(v for k, v in breakdown_lower.items() if any(w in k for w in ["dcim", "camera"]))

    if burst_ratio > 0.25:
        return (
            "storage_saver",
            f"Detected {fingerprint.burst_detected_count} rapid burst shots ({burst_ratio:.0%} of library), activating aggressive deduplication and coarse clustering.",
        )

    if (doc_count / total) > 0.35:
        return (
            "productivity_focus",
            f"Documents and receipts represent {doc_count / total:.0%} of your library, activating high-resolution OCR bounding box extraction and hybrid search.",
        )

    if (camera_count / total) > 0.55 or fingerprint.geotagged_ratio > 0.40:
        return (
            "memories_focus",
            f"Camera photography dominates your library with {fingerprint.geotagged_ratio:.0%} geotagged media, activating fine-grained visual clustering and exhaustive dense captions.",
        )

    return (
        "smart_all_in_one",
        "Balanced distribution across photos, documents, and albums selected for holistic multi-priority vector indexing.",
    )


async def _analyze_with_llm(fingerprint: GalleryFingerprint) -> tuple[str, str] | None:
    if not LLM_API_KEY:
        return None

    catalog_descriptions = "\n".join(
        f"- {p.mode_id}: {p.title} - {p.description} (VLM pixels={p.config.vlm_max_dynamic_pixels}, OCR bboxes={p.config.extract_ocr_bboxes}, dedup cosine={p.config.dedup_cosine_threshold}, cluster eps={p.config.cluster_eps})"
        for p in PRESET_CATALOG
    )
    system_prompt = (
        "You are an AI gallery strategy router. Analyze the device's rich metadata telemetry "
        "(folder distributions, geotag ratios, burst counts, date spans, and storage footprint) "
        "to recommend the optimal vector database indexing and VLM processing mode.\n\n"
        f"Available Modes:\n{catalog_descriptions}\n\n"
        "Provide an insightful, consumer-friendly 1-2 sentence rationale explaining how the selected mode "
        "balances processing across memories, documents, and decluttering."
    )
    user_payload = {
        "total_photos": fingerprint.total_photos,
        "total_videos": fingerprint.total_videos,
        "device_os": fingerprint.device_os,
        "folder_breakdown": fingerprint.folder_breakdown,
        "media_type_counts": fingerprint.media_type_counts,
        "geotagged_ratio": fingerprint.geotagged_ratio,
        "burst_detected_count": fingerprint.burst_detected_count,
        "date_span_days": fingerprint.date_span_days,
        "estimated_storage_mb": fingerprint.estimated_storage_mb,
        "user_intent_hint": fingerprint.user_intent_hint,
        "extra_metadata": fingerprint.extra_metadata,
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_gallery_strategy",
                "description": "Select the optimal gallery vector engine and VLM configuration based on rich metadata telemetry.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_mode": {
                            "type": "string",
                            "enum": [
                                "smart_all_in_one",
                                "memories_focus",
                                "productivity_focus",
                                "storage_saver",
                            ],
                            "description": "The selected mode ID from the catalog.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Consumer-friendly 1-2 sentence rationale detailing how processing is balanced across memories, documents, and decluttering.",
                        },
                    },
                    "required": ["selected_mode", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    async with AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) as client:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "set_gallery_strategy"}},
            temperature=0.1,
        )

    message = response.choices[0].message
    if not message.tool_calls:
        return None

    try:
        call = message.tool_calls[0]
        arguments = json.loads(call.function.arguments)
        mode_id = arguments.get("selected_mode")
        reason = arguments.get("reason", "")
        if mode_id in PRESET_MAP and reason:
            return str(mode_id), str(reason)
    except Exception:
        return None

    return None


async def route_gallery_strategy(fingerprint: GalleryFingerprint) -> IngestionResponse:
    if fingerprint.user_mode_override and fingerprint.user_mode_override in PRESET_MAP:
        selected = PRESET_MAP[fingerprint.user_mode_override]
        return IngestionResponse(
            status="success",
            recommended_mode=selected.mode_id,
            recommendation_reason="User manual override applied.",
            active_config=selected.config,
            available_modes=PRESET_CATALOG,
        )

    decision = None
    try:
        decision = await _analyze_with_llm(fingerprint)
    except Exception:
        decision = None

    if decision is None:
        mode_id, reason = _deterministic_fallback(fingerprint)
    else:
        mode_id, reason = decision

    selected = PRESET_MAP[mode_id]
    return IngestionResponse(
        status="success",
        recommended_mode=selected.mode_id,
        recommendation_reason=reason,
        active_config=selected.config,
        available_modes=PRESET_CATALOG,
    )


if __name__ == "__main__":
    import asyncio

    async def test():
        sample_fingerprint = GalleryFingerprint(
            total_photos=3420,
            total_videos=145,
            device_os="android",
            folder_breakdown={
                "DCIM/Camera": 1950,
                "Screenshots": 820,
                "WhatsApp Images": 450,
                "Download": 200,
            },
            media_type_counts={"jpg": 2600, "png": 820, "mp4": 145},
            geotagged_ratio=0.62,
            burst_detected_count=380,
            date_span_days=730,
            estimated_storage_mb=14200.5,
            user_mode_override=None,
            user_intent_hint="Organize my recent vacation and clean up old screenshots",
        )

        print("Testing LLM Router with rich metadata telemetry...")
        response = await route_gallery_strategy(sample_fingerprint)
        print("\n--- Ingestion Response Result ---")
        print(response.model_dump_json(indent=2))

    asyncio.run(test())
