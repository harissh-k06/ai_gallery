from typing import Any
from pydantic import BaseModel, Field


class VectorEngineConfig(BaseModel):
    dedup_cosine_threshold: float = Field(default=0.96, ge=0.80, le=0.99, description="Cosine similarity cutoff for near-duplicates")
    cluster_eps: float = Field(default=0.28, ge=0.05, le=0.60, description="DBSCAN epsilon for visual clustering")
    temporal_window_hours: int = Field(default=6, ge=1, le=72, description="Temporal grouping window for event segmentation")
    hnsw_m: int = Field(default=16, ge=8, le=64, description="HNSW graph bi-directional link count")
    hnsw_ef_construction: int = Field(default=64, ge=16, le=256, description="HNSW construction search depth")
    vlm_max_dynamic_pixels: int = Field(default=784 * 28 * 28, description="Qwen2.5-VL dynamic resolution ceiling")
    dense_caption_density: str = Field(default="concise", description="'minimal' | 'concise' | 'exhaustive'")
    extract_ocr_bboxes: bool = Field(default=False, description="Extract structured document bounding boxes")
    hybrid_search_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense vs sparse hybrid search balance")


class GalleryFingerprint(BaseModel):
    total_photos: int
    total_videos: int = 0
    device_os: str
    folder_breakdown: dict[str, int]
    media_type_counts: dict[str, int] = Field(default_factory=dict, description="Counts of jpg, png, heic, mp4, etc.")
    geotagged_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of photos containing GPS data")
    burst_detected_count: int = Field(default=0, description="Photos taken within <2s intervals")
    date_span_days: int = Field(default=0, description="Time span from earliest to latest photo")
    estimated_storage_mb: float = 0.0
    user_mode_override: str | None = None
    user_intent_hint: str | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class ModeOption(BaseModel):
    mode_id: str
    title: str
    description: str
    config: VectorEngineConfig


class IngestionResponse(BaseModel):
    status: str
    recommended_mode: str
    recommendation_reason: str
    active_config: VectorEngineConfig
    available_modes: list[ModeOption]
