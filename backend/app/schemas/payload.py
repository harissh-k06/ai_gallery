from typing import Any, Optional
from pydantic import BaseModel, Field


class VectorEngineConfig(BaseModel):
    dedup_cosine_threshold: float = Field(
        default=0.96,
        ge=0.80,
        le=0.99,
        description="Cosine similarity cutoff for near-duplicates",
    )
    cluster_eps: float = Field(
        default=0.28,
        ge=0.05,
        le=0.60,
        description="DBSCAN epsilon for visual clustering",
    )
    temporal_window_hours: int = Field(
        default=6,
        ge=1,
        le=72,
        description="Temporal grouping window for event segmentation",
    )
    hnsw_m: int = Field(
        default=16,
        ge=8,
        le=64,
        description="HNSW graph bi-directional link count",
    )
    hnsw_ef_construction: int = Field(
        default=64,
        ge=16,
        le=256,
        description="HNSW construction search depth",
    )
    vlm_max_dynamic_pixels: int = Field(
        default=784 * 28 * 28,
        description="Qwen2.5-VL dynamic resolution ceiling",
    )
    dense_caption_density: str = Field(
        default="concise",
        description="'minimal' | 'concise' | 'exhaustive'",
    )
    extract_ocr_bboxes: bool = Field(
        default=False,
        description="Extract structured document bounding boxes",
    )
    hybrid_search_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Dense vs sparse hybrid search balance",
    )


class GalleryFingerprint(BaseModel):
    total_photos: int = Field(
        ...,
        ge=0,
        description="Total number of photos in gallery",
    )
    total_videos: int = Field(
        default=0,
        ge=0,
        description="Total number of videos in gallery",
    )
    device_os: str = Field(
        default="android",
        description="Operating system of the client device (e.g. android, ios)",
    )
    folder_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of folder/album names to media item counts",
    )
    media_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of file formats (e.g. jpg, png, heic, mp4)",
    )
    geotagged_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of photos containing GPS coordinates",
    )
    burst_detected_count: int = Field(
        default=0,
        ge=0,
        description="Count of photos taken within rapid intervals (<2s)",
    )
    date_span_days: int = Field(
        default=1,
        ge=0,
        description="Time span from earliest to latest photo in days",
    )
    estimated_storage_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated media storage consumption in megabytes",
    )
    user_mode_override: Optional[str] = Field(
        default=None,
        description="Optional manual override mode ID (e.g. memories_focus, productivity_focus)",
    )
    user_intent_hint: Optional[str] = Field(
        default=None,
        description="Optional user prompt or natural language hint",
    )
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary telemetry metadata",
    )


class ModeOption(BaseModel):
    mode_id: str
    title: str
    description: str
    config: VectorEngineConfig


class StrategyRecommendationResponse(BaseModel):
    recommended_mode: str
    recommendation_reason: str
    active_config: VectorEngineConfig
    status: str = "success"
    available_modes: list[ModeOption] = Field(default_factory=list)


# Alias for backward compatibility
IngestionResponse = StrategyRecommendationResponse


class ImagePayloadItem(BaseModel):
    id: str = Field(..., description="Unique client identifier or key for the media item")
    base64: str = Field(..., description="Base64-encoded image data string")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-side media metadata (e.g. filename, timestamp, folder, location)",
    )


class ProcessGalleryRequest(BaseModel):
    images: list[ImagePayloadItem] = Field(
        ...,
        description="List of images with base64 data to process in the GPU pipeline",
    )
    config: Optional[VectorEngineConfig] = Field(
        default=None,
        description="Optional engine configuration tuning parameters",
    )
    mode_override: Optional[str] = Field(
        default=None,
        description="Optional preset mode override ID",
    )


class ClusterItem(BaseModel):
    cluster_id: int
    representative_id: str
    member_ids: list[str]
    summary: str = ""
    vlm_metadata: dict[str, Any] = Field(default_factory=dict)


class DuplicateItem(BaseModel):
    primary_id: str
    duplicate_ids: list[str]
    similarity_score: float


class DocumentItem(BaseModel):
    image_id: str
    ocr_text: str = ""
    bboxes: list[dict[str, Any]] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class ProcessingMetrics(BaseModel):
    total_ingested: int
    exemplars_processed_vlm: int = 0
    documents_processed_vlm: int = 0
    execution_time_sec: float = 0.0


class ProcessGalleryResponse(BaseModel):
    clusters: list[ClusterItem] = Field(default_factory=list)
    duplicates: list[DuplicateItem] = Field(default_factory=list)
    documents: list[DocumentItem] = Field(default_factory=list)
    metrics: ProcessingMetrics = Field(
        default_factory=lambda: ProcessingMetrics(total_ingested=0)
    )
