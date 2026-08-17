import logging
from typing import Any
from fastapi import APIRouter, HTTPException, status

from app.schemas.payload import (
    GalleryFingerprint,
    ProcessGalleryRequest,
    ProcessGalleryResponse,
    StrategyRecommendationResponse,
    VectorEngineConfig,
)
from app.services.llm_router import PRESET_MAP, get_llm_recommendation
from app.services.modal_runner import run_gallery_pipeline

logger = logging.getLogger("ai_gallery.api.ingest")

router = APIRouter(prefix="/gallery", tags=["Gallery Processing"])


@router.get(
    "/health",
    summary="Service Health Check",
    description="Check the operational status of the AI Gallery backend ingestion service.",
)
async def health_check() -> dict[str, str]:
    """Return basic health status for service liveness checks."""
    return {"status": "ok", "service": "ai-gallery-backend"}


@router.post(
    "/recommend-strategy",
    response_model=StrategyRecommendationResponse,
    summary="Recommend Vector Indexing & VLM Strategy",
    description=(
        "Analyzes device gallery telemetry (folder breakdown, media distribution, "
        "geotagged ratios, burst photo counts, and date spans) using an LLM router "
        "to recommend optimal Vector Engine & VLM hyperparameters."
    ),
)
async def recommend_strategy(fingerprint: GalleryFingerprint) -> StrategyRecommendationResponse:
    """Analyze client gallery fingerprint and return recommended configuration preset."""
    logger.info(
        f"Received strategy recommendation request: {fingerprint.total_photos} photos, "
        f"{fingerprint.total_videos} videos, OS: {fingerprint.device_os}"
    )
    try:
        recommendation = await get_llm_recommendation(fingerprint)
        logger.info(
            f"Successfully resolved recommendation: mode='{recommendation.recommended_mode}' "
            f"reason='{recommendation.recommendation_reason}'"
        )
        return recommendation
    except Exception as exc:
        logger.error(f"Failed to generate strategy recommendation: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy recommendation failed: {str(exc)}",
        ) from exc


@router.post(
    "/process",
    response_model=ProcessGalleryResponse,
    summary="Process Gallery Images via Modal GPU Vision Engine",
    description=(
        "Ingests a batch of base64-encoded images and applies a two-tier GPU vision pipeline: "
        "Tier 1 high-throughput SigLIP vector embedding for clustering & near-duplicate elimination, "
        "and Tier 2 selective Qwen2.5-VL inference for scene summarization and OCR bounding boxes."
    ),
)
async def process_gallery(request: ProcessGalleryRequest) -> ProcessGalleryResponse:
    """Execute two-tier GPU vision pipeline on the provided batch of gallery images."""
    if not request.images:
        logger.warning("Rejected /process request: empty image list provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'images' list cannot be empty. Please supply at least one image to process.",
        )

    # Determine active VectorEngineConfig
    if request.config is not None:
        active_config = request.config
    elif request.mode_override and request.mode_override in PRESET_MAP:
        active_config = PRESET_MAP[request.mode_override].config
    else:
        active_config = VectorEngineConfig()

    logger.info(
        f"Starting gallery processing for {len(request.images)} image(s). "
        f"Config: dedup_thresh={active_config.dedup_cosine_threshold}, "
        f"cluster_eps={active_config.cluster_eps}, OCR={active_config.extract_ocr_bboxes}"
    )

    # Prepare payload for Modal pipeline
    images_payload = [item.model_dump() for item in request.images]

    try:
        pipeline_result = await run_gallery_pipeline(images_payload, active_config)
        logger.info(
            f"Successfully processed {len(request.images)} image(s) via Modal GPU: "
            f"{len(pipeline_result.get('clusters', []))} clusters, "
            f"{len(pipeline_result.get('duplicates', []))} duplicate sets, "
            f"{len(pipeline_result.get('documents', []))} documents."
        )
        return ProcessGalleryResponse(**pipeline_result)
    except Exception as exc:
        logger.error(f"Modal GPU remote execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Remote GPU vision pipeline execution failed: {str(exc)}",
        ) from exc
