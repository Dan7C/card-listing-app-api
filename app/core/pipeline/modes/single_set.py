# Single-Set mode orchestrator.
# All images in the directory are assumed to belong to the same set.
#
# Flow:
#   1. Resolve set context from user input, directory name, filename,
#      or discovery mode fallback
#   2. Collect images
#   3. Build hash registry and detect duplicates
#   4. Classify every image (detects full set images, face, subject)
#      checking rate limits between calls
#   5. Resolve image mode (front-only vs front-back)
#   6. Pair images using classification results
#   7. Extract metadata for each pair, checking rate limits between calls
#   8. Write candidates JSON incrementally for resume capability
#   9. Write deferred JSON for any jobs requiring reprocessing
#
# Supported vs Discovery:
#   Determined once at step 1 for the entire run.
#   SupportedResult → extract_supported (constrained prompting)
#   DiscoveryResult → extract_discovery (unconstrained prompting)

import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from app.core.llm.client import LLMClient, LLMClientError
from app.core.llm.classification import classify_image, ClassificationError
from app.core.llm.extraction import extract_supported, extract_discovery
from app.core.llm.rate_limit import check_batch_feasibility, RateLimitStatus
from app.core.pipeline.pairing import (
    pair_images,
    infer_image_mode,
    ImageMode,
    PairingResult
)
from app.core.pipeline.results import (
    SupportedResult,
    DiscoveryResult,
    ClassificationResult,
    CardPair
)
from app.utils.file_walker import get_image_files, ImageSortOrder
from app.utils.image import hash_image
from app.utils.fuzzy_match import (
    resolve_manufacturer_and_set,
    match_set_key,
    get_set_keys
)

logger = logging.getLogger(__name__)

# Review tier constants - higher tier = higher confidence
TIER_STANDARD = 1         # supported + high confidence + complete
TIER_LOW_CONFIDENCE = 2   # discovery mode or unverified confidence
TIER_INCOMPLETE = 3       # required fields missing
TIER_NO_INFORMATION = 4   # extraction failed entirely


class SingleSetConfig:
    """
    Configuration for a Single-Set mode processing run.

    Args:
        source_path:      Path to directory or single image file
        sets_config:      The full sets_config dictionary
        manufacturer_key: Optional manufacturer config key declared by user
        set_key:          Optional set config key declared by user
        image_mode:       Declared image mode, or UNKNOWN to infer
        max_depth:        Directory scan depth (0=flat, -1=unlimited)
        sort_order:       Image ordering for pairing correctness
        output_dir:       Directory to write JSON output to
        batch_id:         Unique identifier for this run
        resume:           Whether to resume a previous interrupted run
    """

    def __init__(
        self,
        source_path: Path,
        sets_config: dict,
        manufacturer_key: str | None = None,
        set_key: str | None = None,
        image_mode: ImageMode = ImageMode.UNKNOWN,
        max_depth: int = 0,
        sort_order: ImageSortOrder = ImageSortOrder.FILESYSTEM,
        output_dir: Path = Path("outputs"),
        batch_id: str | None = None,
        resume: bool = False
    ):
        self.source_path = source_path
        self.sets_config = sets_config
        self.manufacturer_key = manufacturer_key
        self.set_key = set_key
        self.image_mode = image_mode
        self.max_depth = max_depth
        self.sort_order = sort_order
        self.output_dir = output_dir
        self.batch_id = batch_id or _generate_batch_id()
        self.resume = resume


class SingleSetError(Exception):
    """
    Raised when Single-Set mode cannot be initialised or run.
    Distinct from per-image errors which are captured in results.
    """
    pass


class RateLimitExhaustedError(Exception):
    """
    Raised when the API rate limit is exhausted mid-batch.
    Caught by run_single_set to trigger deferred queue write
    and graceful shutdown.
    """
    pass


async def run_single_set(
    config: SingleSetConfig,
    client: LLMClient
) -> tuple[Path, Path]:
    """
    Runs the Single-Set mode pipeline.

    Returns paths to both output files:
        candidates JSON - all processed results pending review
        deferred JSON   - jobs requiring reprocessing when API available

    Args:
        config: SingleSetConfig instance describing the run
        client: Configured LLMClient instance

    Returns:
        Tuple of (candidates_path, deferred_path)

    Raises:
        SingleSetError if the run cannot be initialised
    """
    logger.info(
        f"Starting Single-Set run {config.batch_id} "
        f"on {config.source_path}"
    )

    # load existing results if resuming
    existing_candidates = {}
    existing_deferred = []
    if config.resume:
        existing_candidates = _load_existing_candidates(
            config.output_dir, config.batch_id
        )
        existing_deferred = _load_existing_deferred(
            config.output_dir, config.batch_id
        )
        logger.info(
            f"Resuming run {config.batch_id} - "
            f"{len(existing_candidates)} images already processed, "
            f"{len(existing_deferred)} deferred jobs"
        )

    # collect images
    images = _collect_images(config)
    if not images:
        raise SingleSetError(
            f"No supported images found at {config.source_path}"
        )
    logger.info(f"Found {len(images)} images")

    # resolve set context
    set_context = _resolve_set_context(config, images)
    logger.info(f"Set context resolved: {set_context}")

    # get current rate limit status from provider
    rate_limit_status = _get_rate_limit_status(client)

    # pre-batch feasibility check
    unprocessed_count = sum(
        1 for img in images
        if str(img) not in existing_candidates
    )
    can_complete, feasibility_message = check_batch_feasibility(
        unprocessed_count, rate_limit_status
    )
    logger.info(f"Batch feasibility: {feasibility_message}")
    if not can_complete:
        logger.warning(feasibility_message)

    # build hash registry and identify duplicates
    hash_registry = {}
    duplicate_map = {}
    for image_path in images:
        image_hash = hash_image(image_path)
        if image_hash in hash_registry:
            duplicate_map[str(image_path)] = str(hash_registry[image_hash])
            logger.info(
                f"Duplicate detected: {image_path.name} is identical "
                f"to {hash_registry[image_hash].name}"
            )
        else:
            hash_registry[image_hash] = image_path

    # classify every image
    classifications: list[ClassificationResult] = []
    deferred_jobs = list(existing_deferred)
    candidates = list(existing_candidates.values())
    processed_paths = set(existing_candidates.keys())

    try:
        for image in images:
            if str(image) in processed_paths:
                logger.debug(
                    f"Skipping already classified: {image.name}"
                )
                # use existing classification placeholder
                classifications.append(
                    _placeholder_classification()
                )
                continue

            if str(image) in duplicate_map:
                logger.debug(
                    f"Skipping duplicate for classification: {image.name}"
                )
                classifications.append(
                    _placeholder_classification()
                )
                continue

            _check_rate_limit(client)

            try:
                classification = await classify_image(image, client)
                classifications.append(classification)
            except ClassificationError as e:
                logger.error(
                    f"Classification failed for {image.name}: {e}"
                )
                classifications.append(_placeholder_classification())

    except RateLimitExhaustedError:
        logger.error(
            "Rate limit exhausted during classification. "
            "Saving progress and deferring remaining images."
        )
        # defer unclassified images
        classified_count = len(classifications)
        for image in images[classified_count:]:
            if str(image) not in duplicate_map:
                deferred_jobs.append(
                    _build_deferred_job(image, None, set_context, config)
                )
        _write_deferred(deferred_jobs, config)
        return _write_candidates(candidates, config), \
               _write_deferred(deferred_jobs, config)

    # resolve image mode
    image_mode = config.image_mode
    if image_mode == ImageMode.UNKNOWN:
        image_mode = infer_image_mode(classifications)
        logger.info(f"Image mode inferred: {image_mode.value}")

    # pair images
    pairing_result = pair_images(
        images=images,
        classifications=classifications,
        image_mode=image_mode,
        duplicate_map=duplicate_map
    )

    # process pairs
    try:
        for pair in pairing_result.pairs:
            pair_key = str(pair.front_path)

            if pair_key in processed_paths:
                logger.debug(
                    f"Skipping already processed: {pair.front_path.name}"
                )
                continue

            _check_rate_limit(client)

            candidate = await _process_pair(
                pair=pair,
                set_context=set_context,
                config=config,
                client=client
            )
            candidates.append(candidate)

            # write incrementally for resume capability
            _write_candidates(candidates, config)

    except RateLimitExhaustedError:
        logger.error(
            "Rate limit exhausted during extraction. "
            "Saving progress and deferring remaining pairs."
        )
        processed_in_this_run = {c["front_image_path"] for c in candidates}
        for pair in pairing_result.pairs:
            if str(pair.front_path) not in processed_in_this_run:
                deferred_jobs.append(
                    _build_deferred_job(
                        pair.front_path,
                        pair.back_path,
                        set_context,
                        config
                    )
                )

    # route orphaned backs to deferred queue
    for orphaned_back in pairing_result.orphaned_backs:
        deferred_jobs.append(
            _build_orphaned_back_job(orphaned_back, set_context, config)
        )

    # process duplicates - clone result from original
    for duplicate_path, original_path in duplicate_map.items():
        original = next(
            (c for c in candidates
             if c["front_image_path"] == original_path),
            None
        )
        if original:
            candidates.append(
                _clone_as_duplicate(original, duplicate_path)
            )

    # final write
    candidates_path = _write_candidates(candidates, config)
    deferred_path = _write_deferred(deferred_jobs, config)

    logger.info(
        f"Single-Set run {config.batch_id} complete. "
        f"{len(candidates)} candidates, "
        f"{len(deferred_jobs)} deferred jobs."
    )

    return candidates_path, deferred_path


async def _process_pair(
    pair: CardPair,
    set_context: SupportedResult | DiscoveryResult,
    config: SingleSetConfig,
    client: LLMClient
) -> dict:
    """
    Processes a single CardPair through the appropriate extraction path.
    Returns a candidate dictionary ready for JSON serialisation.
    Captures errors as TIER_NO_INFORMATION candidates rather than
    crashing the pipeline.
    """
    logger.info(f"Processing: {pair.front_path.name}")

    candidate_id = str(uuid.uuid4())

    try:
        if isinstance(set_context, SupportedResult):
            manufacturer_config = config.sets_config["manufacturers"][
                set_context.manufacturer
            ]
            set_cfg = manufacturer_config["sets"][set_context.set_key]
            extraction = await extract_supported(
                pair=pair,
                manufacturer_config=manufacturer_config,
                set_config=set_cfg,
                client=client
            )
        else:
            extraction = await extract_discovery(
                pair=pair,
                client=client
            )

        review_tier = _assign_review_tier(extraction, set_context)

        return {
            "candidate_id": candidate_id,
            "batch_id": config.batch_id,
            "front_image_path": str(pair.front_path),
            "back_image_path": str(pair.back_path) if pair.has_back else None,
            "manufacturer": set_context.manufacturer
                if isinstance(set_context, SupportedResult)
                else set_context.known_manufacturer,
            "set_key": set_context.set_key
                if isinstance(set_context, SupportedResult) else None,
            "player_name": extraction.player_name,
            "team_name": extraction.team_name,
            "card_number": extraction.card_number,
            "variant": extraction.variant,
            "condition": extraction.condition,
            "condition_notes": extraction.condition_notes,
            "processing_mode": extraction.processing_mode,
            "confidence": extraction.confidence,
            "review_tier": review_tier,
            "review_status": "pending",
            "is_duplicate": False,
            "duplicate_of": None,
            "is_pairing_disruption": pair.is_pairing_disruption,
            "raw_classification": pair.classification.raw_response
                if pair.classification else None,
            "raw_extraction": extraction.raw_response,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

    except LLMClientError as e:
        logger.error(f"LLM call failed for {pair.front_path.name}: {e}")
        return _error_candidate(candidate_id, pair, config, str(e))


def _resolve_set_context(
    config: SingleSetConfig,
    images: list[Path]
) -> SupportedResult | DiscoveryResult:
    """
    Resolves manufacturer and set context using a confidence hierarchy:

        Level 1: User declared both manufacturer and set
                 → SupportedResult directly, no matching needed

        Level 2: User declared manufacturer only
                 → attempt fuzzy match of directory/filename against
                   that manufacturer's sets specifically
                 → SupportedResult if matched, else DiscoveryResult
                   with known_manufacturer

        Level 3: No user declaration
                 → attempt fuzzy match of directory name against
                   full config
                 → attempt fuzzy match of filename against full config
                 → DiscoveryResult if no match found
    """
    # level 1 - user declared both
    if config.manufacturer_key and config.set_key:
        logger.info(
            f"Set context from user declaration: "
            f"{config.manufacturer_key} / {config.set_key}"
        )
        return SupportedResult(
            manufacturer=config.manufacturer_key,
            set_key=config.set_key
        )

    # level 2 - user declared manufacturer only
    if config.manufacturer_key:
        logger.info(
            f"Manufacturer declared: {config.manufacturer_key}. "
            f"Attempting set match."
        )
        set_keys = get_set_keys(config.sets_config, config.manufacturer_key)
        query = _get_name_hint(config, images)

        if query:
            set_result = match_set_key(query, set_keys)
            if not set_result.failed:
                from app.utils.fuzzy_match import _resolve_to_config_key
                set_key = _resolve_to_config_key(
                    set_result.matched_key,
                    config.sets_config["manufacturers"][
                        config.manufacturer_key
                    ]["sets"]
                )
                logger.info(
                    f"Set matched from '{query}': {set_key}"
                )
                return SupportedResult(
                    manufacturer=config.manufacturer_key,
                    set_key=set_key
                )

        logger.info(
            f"No set match found. Routing to discovery with "
            f"known manufacturer: {config.manufacturer_key}"
        )
        return DiscoveryResult(
            reason="unknown_set",
            known_manufacturer=config.manufacturer_key
        )

    # level 3 - no user declaration, try directory then filename
    query = _get_name_hint(config, images)
    if query:
        result = resolve_manufacturer_and_set(
            manufacturer_query=query,
            set_query=query,
            config=config.sets_config
        )
        if isinstance(result, SupportedResult):
            logger.info(
                f"Set context from name hint '{query}': "
                f"{result.manufacturer} / {result.set_key}"
            )
            return result
        logger.info(
            f"Name hint '{query}' did not match any known set"
        )

    logger.info("No set context found - routing to discovery mode")
    return DiscoveryResult(reason="no_context")


def _get_name_hint(
    config: SingleSetConfig,
    images: list[Path]
) -> str | None:
    """
    Returns the best available name hint for fuzzy matching.
    Prefers directory name for directory sources, falls back to
    first image filename stem.
    """
    if config.source_path.is_dir():
        return config.source_path.name

    if images:
        return images[0].stem

    return None


def _collect_images(config: SingleSetConfig) -> list[Path]:
    """
    Collects images from source_path.
    Handles both single file and directory sources.
    """
    if config.source_path.is_file():
        return [config.source_path]

    return get_image_files(
        directory=config.source_path,
        max_depth=config.max_depth,
        sort_order=config.sort_order
    )


def _assign_review_tier(
    extraction,
    set_context: SupportedResult | DiscoveryResult
) -> int:
    """
    Assigns a review tier based on extraction result and set context.

        TIER_1 (standard):        supported + high confidence + complete
        TIER_2 (low confidence):  discovery mode or unverified confidence
        TIER_3 (incomplete):      required fields missing but some data
        TIER_4 (no information):  all fields None, extraction failed
    """
    all_fields = [
        extraction.player_name,
        extraction.team_name,
        extraction.card_number,
        extraction.variant,
        extraction.condition
    ]

    if all(field is None for field in all_fields):
        return TIER_NO_INFORMATION

    required_fields = [
        extraction.player_name,
        extraction.team_name,
        extraction.condition
    ]
    if any(field is None for field in required_fields):
        return TIER_INCOMPLETE

    if (
        isinstance(set_context, DiscoveryResult)
        or extraction.confidence == "unverified"
    ):
        return TIER_LOW_CONFIDENCE

    return TIER_STANDARD


def _check_rate_limit(client: LLMClient) -> None:
    """
    Checks the current rate limit status before making an API call.
    Raises RateLimitExhaustedError if no requests remain.
    """
    status = _get_rate_limit_status(client)
    if status is not None and status.is_exhausted:
        raise RateLimitExhaustedError(
            f"Rate limit exhausted. Resets at {status.reset_at}."
        )


def _get_rate_limit_status(
    client: LLMClient
) -> RateLimitStatus | None:
    """
    Retrieves rate limit status from the primary provider if available.
    Returns None if the provider doesn't support rate limit tracking
    or no calls have been made yet this session.
    """
    provider = client.provider
    return getattr(provider, "rate_limit_status", None)


def _placeholder_classification() -> ClassificationResult:
    """
    Returns an unusable ClassificationResult for images that were
    skipped (duplicates, already processed). Used to maintain
    index alignment between images and classifications lists.
    """
    from app.core.pipeline.results import ClassificationResult
    return ClassificationResult(
        manufacturer=None,
        set_name=None,
        face=None,
        subject=None,
        contains_multiple_cards=False,
        raw_response=""
    )


def _clone_as_duplicate(
    original: dict,
    duplicate_path: str
) -> dict:
    """
    Clones an existing candidate result for a duplicate image.
    Sets is_duplicate=True, links to original, assigns new candidate_id.
    """
    cloned = original.copy()
    cloned["candidate_id"] = str(uuid.uuid4())
    cloned["front_image_path"] = duplicate_path
    cloned["back_image_path"] = None
    cloned["is_duplicate"] = True
    cloned["duplicate_of"] = original["front_image_path"]
    cloned["review_status"] = "pending"
    cloned["processed_at"] = datetime.now(timezone.utc).isoformat()
    return cloned


def _build_deferred_job(
    front_path: Path,
    back_path: Path | None,
    set_context: SupportedResult | DiscoveryResult,
    config: SingleSetConfig
) -> dict:
    """
    Builds a deferred job record for an image that could not be
    processed due to rate limit exhaustion.
    """
    return {
        "job_id": str(uuid.uuid4()),
        "batch_id": config.batch_id,
        "job_type": "extraction",
        "front_image_path": str(front_path),
        "back_image_path": str(back_path) if back_path else None,
        "manufacturer_key": set_context.manufacturer
            if isinstance(set_context, SupportedResult)
            else set_context.known_manufacturer,
        "set_key": set_context.set_key
            if isinstance(set_context, SupportedResult) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending"
    }


def _build_orphaned_back_job(
    back_path: Path,
    set_context: SupportedResult | DiscoveryResult,
    config: SingleSetConfig
) -> dict:
    """
    Builds a deferred job record for an orphaned back image that
    requires manual front pairing during user review.
    """
    return {
        "job_id": str(uuid.uuid4()),
        "batch_id": config.batch_id,
        "job_type": "orphaned_back",
        "front_image_path": None,
        "back_image_path": str(back_path),
        "manufacturer_key": set_context.manufacturer
            if isinstance(set_context, SupportedResult)
            else set_context.known_manufacturer,
        "set_key": set_context.set_key
            if isinstance(set_context, SupportedResult) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_manual_pairing"
    }


def _error_candidate(
    candidate_id: str,
    pair: CardPair,
    config: SingleSetConfig,
    error: str
) -> dict:
    """
    Builds a TIER_NO_INFORMATION candidate for a pair that failed
    to process. Captured in the review queue rather than crashing
    the pipeline.
    """
    return {
        "candidate_id": candidate_id,
        "batch_id": config.batch_id,
        "front_image_path": str(pair.front_path),
        "back_image_path": str(pair.back_path) if pair.has_back else None,
        "manufacturer": None,
        "set_key": None,
        "player_name": None,
        "team_name": None,
        "card_number": None,
        "variant": None,
        "condition": None,
        "condition_notes": None,
        "processing_mode": "error",
        "confidence": "unverified",
        "review_tier": TIER_NO_INFORMATION,
        "review_status": "pending",
        "is_duplicate": False,
        "duplicate_of": None,
        "is_pairing_disruption": pair.is_pairing_disruption,
        "raw_classification": None,
        "raw_extraction": None,
        "error": error,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }


def _write_candidates(
    candidates: list[dict],
    config: SingleSetConfig
) -> Path:
    """
    Writes candidates list to JSON file incrementally.
    Called after each processed pair for resume capability.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.batch_id}_candidates.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": config.batch_id,
                "source_path": str(config.source_path),
                "mode": "single_set",
                "total_candidates": len(candidates),
                "candidates": candidates
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


def _write_deferred(
    deferred_jobs: list[dict],
    config: SingleSetConfig
) -> Path:
    """
    Writes deferred jobs list to JSON file.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.batch_id}_deferred.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": config.batch_id,
                "total_deferred": len(deferred_jobs),
                "deferred_jobs": deferred_jobs
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


def _load_existing_candidates(
    output_dir: Path,
    batch_id: str
) -> dict:
    """
    Loads existing candidates from a previous interrupted run.
    Returns dict keyed by front_image_path for fast lookup.
    """
    path = output_dir / f"{batch_id}_candidates.json"

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            c["front_image_path"]: c
            for c in data.get("candidates", [])
        }
    except Exception as e:
        logger.warning(
            f"Could not load existing candidates for {batch_id}: {e}. "
            f"Starting fresh."
        )
        return {}


def _load_existing_deferred(
    output_dir: Path,
    batch_id: str
) -> list:
    """
    Loads existing deferred jobs from a previous interrupted run.
    """
    path = output_dir / f"{batch_id}_deferred.json"

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("deferred_jobs", [])
    except Exception as e:
        logger.warning(
            f"Could not load existing deferred jobs for {batch_id}: {e}. "
            f"Starting with empty deferred queue."
        )
        return []


def _generate_batch_id() -> str:
    """
    Generates a unique batch ID based on current UTC timestamp.
    Format: single_set_YYYYMMDD_HHMMSS
    """
    return (
        f"single_set_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )