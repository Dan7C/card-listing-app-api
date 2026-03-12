# Single-Set mode orchestrator.
# All images in the directory are assumed to belong to the same set.
# Supports optional manufacturer/set declaration by the user,
# directory name matching, filename matching, and discovery mode fallback.
# Outputs results to JSON for pipeline validation.

import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from app.core.llm.client import LLMClient
from app.core.llm.extraction import extract_supported, extract_discovery
from app.core.pipeline.pairing import pair_images
from app.core.pipeline.results import (
    SupportedResult,
    DiscoveryResult,
    CardPair
)
from app.utils.file_walker import get_image_files, ImageSortOrder
from app.utils.image import load_image_as_base64, hash_image
from app.utils.fuzzy_match import resolve_manufacturer_and_set
from app.core.llm.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# Review tier constants
TIER_DUPLICATE = 1
TIER_INCOMPLETE = 2
TIER_LOW_CONFIDENCE = 3
TIER_STANDARD = 4


class SingleSetConfig:
    """
    Configuration for a Single-Set mode processing run.

    Args:
        source_path: Path to directory or single image file to process
        manufacturer_key: Optional manufacturer config key declared by user
        set_key: Optional set config key declared by user
        sets_config: The full sets_config dictionary
        max_depth: Directory scan depth (0=flat, -1=unlimited)
        sort_order: Image ordering for pairing correctness
        enable_pairing: Whether to pair front/back images
        output_dir: Directory to write JSON output to
        batch_id: Unique identifier for this run
        resume: Whether to resume a previous interrupted run
    """

    def __init__(
        self,
        source_path: Path,
        sets_config: dict,
        manufacturer_key: str | None = None,
        set_key: str | None = None,
        max_depth: int = 0,
        sort_order: ImageSortOrder = ImageSortOrder.FILESYSTEM,
        enable_pairing: bool = True,
        output_dir: Path = Path("outputs"),
        batch_id: str | None = None,
        resume: bool = False
    ):
        self.source_path = source_path
        self.sets_config = sets_config
        self.manufacturer_key = manufacturer_key
        self.set_key = set_key
        self.max_depth = max_depth
        self.sort_order = sort_order
        self.enable_pairing = enable_pairing
        self.output_dir = output_dir
        self.batch_id = batch_id or _generate_batch_id()
        self.resume = resume


class SingleSetError(Exception):
    """
    Raised when Single-Set mode cannot be initialised or run.
    Distinct from per-image errors which are captured in results.
    """
    pass


async def run_single_set(
    config: SingleSetConfig,
    client: LLMClient
) -> Path:
    """
    Runs the Single-Set mode pipeline.

    Flow:
        1. Resolve manufacturer and set from user input, directory name,
           filename, or discovery mode fallback
        2. Collect images from source path
        3. Build hash registry and detect duplicates
        4. Pair images if enabled
        5. Extract metadata for each pair
        6. Write results to JSON
        7. Return path to output JSON file

    Args:
        config: SingleSetConfig instance describing the run
        client: Configured LLMClient instance

    Returns:
        Path to the output JSON file

    Raises:
        SingleSetError if the run cannot be initialised
    """
    logger.info(
        f"Starting Single-Set run {config.batch_id} "
        f"on {config.source_path}"
    )

    # load existing results if resuming
    existing_results = {}
    if config.resume:
        existing_results = _load_existing_results(
            config.output_dir,
            config.batch_id
        )
        logger.info(
            f"Resuming run {config.batch_id} - "
            f"{len(existing_results)} images already processed"
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

    # build hash registry and identify duplicates
    hash_registry = {}
    duplicate_map = {}
    for image_path in images:
        image_hash = hash_image(image_path)
        if image_hash in hash_registry:
            duplicate_map[str(image_path)] = str(
                hash_registry[image_hash]
            )
            logger.info(
                f"Duplicate detected: {image_path.name} is identical "
                f"to {hash_registry[image_hash].name}"
            )
        else:
            hash_registry[image_hash] = image_path

    # pair images
    if config.enable_pairing:
        pairs = pair_images(images, duplicate_map)
    else:
        pairs = [
            CardPair(front_path=img)
            for img in images
            if str(img) not in duplicate_map
        ]

    logger.info(
        f"{len(pairs)} pairs to process, "
        f"{len(duplicate_map)} duplicates detected"
    )

    # process pairs
    results = list(existing_results.values())
    processed_paths = set(existing_results.keys())

    for pair in pairs:
        pair_key = str(pair.front_path)

        # skip if already processed in a previous run
        if pair_key in processed_paths:
            logger.debug(f"Skipping already processed: {pair.front_path.name}")
            continue

        result = await _process_pair(
            pair=pair,
            set_context=set_context,
            config=config,
            client=client
        )
        results.append(result)

        # write incrementally for resume capability
        _write_results(results, config)

    # process duplicates - clone result from original
    for duplicate_path, original_path in duplicate_map.items():
        original_result = next(
            (r for r in results if r["front_image_path"] == original_path),
            None
        )
        if original_result:
            cloned = _clone_as_duplicate(
                original_result,
                duplicate_path
            )
            results.append(cloned)

    # final write with duplicates included
    output_path = _write_results(results, config)

    logger.info(
        f"Single-Set run {config.batch_id} complete. "
        f"{len(results)} results written to {output_path}"
    )

    return output_path


async def _process_pair(
    pair: CardPair,
    set_context: SupportedResult | DiscoveryResult,
    config: SingleSetConfig,
    client: LLMClient
) -> dict:
    """
    Processes a single CardPair through the appropriate extraction path
    and returns a result dictionary ready for JSON serialisation.
    """
    logger.info(f"Processing: {pair.front_path.name}")

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
            "batch_id": config.batch_id,
            "front_image_path": str(pair.front_path),
            "back_image_path": str(pair.back_path) if pair.has_back else None,
            "manufacturer": set_context.manufacturer
                if isinstance(set_context, SupportedResult) else
                set_context.known_manufacturer,
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
            "duplicate_of": None,
            "raw_extraction": extraction.raw_response,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to process {pair.front_path.name}: {e}")
        return {
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
            "review_tier": TIER_INCOMPLETE,
            "review_status": "pending",
            "duplicate_of": None,
            "raw_extraction": None,
            "error": str(e),
            "processed_at": datetime.now(timezone.utc).isoformat()
        }


def _resolve_set_context(
    config: SingleSetConfig,
    images: list[Path]
) -> SupportedResult | DiscoveryResult:
    """
    Resolves manufacturer and set context using a confidence hierarchy:

        1. User declared both manufacturer and set → SupportedResult directly
        2. User declared manufacturer only → DiscoveryResult with known_manufacturer
        3. Read from directory name → fuzzy match against config
        4. Read from filename → fuzzy match against config
        5. No information → DiscoveryResult with no context

    Returns SupportedResult if a full config match is found,
    DiscoveryResult otherwise.
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
    if config.manufacturer_key and not config.set_key:
        logger.info(
            f"Manufacturer declared by user, set unknown: "
            f"{config.manufacturer_key}"
        )
        return DiscoveryResult(
            reason="unknown_set",
            known_manufacturer=config.manufacturer_key
        )

    # level 3 - read from directory name
    if config.source_path.is_dir():
        dir_name = config.source_path.name
        result = resolve_manufacturer_and_set(
            manufacturer_query=dir_name,
            set_query=dir_name,
            config=config.sets_config
        )
        if isinstance(result, SupportedResult):
            logger.info(
                f"Set context from directory name '{dir_name}': "
                f"{result.manufacturer} / {result.set_key}"
            )
            return result
        logger.info(
            f"Directory name '{dir_name}' did not match any known set"
        )

    # level 4 - read from filename (single file or first image in directory)
    source_file = (
        config.source_path
        if config.source_path.is_file()
        else next(iter(images), None)
    )
    if source_file:
        filename_stem = source_file.stem
        result = resolve_manufacturer_and_set(
            manufacturer_query=filename_stem,
            set_query=filename_stem,
            config=config.sets_config
        )
        if isinstance(result, SupportedResult):
            logger.info(
                f"Set context from filename '{filename_stem}': "
                f"{result.manufacturer} / {result.set_key}"
            )
            return result
        logger.info(
            f"Filename '{filename_stem}' did not match any known set"
        )

    # level 5 - no context available
    logger.info("No set context found - routing to discovery mode")
    return DiscoveryResult(reason="no_context")


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

        Tier 1 - duplicate:       handled separately via _clone_as_duplicate
        Tier 2 - incomplete:      any required field is None
        Tier 3 - low confidence:  discovery mode or unverified confidence
        Tier 4 - standard:        high confidence supported extraction

    Args:
        extraction: ExtractionResult from the LLM
        set_context: SupportedResult or DiscoveryResult for this run
    """
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


def _clone_as_duplicate(
    original_result: dict,
    duplicate_path: str
) -> dict:
    """
    Clones an existing result for a duplicate image.
    Sets review_tier to TIER_DUPLICATE and links to the original.
    """
    cloned = original_result.copy()
    cloned["front_image_path"] = duplicate_path
    cloned["back_image_path"] = None
    cloned["review_tier"] = TIER_DUPLICATE
    cloned["review_status"] = "pending"
    cloned["duplicate_of"] = original_result["front_image_path"]
    cloned["processed_at"] = datetime.now(timezone.utc).isoformat()
    return cloned


def _write_results(results: list[dict], config: SingleSetConfig) -> Path:
    """
    Writes results list to a JSON file in the output directory.
    Called incrementally after each image and on completion,
    enabling resume capability if the run is interrupted.

    Output filename format: {batch_id}_results.json
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{config.batch_id}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": config.batch_id,
                "source_path": str(config.source_path),
                "mode": "single_set",
                "total_results": len(results),
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


def _load_existing_results(
    output_dir: Path,
    batch_id: str
) -> dict:
    """
    Loads existing results from a previous interrupted run.
    Returns a dictionary keyed by front_image_path for fast lookup.
    Returns empty dict if no existing results found.
    """
    output_path = output_dir / f"{batch_id}_results.json"

    if not output_path.exists():
        logger.info(
            f"No existing results found for batch {batch_id}"
        )
        return {}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = {
            r["front_image_path"]: r
            for r in data.get("results", [])
        }
        return results
    except Exception as e:
        logger.warning(
            f"Could not load existing results for {batch_id}: {e}. "
            f"Starting fresh."
        )
        return {}


def _generate_batch_id() -> str:
    """
    Generates a unique batch ID based on current UTC timestamp.
    Format: single_set_YYYYMMDD_HHMMSS
    """
    return f"single_set_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"