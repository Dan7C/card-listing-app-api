# Extraction calls - extracts structured card metadata from images.
# Two extraction paths:
#   Supported: constrained prompting using sets_config values (Call 2)
#   Discovery: unconstrained prompting, flagged as unverified (Call 3)

import json
import logging
from pathlib import Path
from app.core.llm.client import LLMClient
from app.core.llm.prompt_loader import load_prompt_with_variables, load_prompt
from app.utils.image import load_image_as_base64, get_mime_type, validate_image
from app.core.pipeline.results import ExtractionResult, CardPair

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """
    Raised when an extraction call cannot be completed.
    Distinct from LLMClientError - indicates a problem preparing
    the request rather than a failure in the API call itself.
    """
    pass


async def extract_supported(
    pair: CardPair,
    manufacturer_config: dict,
    set_config: dict,
    client: LLMClient
) -> ExtractionResult:
    """
    Extracts structured metadata from a card image using constrained
    prompting. Used when both manufacturer and set have been matched
    to a known sets_config entry.

    Injects set-specific values into the prompt template so the model
    is guided toward known valid values for this set.

    Args:
        pair: CardPair containing front and optional back image paths
        manufacturer_config: The matched manufacturer entry from sets_config
        set_config: The matched set entry from sets_config
        client: Configured LLMClient instance

    Returns:
        ExtractionResult with confidence="high"

    Raises:
        ExtractionError if images cannot be loaded or validated
    """
    logger.info(
        f"Extracting supported card: "
        f"{manufacturer_config['display_name']} - "
        f"{set_config['display_name']}"
    )

    _validate_pair(pair)

    prompt = load_prompt_with_variables(
        "extraction_supported",
        _build_prompt_variables(manufacturer_config, set_config)
    )

    front_b64 = load_image_as_base64(pair.front_path)
    mime_type = get_mime_type(pair.front_path)
    back_b64 = load_image_as_base64(pair.back_path) if pair.has_back else None

    result = await client.extract(
        front_image_b64=front_b64,
        mime_type=mime_type,
        prompt=prompt,
        back_image_b64=back_b64
    )

    result.processing_mode = "supported"
    result.confidence = "high"

    logger.info(
        f"Extraction result: "
        f"player={result.player_name}, "
        f"team={result.team_name}, "
        f"variant={result.variant}, "
        f"condition={result.condition}"
    )

    return result


async def extract_discovery(
    pair: CardPair,
    client: LLMClient
) -> ExtractionResult:
    """
    Extracts structured metadata from a card image using unconstrained
    prompting. Used when the card cannot be matched to a known set.

    Results are flagged as unverified and always require user review
    before listing generation.

    Args:
        pair: CardPair containing front and optional back image paths
        client: Configured LLMClient instance

    Returns:
        ExtractionResult with confidence="unverified"

    Raises:
        ExtractionError if images cannot be loaded or validated
    """
    logger.info(
        f"Extracting discovery card: {pair.front_path.name}"
    )

    _validate_pair(pair)

    prompt = load_prompt("extraction_discovery")

    front_b64 = load_image_as_base64(pair.front_path)
    mime_type = get_mime_type(pair.front_path)
    back_b64 = load_image_as_base64(pair.back_path) if pair.has_back else None

    result = await client.extract(
        front_image_b64=front_b64,
        mime_type=mime_type,
        prompt=prompt,
        back_image_b64=back_b64
    )

    result.processing_mode = "discovery"
    result.confidence = "unverified"

    logger.info(
        f"Discovery extraction result: "
        f"player={result.player_name}, "
        f"team={result.team_name}"
    )

    return result


def _validate_pair(pair: CardPair) -> None:
    """
    Validates that the front image exists and is supported.
    Also validates the back image if present.
    Raises ExtractionError if validation fails.
    """
    try:
        validate_image(pair.front_path)
        if pair.has_back:
            validate_image(pair.back_path)
    except Exception as e:
        raise ExtractionError(
            f"Image validation failed for pair "
            f"{pair.front_path.name}: {e}"
        )


def _build_prompt_variables(
    manufacturer_config: dict,
    set_config: dict
) -> dict:
    """
    Builds the variable dictionary for supported extraction prompt
    substitution from manufacturer and set config entries.

    These variables are injected into the extraction_supported prompt
    template to constrain the model toward known valid values.
    """
    return {
        "manufacturer_display_name": manufacturer_config["display_name"],
        "set_display_name": set_config["display_name"],
        "year": set_config.get("year", "unknown"),
        "set_type": set_config.get("set_type", "card"),
        "known_variants": ", ".join(set_config.get("known_variants", [])),
        "supported_subjects": ", ".join(
            set_config.get("supported_subjects", [])
        )
    }