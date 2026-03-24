# Classification call - determines manufacturer, set, face, and subject
# for a single card image. Used by Messy mode for every image, and by
# Mixed mode for unrecognised subdirectory samples (Call 0/1).

import logging
from pathlib import Path
from app.core.llm.client import LLMClient
from app.core.llm.prompt_loader import load_prompt
from app.utils.image import load_image_as_base64, get_mime_type, validate_image
from app.core.pipeline.results import ClassificationResult

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """
    Raised when a classification call cannot be completed.
    Distinct from LLMClientError - indicates a problem preparing
    the request rather than a failure in the API call itself.
    """
    pass


async def classify_image(
    image_path: Path,
    client: LLMClient
) -> ClassificationResult:
    """
    Classifies a single card image by calling the LLM classification prompt.
    Returns a ClassificationResult describing the card's manufacturer, set,
    face (front/back), subject, and whether it contains multiple cards.

    A failed or unparseable response returns a ClassificationResult with
    None fields rather than raising - the pipeline routes these to the
    review queue rather than crashing.

    Args:
        image_path: Path to the image file to classify
        client: Configured LLMClient instance to use for the API call

    Returns:
        ClassificationResult

    Raises:
        ClassificationError if the image cannot be loaded or validated
    """
    logger.info(f"Classifying image: {image_path.name}")

    _validate(image_path)

    mime_type = get_mime_type(image_path)
    image_b64 = load_image_as_base64(image_path)
    prompt = load_prompt("classification")

    result = await client.classify(
        front_image_b64=image_b64,
        mime_type=mime_type,
        prompt=prompt
    )

    logger.info(
        f"Classification result for {image_path.name}: "
        f"manufacturer={result.manufacturer}, "
        f"set={result.set_name}, "
        f"face={result.face}, "
        f"subject={result.subject}"
    )

    return result


def _validate(image_path: Path) -> None:
    """
    Validates that the image exists and is in a supported format.
    Raises ClassificationError if validation fails.
    """
    try:
        validate_image(image_path)
    except Exception as e:
        raise ClassificationError(
            f"Image validation failed for {image_path}: {e}"
        )