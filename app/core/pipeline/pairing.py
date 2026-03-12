# Image pairing logic for front/back card matching.
# Primary strategy: positional pairing assuming ABABAB scan order.
# Enhancement: classification-informed disruption handling using face
# field from ClassificationResult to detect and recover from sequence
# breaks.
#
# Future enhancement: filename-based pairing for unordered datasets.
# See edge case register for details.

import logging
from pathlib import Path
from enum import Enum
from app.core.pipeline.results import CardPair, ClassificationResult

logger = logging.getLogger(__name__)


class ImageMode(Enum):
    """
    Declares whether a set contains front images only, or front and
    back images in alternating order.

    FRONT_ONLY:   every image is a card front, no pairing attempted
    FRONT_BACK:   images alternate front/back (ABABAB), pairing enabled
    UNKNOWN:      infer from first two valid classification results
    """
    FRONT_ONLY = "front_only"
    FRONT_BACK = "front_back"
    UNKNOWN = "unknown"


class PairingState(Enum):
    """
    Internal state machine states for positional pairing.

    AWAITING_FRONT: expecting the next image to be a front
    AWAITING_BACK:  a front has been seen, expecting a back
    """
    AWAITING_FRONT = "awaiting_front"
    AWAITING_BACK = "awaiting_back"


class PairingResult:
    """
    Output of a pairing run. Contains completed pairs, front-only
    cards, and a log of any disruptions encountered.
    """

    def __init__(self):
        self.pairs: list[CardPair] = []
        self.disruptions: list[str] = []

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    @property
    def total_disruptions(self) -> int:
        return len(self.disruptions)

    def add_pair(self, pair: CardPair) -> None:
        self.pairs.append(pair)

    def add_disruption(self, message: str) -> None:
        self.disruptions.append(message)
        logger.warning(f"Pairing disruption: {message}")


def infer_image_mode(
    classifications: list[ClassificationResult]
) -> ImageMode:
    """
    Infers image mode from the first two valid classification results.
    A valid result is one where is_usable is True.

    Logic:
        first=front, second=back  → FRONT_BACK
        first=front, second=front → FRONT_ONLY
        insufficient valid results → FRONT_ONLY (safe default)

    Args:
        classifications: List of ClassificationResults in image order.
                         May contain unusable results (full set images,
                         unsupported subjects etc.) which are skipped.

    Returns:
        ImageMode inferred from the first two valid results.
    """
    valid = [c for c in classifications if c.is_usable]

    if len(valid) < 2:
        logger.warning(
            "Insufficient valid classifications to infer image mode. "
            "Defaulting to FRONT_ONLY."
        )
        return ImageMode.FRONT_ONLY

    first, second = valid[0], valid[1]

    if first.is_front and second.is_back:
        logger.info("Image mode inferred: FRONT_BACK")
        return ImageMode.FRONT_BACK

    if first.is_front and second.is_front:
        logger.info("Image mode inferred: FRONT_ONLY")
        return ImageMode.FRONT_ONLY

    # unexpected sequence - default to front only
    logger.warning(
        f"Unexpected initial sequence: {first.face}, {second.face}. "
        f"Defaulting to FRONT_ONLY."
    )
    return ImageMode.FRONT_ONLY


def pair_images(
    images: list[Path],
    classifications: list[ClassificationResult],
    image_mode: ImageMode,
    duplicate_map: dict[str, str]
) -> PairingResult:
    """
    Pairs images using positional pairing with classification-informed
    disruption handling.

    Skips images that are:
        - Exact duplicates (present in duplicate_map)
        - Unusable classifications (full set images, unsupported subjects)

    Disruption handling:
        Consecutive fronts:  first front processed as front-only,
                             second front starts a new pair
        Consecutive backs:   orphaned back logged and discarded
        Odd count:           final front processed as front-only
                             with is_flagged=True

    Args:
        images: List of image paths in filesystem order
        classifications: ClassificationResults in same order as images.
                         Must be same length as images.
        image_mode: Declared or inferred image mode for this set
        duplicate_map: Dict mapping duplicate path → original path.
                       Duplicates are excluded from pairing.

    Returns:
        PairingResult containing pairs and disruption log
    """
    if len(images) != len(classifications):
        raise ValueError(
            f"images and classifications must be same length. "
            f"Got {len(images)} images and "
            f"{len(classifications)} classifications."
        )

    result = PairingResult()

    if image_mode == ImageMode.FRONT_ONLY:
        _pair_front_only(images, classifications, duplicate_map, result)
    else:
        _pair_front_back(images, classifications, duplicate_map, result)

    logger.info(
        f"Pairing complete: {result.total_pairs} pairs, "
        f"{result.total_disruptions} disruptions"
    )

    return result


def _pair_front_only(
    images: list[Path],
    classifications: list[ClassificationResult],
    duplicate_map: dict[str, str],
    result: PairingResult
) -> None:
    """
    Processes a front-only set - each valid image becomes a front-only
    CardPair with no back image.
    """
    for image, classification in zip(images, classifications):
        if str(image) in duplicate_map:
            continue

        if not classification.is_usable:
            logger.info(
                f"Skipping unusable image: {image.name} "
                f"(contains_multiple={classification.contains_multiple_cards}, "
                f"subject={classification.subject})"
            )
            continue

        result.add_pair(CardPair(
            front_path=image,
            back_path=None,
            classification=classification
        ))


def _pair_front_back(
    images: list[Path],
    classifications: list[ClassificationResult],
    duplicate_map: dict[str, str],
    result: PairingResult
) -> None:
    """
    Processes a front-back set using a state machine.
    Assumes ABABAB positional order, uses face field from classification
    to detect and recover from disruptions.

    State machine:
        AWAITING_FRONT:
            sees front  → store as pending_front, → AWAITING_BACK
            sees back   → orphaned back, log disruption, stay in state
            unusable    → skip, stay in state

        AWAITING_BACK:
            sees back   → complete pair with pending_front → AWAITING_FRONT
            sees front  → pending_front processed as front-only,
                          store new front as pending_front, stay in state
            unusable    → skip, stay in state
    """
    state = PairingState.AWAITING_FRONT
    pending_front: Path | None = None
    pending_classification: ClassificationResult | None = None

    for image, classification in zip(images, classifications):

        # skip duplicates
        if str(image) in duplicate_map:
            logger.debug(f"Skipping duplicate: {image.name}")
            continue

        # skip unusable images
        if not classification.is_usable:
            logger.info(
                f"Skipping unusable image: {image.name} "
                f"(contains_multiple={classification.contains_multiple_cards}, "
                f"subject={classification.subject})"
            )
            continue

        face = classification.face

        if state == PairingState.AWAITING_FRONT:
            if face == "front":
                pending_front = image
                pending_classification = classification
                state = PairingState.AWAITING_BACK

            elif face == "back":
                result.add_disruption(
                    f"Orphaned back image discarded: {image.name}. "
                    f"Expected front."
                )

        elif state == PairingState.AWAITING_BACK:
            if face == "back":
                result.add_pair(CardPair(
                    front_path=pending_front,
                    back_path=image,
                    classification=pending_classification
                ))
                pending_front = None
                pending_classification = None
                state = PairingState.AWAITING_FRONT

            elif face == "front":
                result.add_disruption(
                    f"Consecutive fronts detected. "
                    f"{pending_front.name} processed as front-only. "
                    f"{image.name} starts new pair."
                )
                result.add_pair(CardPair(
                    front_path=pending_front,
                    back_path=None,
                    classification=pending_classification,
                    is_flagged=True
                ))
                pending_front = image
                pending_classification = classification

    # handle remaining pending front after all images processed
    if pending_front is not None:
        result.add_disruption(
            f"Odd number of images. "
            f"{pending_front.name} processed as front-only."
        )
        result.add_pair(CardPair(
            front_path=pending_front,
            back_path=None,
            classification=pending_classification,
            is_flagged=True
        ))