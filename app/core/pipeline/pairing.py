# Image pairing logic for front/back card matching.
# Primary strategy: positional pairing assuming ABABAB scan order.
# Enhancement: classification-informed disruption handling using face
# field from ClassificationResult to detect and recover from sequence
# breaks.
#
# Pairing scenarios handled:
#   Front-back mode:
#     Scenario 1: Consecutive fronts
#                 → first front processed as front-only, flagged
#                 → second front starts new pair
#     Scenario 2: Orphaned back
#                 → routed to review queue for manual pairing or discard
#
#   Front-only mode:
#     Scenario 3: Unexpected back image found
#                 → routed to review queue as orphaned back
#                 → same handling as Scenario 2
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
    Output of a pairing run.

    Contains:
        pairs:          completed CardPairs ready for extraction
        orphaned_backs: back images that could not be paired,
                        routed to review queue for manual pairing
        disruptions:    log of pairing disruptions encountered
    """

    def __init__(self):
        self.pairs: list[CardPair] = []
        self.orphaned_backs: list[Path] = []
        self.disruptions: list[str] = []

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    @property
    def total_orphaned_backs(self) -> int:
        return len(self.orphaned_backs)

    @property
    def total_disruptions(self) -> int:
        return len(self.disruptions)

    def add_pair(self, pair: CardPair) -> None:
        self.pairs.append(pair)

    def add_orphaned_back(self, path: Path, reason: str) -> None:
        self.orphaned_backs.append(path)
        self._add_disruption(
            f"Orphaned back routed to review queue: {path.name}. "
            f"Reason: {reason}"
        )

    def _add_disruption(self, message: str) -> None:
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
                         May contain unusable results which are skipped.

    Returns:
        Inferred ImageMode.
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

    Args:
        images:          List of image paths in filesystem order
        classifications: ClassificationResults in same order as images.
                         Must be same length as images.
        image_mode:      Declared or inferred image mode for this set.
                         Must be FRONT_ONLY or FRONT_BACK - resolve
                         UNKNOWN before calling this function.
        duplicate_map:   Dict mapping duplicate path → original path.
                         Duplicates are excluded from pairing.

    Returns:
        PairingResult containing pairs, orphaned backs, and disruption log.

    Raises:
        ValueError if images and classifications are different lengths,
        or if image_mode is UNKNOWN.
    """
    if len(images) != len(classifications):
        raise ValueError(
            f"images and classifications must be same length. "
            f"Got {len(images)} images and "
            f"{len(classifications)} classifications."
        )

    if image_mode == ImageMode.UNKNOWN:
        raise ValueError(
            "image_mode must be resolved before calling pair_images. "
            "Call infer_image_mode() first."
        )

    result = PairingResult()

    if image_mode == ImageMode.FRONT_ONLY:
        _pair_front_only(images, classifications, duplicate_map, result)
    else:
        _pair_front_back(images, classifications, duplicate_map, result)

    logger.info(
        f"Pairing complete: {result.total_pairs} pairs, "
        f"{result.total_orphaned_backs} orphaned backs, "
        f"{result.total_disruptions} disruptions"
    )

    return result


def _is_skippable(
    image: Path,
    classification: ClassificationResult,
    duplicate_map: dict[str, str]
) -> bool:
    """
    Returns True if an image should be skipped during pairing.
    Skips duplicates and unusable classifications.
    """
    if str(image) in duplicate_map:
        logger.debug(f"Skipping duplicate: {image.name}")
        return True

    if not classification.is_usable:
        logger.info(
            f"Skipping unusable image: {image.name} "
            f"(contains_multiple={classification.contains_multiple_cards}, "
            f"subject={classification.subject})"
        )
        return True

    return False


def _pair_front_only(
    images: list[Path],
    classifications: list[ClassificationResult],
    duplicate_map: dict[str, str],
    result: PairingResult
) -> None:
    """
    Processes a front-only set.

    Each valid front image becomes a front-only CardPair.

    Scenario 3: If a back image is found during front-only processing,
    it is routed to the review queue as an orphaned back. The user can
    pair it with a front-only card or discard it.
    """
    for image, classification in zip(images, classifications):
        if _is_skippable(image, classification, duplicate_map):
            continue

        if classification.is_back:
            result.add_orphaned_back(
                path=image,
                reason="unexpected back image found during front-only processing"
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
            sees front   → store as pending_front → AWAITING_BACK
            sees back    → Scenario 2: orphaned back routed to review queue
                           stay in AWAITING_FRONT
            unusable     → skip, stay in AWAITING_FRONT
            duplicate    → skip, stay in AWAITING_FRONT

        AWAITING_BACK:
            sees back    → complete pair with pending_front
                           → AWAITING_FRONT
            sees front   → Scenario 1: consecutive fronts
                           pending_front processed as front-only, flagged
                           new front becomes pending_front
                           stay in AWAITING_BACK
            unusable     → skip, stay in AWAITING_BACK
            duplicate    → skip, stay in AWAITING_BACK

    End of sequence:
        pending_front remaining → treated as consecutive front at boundary
                                  processed as front-only, flagged
    """
    state = PairingState.AWAITING_FRONT
    pending_front: Path | None = None
    pending_classification: ClassificationResult | None = None

    for image, classification in zip(images, classifications):

        if _is_skippable(image, classification, duplicate_map):
            continue

        face = classification.face

        if state == PairingState.AWAITING_FRONT:
            if face == "front":
                pending_front = image
                pending_classification = classification
                state = PairingState.AWAITING_BACK

            elif face == "back":
                # Scenario 2: orphaned back
                result.add_orphaned_back(
                    path=image,
                    reason="back image found while awaiting front"
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
                # Scenario 1: consecutive fronts
                result._add_disruption(
                    f"Consecutive fronts detected. "
                    f"{pending_front.name} processed as front-only. "
                    f"{image.name} starts new pair."
                )
                result.add_pair(CardPair(
                    front_path=pending_front,
                    back_path=None,
                    classification=pending_classification,
                    is_pairing_disruption=True
                ))
                pending_front = image
                pending_classification = classification

    # end of sequence - treat remaining pending front as consecutive
    # front at boundary
    if pending_front is not None:
        result._add_disruption(
            f"No back found for final image. "
            f"{pending_front.name} processed as front-only."
        )
        result.add_pair(CardPair(
            front_path=pending_front,
            back_path=None,
            classification=pending_classification,
            is_pairing_disruption=True
        ))