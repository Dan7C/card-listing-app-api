# Pipeline routing result types - used across all pipeline stages

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SupportedResult:
    """
    Represents a successful config match.
    Card will be processed using constrained prompting from sets_config.
    """
    manufacturer: str
    set_key: str


@dataclass
class DiscoveryResult:
    """
    Represents a failed config match routed to Discovery mode.
    Card will be processed using unconstrained prompting.
    Carries any partial information gathered before matching failed.
    """
    reason: str
    known_manufacturer: str | None = None


@dataclass
class ClassificationResult:
    """
    Represents the output of a classification API call (Call 1).
    Produced by app/core/llm/classification.py.
    """
    manufacturer: str | None
    set_name: str | None
    face: str | None                    # "front" or "back"
    subject: str | None                 # "player", "team_badge" etc.
    contains_multiple_cards: bool = False
    raw_response: str = ""              # original LLM response for logging

    @property
    def is_front(self) -> bool:
        return self.face == "front"

    @property
    def is_back(self) -> bool:
        return self.face == "back"

    @property
    def is_usable(self) -> bool:
        """
        Returns True if the classification result has enough information
        to proceed with matching and extraction.
        """
        return (
            not self.contains_multiple_cards
            and self.face is not None
            and self.subject is not None
        )


@dataclass
class ExtractionResult:
    """
    Represents the output of an extraction API call (Call 2 or 3).
    Produced by app/core/llm/extraction.py.
    """
    player_name: str | None
    team_name: str | None
    card_number: str | None
    variant: str | None
    condition_observations: list[str] | None
    condition_recommendation: str | None
    condition_confidence: str | None
    condition_notes: str | None
    back_content_summary: str | None
    subject: str | None
    processing_mode: str = ""           # "supported" or "discovery"
    raw_response: str = ""              # original LLM response for logging
    confidence: str = "unverified"      # "high" or "unverified"


@dataclass
class CardPair:
    """
    Represents a paired front and back image ready for extraction.
    Back image is optional - None indicates a front-only card.
    orphaned_back_path is set when a back image could not be paired
    and is routed to the review queue for manual pairing.
    """
    front_path: Path
    back_path: Path | None = None
    classification: ClassificationResult | None = None
    is_pairing_disruption: bool = False
    orphaned_back_path: Path | None = None

    @property
    def has_back(self) -> bool:
        return self.back_path is not None

    @property
    def is_front_only(self) -> bool:
        return self.back_path is None