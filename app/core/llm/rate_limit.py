# Rate limit tracking for LLM API providers.
# Populated from response headers after each API call.
# In-memory only - no persistence between sessions.

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RateLimitStatus:
    """
    Represents the current rate limit state for an API provider.
    Populated from response headers after each API call.

    Attributes:
        limit:        Total requests allowed per day
        remaining:    Requests remaining in current window
        reset_at:     ISO timestamp when the limit resets
        last_updated: When this status was last populated from headers
        provider:     Name of the provider this status belongs to
    """
    limit: int
    remaining: int
    reset_at: str
    last_updated: datetime
    provider: str

    @property
    def is_exhausted(self) -> bool:
        """Returns True if no requests remain."""
        return self.remaining == 0

    @property
    def is_low(self) -> bool:
        """
        Returns True if remaining requests are below 50.
        Used to surface a warning indicator in the UI.
        """
        return self.remaining < 50

    @property
    def usage_percentage(self) -> float:
        """
        Returns percentage of daily limit consumed.
        Used to drive the progress bar in the UI.
        """
        if self.limit == 0:
            return 0.0
        return ((self.limit - self.remaining) / self.limit) * 100

    @property
    def remaining_percentage(self) -> float:
        """Returns percentage of daily limit remaining."""
        return 100.0 - self.usage_percentage

    def __repr__(self) -> str:
        return (
            f"RateLimitStatus("
            f"provider='{self.provider}', "
            f"remaining={self.remaining}/{self.limit}, "
            f"usage={self.usage_percentage:.1f}%, "
            f"resets='{self.reset_at}')"
        )


def extract_groq_rate_limit(
    headers: dict,
    last_updated: datetime | None = None
) -> RateLimitStatus | None:
    """
    Extracts rate limit information from Groq API response headers.

    Groq returns the following headers on every response:
        x-ratelimit-limit-requests:     daily request limit
        x-ratelimit-remaining-requests: requests remaining today
        x-ratelimit-reset-requests:     ISO timestamp of next reset

    Args:
        headers:      Response headers dictionary from the Groq API
        last_updated: Timestamp to record as last_updated.
                      Defaults to current UTC time if not provided.

    Returns:
        RateLimitStatus if all headers are present and valid.
        None if headers are absent or cannot be parsed — this is
        expected on the first call of a session before any response
        has been received.
    """
    try:
        limit = int(headers.get("x-ratelimit-limit-requests", 0))
        remaining = int(headers.get("x-ratelimit-remaining-requests", 0))
        reset_at = headers.get("x-ratelimit-reset-requests", "")

        if not limit or not reset_at:
            logger.debug(
                "Rate limit headers absent or incomplete — "
                "status will be unknown until first API call completes."
            )
            return None

        return RateLimitStatus(
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            last_updated=last_updated or datetime.now(timezone.utc),
            provider="groq"
        )

    except Exception as e:
        logger.warning(f"Failed to parse Groq rate limit headers: {e}")
        return None


def estimate_batch_calls(total_images: int) -> int:
    """
    Estimates the number of API calls required to process a batch.

    Calculation:
        classification_calls = total_images (one per image)
        extraction_calls     = total_images // 2 (one per pair, assuming
                               front-back sets. Conservative estimate —
                               front-only sets will use fewer calls.)

    This is intentionally conservative — better to warn the user of a
    potential limit issue than to silently exhaust their daily quota.

    Args:
        total_images: Number of images in the batch

    Returns:
        Estimated total API calls required
    """
    classification_calls = total_images
    extraction_calls = total_images // 2
    return classification_calls + extraction_calls


def check_batch_feasibility(
    total_images: int,
    rate_limit_status: RateLimitStatus | None
) -> tuple[bool, str]:
    """
    Checks whether a batch can be completed within the current rate limit.

    Args:
        total_images:       Number of images to process
        rate_limit_status:  Current rate limit state, or None if unknown

    Returns:
        Tuple of (can_complete: bool, message: str)
        can_complete is True if the batch fits within remaining requests.
        message describes the situation for display in the UI.
    """
    estimated = estimate_batch_calls(total_images)

    if rate_limit_status is None:
        return True, (
            f"Rate limit status unknown — will be confirmed after "
            f"first API call. Estimated calls required: {estimated}."
        )

    remaining = rate_limit_status.remaining

    if estimated <= remaining:
        return True, (
            f"Batch requires approximately {estimated} API calls. "
            f"{remaining} remaining today."
        )

    overage = estimated - remaining
    return False, (
        f"This batch requires approximately {estimated} API calls "
        f"but only {remaining} remain today. "
        f"Processing will pause when the limit is reached "
        f"({overage} calls will be deferred to tomorrow). "
        f"Rate limit resets at {rate_limit_status.reset_at}."
    )