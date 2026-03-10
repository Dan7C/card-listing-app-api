# Set name fuzzy matching against sets_config keys

import os
from difflib import get_close_matches
from dotenv import load_dotenv
from app.core.pipeline.results import SupportedResult, DiscoveryResult

load_dotenv()

# Minimum similarity threshold for a fuzzy match to be accepted.
# 0.8 means 80% similar - adjust in .env if too many valid matches are
# missed (lower) or too many false matches are accepted (raise).
FUZZY_MATCH_CUTOFF = float(os.getenv("FUZZY_MATCH_CUTOFF", "0.8"))


class FuzzyMatchResult:
    """Represents the result of a fuzzy match attempt."""

    def __init__(
        self,
        query: str,
        matched_key: str | None,
        was_exact: bool
    ):
        self.query = query
        self.matched_key = matched_key
        self.was_exact = was_exact
        self.was_fuzzy = matched_key is not None and not was_exact
        self.failed = matched_key is None

    def __repr__(self):
        if self.was_exact:
            return f"FuzzyMatchResult(exact: '{self.matched_key}')"
        elif self.was_fuzzy:
            return f"FuzzyMatchResult(fuzzy: '{self.query}' → '{self.matched_key}')"
        else:
            return f"FuzzyMatchResult(no match: '{self.query}')"


def match_set_key(
    query: str,
    known_keys: list[str]
) -> FuzzyMatchResult:
    """
    Attempts to match a query string against a list of known keys.

    First attempts an exact match (case-insensitive).
    If no exact match, attempts fuzzy matching at FUZZY_MATCH_CUTOFF.

    Args:
        query: The string returned by the LLM to match
        known_keys: List of valid keys and aliases to match against

    Returns:
        FuzzyMatchResult describing the outcome.
    """
    if not query or not known_keys:
        return FuzzyMatchResult(
            query=query,
            matched_key=None,
            was_exact=False
        )

    # normalise for case-insensitive comparison
    query_normalised = query.strip().lower()
    keys_normalised = {key.lower(): key for key in known_keys}

    # attempt exact match first
    if query_normalised in keys_normalised:
        original_key = keys_normalised[query_normalised]
        return FuzzyMatchResult(
            query=query,
            matched_key=original_key,
            was_exact=True
        )

    # attempt fuzzy match
    close_matches = get_close_matches(
        query_normalised,
        keys_normalised.keys(),
        n=1,
        cutoff=FUZZY_MATCH_CUTOFF
    )

    if close_matches:
        original_key = keys_normalised[close_matches[0]]
        return FuzzyMatchResult(
            query=query,
            matched_key=original_key,
            was_exact=False
        )

    return FuzzyMatchResult(
        query=query,
        matched_key=None,
        was_exact=False
    )


def get_manufacturer_keys(config: dict) -> list[str]:
    """
    Returns all matchable strings for manufacturer identification.
    Includes config keys and aliases for each manufacturer.

    Args:
        config: The full sets_config dictionary
    """
    keys = []
    for key, data in config["manufacturers"].items():
        keys.append(key)
        keys.extend(data.get("aliases", []))
    return keys


def get_set_keys(config: dict, manufacturer_key: str) -> list[str]:
    """
    Returns all matchable strings for set identification within a
    specific manufacturer. Includes config keys and aliases.

    Args:
        config: The full sets_config dictionary
        manufacturer_key: The matched manufacturer config key
    """
    sets = config["manufacturers"][manufacturer_key]["sets"]
    keys = []
    for key, data in sets.items():
        keys.append(key)
        keys.extend(data.get("aliases", []))
    return keys


def resolve_manufacturer_and_set(
    manufacturer_query: str,
    set_query: str,
    config: dict
) -> SupportedResult | DiscoveryResult:
    """
    Performs tiered matching — manufacturer first, then set within that
    manufacturer. Returns a SupportedResult if both match, or a
    DiscoveryResult with as much partial information as was gathered.

    Args:
        manufacturer_query: Manufacturer string returned by the LLM
        set_query: Set name string returned by the LLM
        config: The full sets_config dictionary
    """
    # step 1 - match manufacturer
    manufacturer_keys = get_manufacturer_keys(config)
    manufacturer_result = match_set_key(manufacturer_query, manufacturer_keys)

    if manufacturer_result.failed:
        return DiscoveryResult(
            reason="unknown_manufacturer",
            known_manufacturer=None
        )

    # resolve alias match back to config key
    manufacturer_key = _resolve_to_config_key(
        manufacturer_result.matched_key,
        config["manufacturers"]
    )

    # step 2 - match set within confirmed manufacturer only
    set_keys = get_set_keys(config, manufacturer_key)
    set_result = match_set_key(set_query, set_keys)

    if set_result.failed:
        return DiscoveryResult(
            reason="unknown_set",
            known_manufacturer=manufacturer_key
        )

    # resolve alias match back to config key
    set_key = _resolve_to_config_key(
        set_result.matched_key,
        config["manufacturers"][manufacturer_key]["sets"]
    )

    return SupportedResult(
        manufacturer=manufacturer_key,
        set_key=set_key
    )


def _resolve_to_config_key(matched_value: str, config_section: dict) -> str:
    """
    Resolves a matched value (which may be an alias) back to its parent
    config key.

    For example, if "topps uk" matched as an alias of the "topps" config
    entry, this returns "topps".

    Args:
        matched_value: The string that was matched (key or alias)
        config_section: The config dictionary section to search within
    """
    matched_lower = matched_value.lower()

    for key, data in config_section.items():
        if key.lower() == matched_lower:
            return key
        aliases = [a.lower() for a in data.get("aliases", [])]
        if matched_lower in aliases:
            return key

    # fallback - return as-is if no parent found
    # this shouldn't happen if called correctly but is safer than crashing
    return matched_value