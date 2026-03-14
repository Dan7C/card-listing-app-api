# Set name fuzzy matching against sets_config keys.
# Supports tiered manufacturer → set matching with alias resolution.
# All alias resolution is handled internally - callers always receive
# the actual config key from FuzzyMatchResult.matched_key.

import os
from difflib import get_close_matches
from dotenv import load_dotenv
from app.core.pipeline.results import SupportedResult, DiscoveryResult

load_dotenv()

FUZZY_MATCH_CUTOFF = float(os.getenv("FUZZY_MATCH_CUTOFF", "0.8"))


class FuzzyMatchResult:
    """
    Represents the result of a single fuzzy match attempt.

    matched_key is always the actual config key, never an alias.
    matched_via_alias contains the alias string if the match was made
    via an alias rather than the key directly — useful for presenting
    confirmation prompts to the user.

    Confidence levels:
        Exact match on config key:   was_exact=True,  matched_via_alias=None
        Fuzzy match on config key:   was_exact=False, matched_via_alias=None
        Exact match via alias:       was_exact=True,  matched_via_alias set
        Fuzzy match via alias:       was_exact=False, matched_via_alias set
        No match:                    failed=True
    """

    def __init__(
        self,
        query: str,
        matched_key: str | None,
        was_exact: bool,
        matched_via_alias: str | None = None
    ):
        self.query = query
        self.matched_key = matched_key
        self.was_exact = was_exact
        self.matched_via_alias = matched_via_alias
        self.was_fuzzy = matched_key is not None and not was_exact
        self.failed = matched_key is None

    @property
    def confidence_description(self) -> str:
        """
        Returns a human-readable description of match confidence.
        Used for building user-facing confirmation prompts.
        """
        if self.failed:
            return f"No match found for '{self.query}'"

        if self.was_exact and not self.matched_via_alias:
            return f"Exact match: '{self.query}' → '{self.matched_key}'"

        if self.was_exact and self.matched_via_alias:
            return (
                f"Matched via alias: '{self.query}' → "
                f"'{self.matched_via_alias}' → '{self.matched_key}'"
            )

        if self.was_fuzzy and not self.matched_via_alias:
            return (
                f"Fuzzy match: '{self.query}' → '{self.matched_key}' "
                f"(please confirm)"
            )

        # fuzzy match via alias - lowest confidence
        return (
            f"Fuzzy match via alias: '{self.query}' → "
            f"'{self.matched_via_alias}' → '{self.matched_key}' "
            f"(please confirm)"
        )

    @property
    def requires_confirmation(self) -> bool:
        """
        Returns True if the match confidence is low enough to warrant
        presenting a confirmation prompt to the user.
        Fuzzy matches and alias matches both require confirmation.
        """
        return self.was_fuzzy or self.matched_via_alias is not None

    def __repr__(self) -> str:
        if self.failed:
            return f"FuzzyMatchResult(failed: '{self.query}')"
        return f"FuzzyMatchResult({self.confidence_description})"


def match_set_key(
    query: str,
    known_keys: list[str],
    config_section: dict | None = None
) -> FuzzyMatchResult:
    """
    Attempts to match a query string against a list of known keys.

    Matching order:
        1. Exact match against config keys (case-insensitive)
        2. Fuzzy match against config keys
        3. Exact match against aliases (if config_section provided)
        4. Fuzzy match against aliases (if config_section provided)

    If config_section is provided, alias resolution is handled
    internally and matched_key always returns the actual config key.
    matched_via_alias is populated if the match was made via an alias.

    Args:
        query:          The string to match (e.g. from LLM or directory name)
        known_keys:     List of config keys and aliases to match against
        config_section: Optional config dict section for alias resolution.
                        If provided, matched_key is always a config key.
                        If None, matched_key may be an alias.

    Returns:
        FuzzyMatchResult describing the outcome.
    """
    if not query or not known_keys:
        return FuzzyMatchResult(
            query=query,
            matched_key=None,
            was_exact=False
        )

    query_normalised = query.strip().lower()
    keys_normalised = {key.lower(): key for key in known_keys}

    # 1. exact match against keys
    if query_normalised in keys_normalised:
        original_key = keys_normalised[query_normalised]
        resolved = _resolve_to_config_key(original_key, config_section)
        alias = original_key if resolved != original_key else None
        return FuzzyMatchResult(
            query=query,
            matched_key=resolved,
            was_exact=True,
            matched_via_alias=alias
        )

    # 2. fuzzy match against keys
    close_matches = get_close_matches(
        query_normalised,
        keys_normalised.keys(),
        n=1,
        cutoff=FUZZY_MATCH_CUTOFF
    )
    if close_matches:
        original_key = keys_normalised[close_matches[0]]
        resolved = _resolve_to_config_key(original_key, config_section)
        alias = original_key if resolved != original_key else None
        return FuzzyMatchResult(
            query=query,
            matched_key=resolved,
            was_exact=False,
            matched_via_alias=alias
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
    Performs tiered matching — manufacturer first, then set within
    that manufacturer. Returns SupportedResult if both match, or
    DiscoveryResult with as much partial information as was gathered.

    matched_key on both results is always the actual config key.
    requires_confirmation on each FuzzyMatchResult indicates whether
    the user should be prompted to confirm the match.

    Args:
        manufacturer_query: Manufacturer string to match
        set_query:          Set name string to match
        config:             The full sets_config dictionary
    """
    # step 1 - match manufacturer
    manufacturer_keys = get_manufacturer_keys(config)
    manufacturer_result = match_set_key(
        query=manufacturer_query,
        known_keys=manufacturer_keys,
        config_section=config["manufacturers"]
    )

    if manufacturer_result.failed:
        return DiscoveryResult(
            reason="unknown_manufacturer",
            known_manufacturer=None
        )

    manufacturer_key = manufacturer_result.matched_key

    # step 2 - match set within confirmed manufacturer
    set_keys = get_set_keys(config, manufacturer_key)
    set_result = match_set_key(
        query=set_query,
        known_keys=set_keys,
        config_section=config["manufacturers"][manufacturer_key]["sets"]
    )

    if set_result.failed:
        return DiscoveryResult(
            reason="unknown_set",
            known_manufacturer=manufacturer_key
        )

    return SupportedResult(
        manufacturer=manufacturer_key,
        set_key=set_result.matched_key
    )


def _resolve_to_config_key(
    matched_value: str,
    config_section: dict | None
) -> str:
    """
    Resolves a matched value (which may be an alias) back to its
    parent config key.

    If config_section is None, returns matched_value unchanged since
    there is no section to search for aliases.

    Args:
        matched_value:  The string that was matched (key or alias)
        config_section: The config dict section to search within,
                        or None if alias resolution is not needed
    """
    if config_section is None:
        return matched_value

    matched_lower = matched_value.lower()

    for key, data in config_section.items():
        if key.lower() == matched_lower:
            return key
        aliases = [a.lower() for a in data.get("aliases", [])]
        if matched_lower in aliases:
            return key

    # fallback - return as-is if no parent found
    return matched_value