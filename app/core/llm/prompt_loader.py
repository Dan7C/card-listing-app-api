# Layered prompt loader for extraction and classification prompts.
#
# Prompt directory structure:
#   prompts/
#       classification/
#           classification.txt
#       extraction/
#           base/
#               extraction_base.txt
#               extraction_rules.txt
#           condition/
#               condition_instructions.txt
#           manufacturers/
#               {manufacturer}/
#                   extraction_{manufacturer}.txt
#           subjects/
#               extraction_{subject}.txt
#               extraction_generic.txt
#           extraction_discovery.txt
#
# Assembly order for supported extraction:
#   1. Load base template
#   2. Build set_context_instructions from config
#   3. Load manufacturer_instructions from manufacturers/ dir
#   4. Resolve and load subject_instructions from subjects/ dir
#   5. Build card_name_instructions from config hints
#   6. Build card_number_instructions from config hints
#   7. Build variant_instructions from config hints
#   8. Load condition_instructions from condition/ dir
#   9. Load extraction_rules from base/ dir
#   10. Build field_definitions from config + subject + mode
#   11. Substitute all sections into base template

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("prompts")

CLASSIFICATION_DIR = PROMPTS_DIR / "classification"
EXTRACTION_DIR = PROMPTS_DIR / "extraction"
BASE_DIR = EXTRACTION_DIR / "base"
CONDITION_DIR = EXTRACTION_DIR / "condition"
MANUFACTURERS_DIR = EXTRACTION_DIR / "manufacturers"
SUBJECTS_DIR = EXTRACTION_DIR / "subjects"


class PromptLoadError(Exception):
    """
    Raised when a prompt file cannot be found or loaded,
    or when required template variables are missing.
    """
    pass


def load_classification_prompt() -> str:
    """
    Loads the classification prompt.

    Returns:
        Classification prompt string.

    Raises:
        PromptLoadError if the prompt file cannot be found or loaded.
    """
    return _load_file(CLASSIFICATION_DIR / "classification.txt")


def load_extraction_prompt(
    manufacturer_config: dict,
    set_config: dict,
    subject: str,
    processing_mode: str
) -> str:
    """
    Assembles the full extraction prompt from layered components.

    For supported mode, assembles from:
        base template + manufacturer layer + subject layer +
        config-derived sections + field definitions

    For discovery mode, loads the discovery prompt and injects
    available context.

    Args:
        manufacturer_config: Manufacturer config dict, or empty dict
                             if manufacturer is unknown
        set_config:          Set config dict, or empty dict if set
                             is unknown
        subject:             Classified subject for this card
        processing_mode:     "supported" or "discovery"

    Returns:
        Assembled prompt string ready to send to the LLM.

    Raises:
        PromptLoadError if required prompt files cannot be found.
    """
    if processing_mode == "discovery":
        return _assemble_discovery_prompt(manufacturer_config)

    return _assemble_supported_prompt(
        manufacturer_config=manufacturer_config,
        set_config=set_config,
        subject=subject
    )


def list_available_prompts() -> dict:
    """
    Returns a dictionary of all available prompt files organised
    by directory. Used for startup validation and health checks.

    Returns:
        Dict with keys: classification, manufacturers, subjects, base
    """
    result = {
        "classification": [],
        "manufacturers": {},
        "subjects": [],
        "base": []
    }

    if CLASSIFICATION_DIR.exists():
        result["classification"] = [
            f.stem for f in CLASSIFICATION_DIR.glob("*.txt")
        ]

    if BASE_DIR.exists():
        result["base"] = [
            f.stem for f in BASE_DIR.glob("*.txt")
        ]

    if SUBJECTS_DIR.exists():
        result["subjects"] = [
            f.stem for f in SUBJECTS_DIR.glob("*.txt")
        ]

    if MANUFACTURERS_DIR.exists():
        for manufacturer_dir in MANUFACTURERS_DIR.iterdir():
            if manufacturer_dir.is_dir():
                result["manufacturers"][manufacturer_dir.name] = [
                    f.stem for f in manufacturer_dir.glob("*.txt")
                ]

    return result


def _assemble_supported_prompt(
    manufacturer_config: dict,
    set_config: dict,
    subject: str
) -> str:
    """
    Assembles the full supported extraction prompt from layers.
    """
    base_template = _load_file(BASE_DIR / "extraction_base.txt")
    extraction_rules = _load_file(BASE_DIR / "extraction_rules.txt")
    condition_instructions = _load_file(
        CONDITION_DIR / "condition_instructions.txt"
    )

    sections = {
        "set_context_instructions": _build_set_context(
            manufacturer_config, set_config
        ),
        "manufacturer_instructions": _load_manufacturer_instructions(
            manufacturer_config, set_config
        ),
        "subject_instructions": _load_subject_instructions(
            subject, manufacturer_config, set_config
        ),
        "card_name_instructions": _build_card_name_instructions(
            manufacturer_config, set_config
        ),
        "card_number_instructions": _build_card_number_instructions(
            manufacturer_config, set_config
        ),
        "variant_instructions": _build_variant_instructions(set_config),
        "condition_instructions": condition_instructions,
        "extraction_rules": extraction_rules,
        "field_definitions": _build_field_definitions(
            manufacturer_config, set_config, subject, "supported"
        )
    }

    try:
        return base_template.format(**sections)
    except KeyError as e:
        raise PromptLoadError(
            f"Base template contains unknown placeholder: {e}. "
            f"Available sections: {list(sections.keys())}"
        )


def _assemble_discovery_prompt(
    manufacturer_config: dict
) -> str:
    """
    Assembles the discovery mode extraction prompt.
    Injects manufacturer context if available.
    """
    template = _load_file(EXTRACTION_DIR / "extraction_discovery.txt")
    extraction_rules = _load_file(BASE_DIR / "extraction_rules.txt")

    manufacturer_context = _build_manufacturer_context_instructions(
        manufacturer_config
    )

    try:
        return template.format(
            manufacturer_context_instructions=manufacturer_context,
            extraction_rules=extraction_rules
        )
    except KeyError as e:
        raise PromptLoadError(
            f"Discovery prompt contains unknown placeholder: {e}"
        )


def _build_set_context(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Builds the set context section from config values.
    """
    manufacturer_name = manufacturer_config.get("display_name", "Unknown")
    set_name = set_config.get("display_name", "Unknown")
    year = set_config.get("year", "Unknown")

    return (
        f"This card is from the {manufacturer_name} "
        f"{set_name} set ({year})."
    )


def _load_manufacturer_instructions(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Loads manufacturer instructions and injects config hints.
    Falls back to empty string if no manufacturer prompt exists.
    """
    prompt_layer = manufacturer_config.get("prompt_layer")
    if not prompt_layer:
        logger.debug("No manufacturer prompt layer specified.")
        return ""

    manufacturer_file = (
        MANUFACTURERS_DIR / prompt_layer /
        f"extraction_{prompt_layer}.txt"
    )

    if not manufacturer_file.exists():
        logger.warning(
            f"Manufacturer prompt file not found: {manufacturer_file}. "
            f"Skipping manufacturer layer."
        )
        return ""

    template = _load_file(manufacturer_file)

    # resolve card name and number location from config hierarchy
    card_name_location = (
        set_config.get("card_name_location_override")
        or manufacturer_config.get("card_name_location")
        or "unknown location"
    )
    card_name_format = (
        set_config.get("card_name_format_override")
        or manufacturer_config.get("card_name_format")
        or "unknown format"
    )
    card_number_location = (
        set_config.get("card_number_location_override")
        or manufacturer_config.get("card_number_location")
        or "unknown location"
    )

    try:
        return template.format(
            card_name_location=card_name_location,
            card_name_format=card_name_format,
            card_number_location=card_number_location
        )
    except KeyError as e:
        raise PromptLoadError(
            f"Manufacturer prompt {manufacturer_file} contains "
            f"unknown placeholder: {e}"
        )


def _load_subject_instructions(
    subject: str,
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Loads subject instructions following priority chain:
        1. Set-level subject_prompt_map entry
        2. Manufacturer-level subject_prompt_map entry
        3. Global subject file
        4. Generic fallback

    Injects card name instructions into subject template
    where {card_name_specific_instructions} placeholder exists.
    """
    resolved_name = _resolve_subject_prompt(
        subject, manufacturer_config, set_config
    )

    subject_file = SUBJECTS_DIR / f"extraction_{resolved_name}.txt"

    if not subject_file.exists():
        logger.warning(
            f"Subject prompt file not found: {subject_file}. "
            f"Falling back to extraction_generic.txt."
        )
        subject_file = SUBJECTS_DIR / "extraction_generic.txt"

    if not subject_file.exists():
        raise PromptLoadError(
            "extraction_generic.txt not found in subjects directory. "
            "This file is required as a fallback."
        )

    template = _load_file(subject_file)

    # inject card name specific instructions if placeholder present
    if "{card_name_specific_instructions}" in template:
        card_name_instructions = _build_card_name_specific_instructions(
            manufacturer_config, set_config
        )
        try:
            template = template.format(
                card_name_specific_instructions=card_name_instructions
            )
        except KeyError as e:
            raise PromptLoadError(
                f"Subject prompt {subject_file} contains "
                f"unknown placeholder: {e}"
            )

    return template


def _resolve_subject_prompt(
    subject: str,
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Resolves subject to prompt filename following priority chain:
    set map → manufacturer map → global file → generic fallback.

    Returns the filename stem (without extension) of the prompt
    file to load.
    """
    # 1. set-level override
    set_map = set_config.get("subject_prompt_map", {})
    if subject in set_map:
        logger.debug(
            f"Subject '{subject}' resolved from set map: "
            f"{set_map[subject]}"
        )
        return set_map[subject]

    # 2. manufacturer-level default
    manufacturer_map = manufacturer_config.get("subject_prompt_map", {})
    if subject in manufacturer_map:
        logger.debug(
            f"Subject '{subject}' resolved from manufacturer map: "
            f"{manufacturer_map[subject]}"
        )
        return manufacturer_map[subject]

    # 3. global subject file
    global_path = SUBJECTS_DIR / f"extraction_{subject}.txt"
    if global_path.exists():
        logger.debug(
            f"Subject '{subject}' resolved from global subjects dir"
        )
        return subject

    # 4. generic fallback
    logger.warning(
        f"No prompt found for subject '{subject}'. "
        f"Falling back to extraction_generic.txt."
    )
    return "generic"


def _build_card_name_instructions(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Builds the card name instructions section from config hints.
    Used in the base template {card_name_instructions} section.
    """
    location = (
        set_config.get("card_name_location_override")
        or manufacturer_config.get("card_name_location")
    )
    name_format = (
        set_config.get("card_name_format_override")
        or manufacturer_config.get("card_name_format")
    )

    if not location and not name_format:
        return (
            "The card name or subject identifier may appear on the "
            "front of the card, typically near the top or bottom. "
            "Return null if you cannot confidently identify it."
        )

    parts = []

    if location:
        parts.append(f"The card name appears at {location}.")

    if name_format:
        parts.append(f"It is formatted as {name_format}.")

    parts.append(
        "Extract the name exactly as printed. "
        "Return null if you cannot confidently identify it."
    )

    return " ".join(parts)


def _build_card_name_specific_instructions(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Builds card name specific instructions for injection into
    subject prompts via {card_name_specific_instructions}.
    Handles the case where front and back have different formats.
    """
    location = (
        set_config.get("card_name_location_override")
        or manufacturer_config.get("card_name_location")
    )
    name_format = (
        set_config.get("card_name_format_override")
        or manufacturer_config.get("card_name_format")
    )

    if not location and not name_format:
        return (
            "Look for the name on the front of the card, "
            "typically near the top or bottom."
        )

    # check if format differs between front and back
    if name_format and " on front" in name_format.lower():
        # format specifies different front and back formats
        parts = name_format.split(",")
        front_format = parts[0].strip() if parts else name_format
        back_format = parts[1].strip() if len(parts) > 1 else None

        instructions = [
            f"The name appears at {location}." if location else ""
        ]

        if back_format:
            instructions.append(
                f"When a back image is available, prefer the back "
                f"format: {back_format}."
            )
            instructions.append(
                f"When front only, use the front format: {front_format}."
            )
        else:
            instructions.append(f"Name format: {front_format}.")

        return " ".join(i for i in instructions if i)

    parts = []
    if location:
        parts.append(f"The name appears at {location}.")
    if name_format:
        parts.append(f"Name format: {name_format}.")

    return " ".join(parts)


def _build_card_number_instructions(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Builds the card number instructions section from config hints.
    Returns empty string when has_card_number is false — the section
    is omitted from the assembled prompt entirely.
    """
    has_card_number = set_config.get("has_card_number", True)

    if not has_card_number:
        return ""

    location = (
        set_config.get("card_number_location_override")
        or manufacturer_config.get("card_number_location")
    )
    number_format = set_config.get("card_number_format")
    number_range = set_config.get("card_number_range")

    parts = []

    if location:
        parts.append(
            f"The card number is printed at {location}."
        )
    else:
        parts.append(
            "The card number may appear in a corner or along "
            "an edge of the card."
        )

    if number_format:
        parts.append(
            f"The card number format is {number_format} "
            f"(a plain number with no prefix or suffix)."
        )

    if number_range:
        parts.append(
            f"Valid card numbers for this set are in the range "
            f"1 to {number_range}. "
            f"If the number you identify falls outside this range, "
            f"return null."
        )

    parts.append(
        "Do not use squad numbers, shirt numbers, years, "
        "or statistics as the card number. "
        "Return null if you cannot confidently identify "
        "the card number."
    )

    return " ".join(parts)


def _build_variant_instructions(set_config: dict) -> str:
    """
    Builds the variant instructions section from config hints.
    Returns empty string when only one variant exists.
    """
    known_variants = set_config.get("known_variants", ["base"])

    if len(known_variants) <= 1:
        return ""

    variants_list = ", ".join(known_variants)
    return (
        f"This set has the following known variants: {variants_list}. "
        f"Identify the variant from visual cues such as surface finish, "
        f"border colour, or foil treatment. "
        f"Return null if you cannot confidently identify the variant."
    )


def _build_field_definitions(
    manufacturer_config: dict,
    set_config: dict,
    subject: str,
    processing_mode: str
) -> str:
    """
    Builds the field definitions section with hardcoded values
    where the context makes them certain.

    Hardcoded values:
        subject:         always the classified subject
        processing_mode: always "supported" or "discovery"
        confidence:      calculated from injection depth
        card_number:     null when has_card_number is false
        variant:         hardcoded when only one variant exists
    """
    has_card_number = set_config.get("has_card_number", True)
    known_variants = set_config.get("known_variants", ["base"])
    confidence = _calculate_confidence(manufacturer_config, set_config)

    card_number_definition = (
        "null" if not has_card_number
        else "string or null"
    )

    variant_definition = (
        f'"{known_variants[0]}"' if len(known_variants) == 1
        else "string or null"
    )

    return f"""Return a JSON object with exactly these fields:
{{
    "card_name": string or null,
    "team_name": string or null,
    "card_number": {card_number_definition},
    "variant": {variant_definition},
    "subject": "{subject}",
    "condition_observations": list of strings or null,
    "condition_recommendation": string or null,
    "processing_mode": "{processing_mode}",
    "confidence": "{confidence}"
}}

Include all fields exactly as listed above.
Do not add additional fields.
Do not omit any fields.
Fields shown as hardcoded values must be returned exactly as shown."""


def _calculate_confidence(
    manufacturer_config: dict,
    set_config: dict
) -> str:
    """
    Calculates extraction confidence based on prompt injection depth.
    Confidence reflects how much context was available, not LLM
    self-assessment.

        high:       manufacturer prompt + subject map + card number
                    hints + set context
        medium:     manufacturer prompt + set context
        medium_low: set context only
        low:        discovery mode (handled separately)
    """
    has_manufacturer_prompt = bool(
        manufacturer_config.get("prompt_layer")
    )
    has_subject_map = bool(
        set_config.get("subject_prompt_map") or
        manufacturer_config.get("subject_prompt_map")
    )
    has_card_number_hints = (
        set_config.get("has_card_number") is not None and
        bool(
            set_config.get("card_number_format") or
            manufacturer_config.get("card_number_location")
        )
    )
    has_set_context = bool(set_config.get("display_name"))

    if (
        has_manufacturer_prompt and
        has_subject_map and
        has_card_number_hints and
        has_set_context
    ):
        return "high"

    if has_manufacturer_prompt and has_set_context:
        return "medium"

    if has_set_context:
        return "medium_low"

    return "low"


def _build_manufacturer_context_instructions(
    manufacturer_config: dict
) -> str:
    """
    Builds manufacturer context for discovery mode prompts.
    Used when manufacturer is known but set is not.
    Returns empty string if no manufacturer context is available.
    """
    if not manufacturer_config:
        return ""

    manufacturer_name = manufacturer_config.get("display_name")
    if not manufacturer_name:
        return ""

    parts = [
        f"The manufacturer has been identified as {manufacturer_name}."
    ]

    card_name_location = manufacturer_config.get("card_name_location")
    if card_name_location:
        parts.append(
            f"Card name typically appears at {card_name_location}."
        )

    card_number_location = manufacturer_config.get("card_number_location")
    if card_number_location:
        parts.append(
            f"Card number when present typically appears at "
            f"{card_number_location}."
        )

    return " ".join(parts)


def _load_file(path: Path) -> str:
    """
    Loads a text file and returns its contents.
    Raises PromptLoadError if the file cannot be found or read.
    """
    if not path.exists():
        raise PromptLoadError(
            f"Prompt file not found: {path}. "
            f"Check that all required prompt files exist in "
            f"the prompts/ directory."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise PromptLoadError(
            f"Could not read prompt file {path}: {e}"
        )