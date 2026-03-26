#!/usr/bin/env python3
# Validates all manufacturer and set config files against required
# field schemas. Run from the project root before committing:
#   python tools/validate_set_config.py

import json
import sys
from pathlib import Path

CONFIG_DIR = Path("config/manufacturers")

# required fields and their expected types for manufacturer.json
MANUFACTURER_REQUIRED_FIELDS = {
    "display_name":         str,
    "aliases":              list,
    "prompt_layer":         str,
    "card_name_location":   str,
    "card_name_format":     str,
    "card_number_location": str,
    "subject_prompt_map":   dict
}

# required fields and their expected types for set json files
SET_REQUIRED_FIELDS = {
    "display_name":      str,
    "aliases":           list,
    "year":              int,
    "image_mode":        str,
    "has_card_number":   bool,
    "known_variants":    list,
    "supported_subjects": list,
    "subject_prompt_map": dict,
    "directory_name_hint": str
}

# fields required only when has_card_number is true
CARD_NUMBER_FIELDS = {
    "card_number_format": (str, type(None)),
}

# valid values for constrained fields
VALID_IMAGE_MODES = {"front_back", "front_only"}
VALID_SUBJECTS = {
    "player", "team_badge", "trophy", "poster",
    "mascot", "manager", "squad", "stadium", "other"
}


def main():
    print("\n=== Set Config Validator ===\n")

    if not CONFIG_DIR.exists():
        print(f"✗ Config directory not found: {CONFIG_DIR}")
        print(
            "  Copy config.example/manufacturers to "
            "config/manufacturers and populate with your sets."
        )
        sys.exit(1)

    errors = []
    warnings = []
    manufacturer_count = 0
    set_count = 0

    manufacturer_dirs = [
        d for d in sorted(CONFIG_DIR.iterdir())
        if d.is_dir()
    ]

    if not manufacturer_dirs:
        print(f"✗ No manufacturer directories found in {CONFIG_DIR}")
        sys.exit(1)

    for manufacturer_dir in manufacturer_dirs:
        manufacturer_key = manufacturer_dir.name
        manufacturer_file = manufacturer_dir / "manufacturer.json"

        # validate manufacturer.json
        if not manufacturer_file.exists():
            errors.append(
                f"{manufacturer_key}: missing manufacturer.json"
            )
            continue

        manufacturer_config, load_errors = _load_json(manufacturer_file)
        if load_errors:
            errors.extend([
                f"{manufacturer_key}/manufacturer.json: {e}"
                for e in load_errors
            ])
            continue

        field_errors = _validate_fields(
            manufacturer_config,
            MANUFACTURER_REQUIRED_FIELDS,
            f"{manufacturer_key}/manufacturer.json"
        )
        errors.extend(field_errors)
        manufacturer_count += 1

        # validate set files
        set_files = [
            f for f in sorted(manufacturer_dir.iterdir())
            if f.is_file()
            and f.suffix == ".json"
            and f.name != "manufacturer.json"
        ]

        if not set_files:
            warnings.append(
                f"{manufacturer_key}: no set files found"
            )

        for set_file in set_files:
            set_key = set_file.stem
            context = f"{manufacturer_key}/{set_key}.json"

            set_config, load_errors = _load_json(set_file)
            if load_errors:
                errors.extend([
                    f"{context}: {e}" for e in load_errors
                ])
                continue

            field_errors = _validate_fields(
                set_config,
                SET_REQUIRED_FIELDS,
                context
            )
            errors.extend(field_errors)

            # validate card number fields when has_card_number is true
            if set_config.get("has_card_number"):
                for field, expected_type in CARD_NUMBER_FIELDS.items():
                    if field not in set_config:
                        errors.append(
                            f"{context}: missing field "
                            f"'{field}' (required when "
                            f"has_card_number is true)"
                        )

            # validate image_mode
            image_mode = set_config.get("image_mode")
            if image_mode and image_mode not in VALID_IMAGE_MODES:
                errors.append(
                    f"{context}: invalid image_mode '{image_mode}'. "
                    f"Must be one of: "
                    f"{', '.join(sorted(VALID_IMAGE_MODES))}"
                )

            # validate supported_subjects
            supported = set_config.get("supported_subjects", [])
            invalid_subjects = [
                s for s in supported
                if s not in VALID_SUBJECTS
            ]
            if invalid_subjects:
                errors.append(
                    f"{context}: invalid subjects: "
                    f"{', '.join(invalid_subjects)}. "
                    f"Valid subjects: "
                    f"{', '.join(sorted(VALID_SUBJECTS))}"
                )

            # validate subject_prompt_map keys
            prompt_map = set_config.get("subject_prompt_map", {})
            for subject in prompt_map:
                if subject not in VALID_SUBJECTS:
                    warnings.append(
                        f"{context}: subject_prompt_map contains "
                        f"unknown subject '{subject}'"
                    )

            # warn if supported_subjects has entries not in
            # either subject_prompt_map
            manufacturer_map = manufacturer_config.get(
                "subject_prompt_map", {}
            )
            for subject in supported:
                if (
                    subject not in prompt_map
                    and subject not in manufacturer_map
                ):
                    warnings.append(
                        f"{context}: subject '{subject}' has no "
                        f"prompt map entry at set or manufacturer "
                        f"level — will fall back to global subject "
                        f"prompt or extraction_generic.txt"
                    )

            set_count += 1

    # print results
    print(
        f"Validated {manufacturer_count} manufacturers, "
        f"{set_count} sets\n"
    )

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()

    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  ✗ {error}")
        print()
        print("Fix errors before committing.")
        sys.exit(1)
    else:
        print("✓ All config files are valid.")
        sys.exit(0)


def _load_json(path: Path) -> tuple[dict, list[str]]:
    """
    Loads a JSON file. Returns (config, errors).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), []
    except json.JSONDecodeError as e:
        return {}, [f"invalid JSON: {e}"]
    except OSError as e:
        return {}, [f"could not read file: {e}"]


def _validate_fields(
    config: dict,
    required_fields: dict,
    context: str
) -> list[str]:
    """
    Validates that all required fields are present and
    have the correct type. Returns list of error strings.
    """
    errors = []

    for field, expected_type in required_fields.items():
        if field not in config:
            errors.append(
                f"{context}: missing required field '{field}'"
            )
            continue

        value = config[field]

        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                type_names = " or ".join(
                    t.__name__ for t in expected_type
                )
                errors.append(
                    f"{context}: field '{field}' must be "
                    f"{type_names}, got {type(value).__name__}"
                )
        else:
            if not isinstance(value, expected_type):
                errors.append(
                    f"{context}: field '{field}' must be "
                    f"{expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    return errors


if __name__ == "__main__":
    main()
