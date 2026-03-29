#!/usr/bin/env python3
# Interactive CLI script for creating manufacturer and set config files.
# Run from the project root:
#   python tools/create_set_config.py

import json
import sys
from pathlib import Path

CONFIG_DIR = Path("config/manufacturers")


def main():
    print("\n=== Card Config Creator ===\n")
    print("What would you like to create?")
    print("  1. New manufacturer")
    print("  2. New set")
    print("  3. Exit")

    choice = _prompt("Choice", options=["1", "2", "3"])

    if choice == "1":
        create_manufacturer()
    elif choice == "2":
        create_set()
    else:
        print("Exiting.")
        sys.exit(0)


def create_manufacturer():
    print("\n=== New Manufacturer ===\n")

    key = _prompt(
        "Manufacturer key (lowercase, underscores, e.g. topps)",
        validator=_validate_key
    )

    # check if already exists
    manufacturer_dir = CONFIG_DIR / key
    if (manufacturer_dir / "manufacturer.json").exists():
        confirm = _prompt(
            f"manufacturer.json already exists for '{key}'. Overwrite?",
            options=["y", "n"]
        )
        if confirm == "n":
            print("Cancelled.")
            return

    display_name = _prompt("Display name (e.g. Topps)")

    aliases_input = _prompt(
        "Aliases (comma-separated, e.g. topps,topps uk)",
    )
    aliases = [a.strip().lower() for a in aliases_input.split(",") if a.strip()]

    prompt_layer = _prompt(
        "Prompt layer filename (e.g. topps — refers to "
        "prompts/extraction/manufacturers/topps/extraction_topps.txt)",
        default=key
    )

    card_name_location = _prompt(
        "Card name location (e.g. bottom centre of front)",
        default="unknown"
    )

    card_name_format = _prompt(
        "Card name format (e.g. FIRSTNAME LASTNAME / LASTNAME / uppercase)",
        default="unknown"
    )

    card_number_location = _prompt(
        "Card number location (e.g. bottom right corner of back)",
        default="unknown"
    )

    print("\nSubject prompt map — enter subject types common across "
          "this manufacturer's sets.")
    print("Leave blank when done.")
    subject_prompt_map = _collect_subject_prompt_map()

    config = {
        "display_name": display_name,
        "aliases": aliases,
        "prompt_layer": prompt_layer,
        "card_name_location": card_name_location,
        "card_name_format": card_name_format,
        "card_number_location": card_number_location,
        "subject_prompt_map": subject_prompt_map
    }

    # write to config
    _write_config(
        CONFIG_DIR / key / "manufacturer.json",
        config
    )

    print(f"\n✓ Created manufacturer config for '{key}'")
    print(f"  config/manufacturers/{key}/manufacturer.json")
    print("\nNext: add sets using option 2.")


def create_set():
    print("\n=== New Set ===\n")

    # select manufacturer
    manufacturers = _get_existing_manufacturers()
    if not manufacturers:
        print(
            "No manufacturers found. "
            "Create a manufacturer first using option 1."
        )
        return

    print("Existing manufacturers:")
    for i, m in enumerate(manufacturers, 1):
        print(f"  {i}. {m}")

    manufacturer_key = _prompt(
        "Manufacturer key",
        options=manufacturers
    )

    set_key = _prompt(
        "Set key (lowercase, underscores, e.g. merlin_1998)",
        validator=_validate_key
    )

    # check if already exists
    set_path = CONFIG_DIR / manufacturer_key / f"{set_key}.json"
    if set_path.exists():
        confirm = _prompt(
            f"'{set_key}.json' already exists for '{manufacturer_key}'. "
            f"Overwrite?",
            options=["y", "n"]
        )
        if confirm == "n":
            print("Cancelled.")
            return

    display_name = _prompt("Display name (e.g. Merlin 1998)")

    aliases_input = _prompt(
        "Aliases (comma-separated, e.g. merlin 1998,merlin '98,merlin98)"
    )
    aliases = [a.strip().lower() for a in aliases_input.split(",") if a.strip()]

    year = _prompt("Year (e.g. 1998)", validator=_validate_year)

    image_mode = _prompt(
        "Image mode",
        options=["front_back", "front_only"],
        default="front_back"
    )

    has_card_number = _prompt(
        "Does this set have card numbers?",
        options=["y", "n"]
    ) == "y"

    card_number_format = None
    card_number_range = None
    card_number_location_override = None

    with open(CONFIG_DIR / manufacturer_key / "manufacturer.json") as f:
        d = json.load(f)
        manufacturer_card_number_location = d["card_number_location"]
        manufacturer_card_name_location = d["card_name_location"]
        manufacturer_card_name_format = d["card_number_format"]

    if has_card_number:
        card_number_format = _prompt(
            "Card number format (e.g. NNN / #NNN / NNN/TTT)",
            default="unknown"
        )
        card_number_range = _prompt(
            "Card number range (e.g. 1-500, or leave blank if unknown)",
            default=None,
            optional=True
        )
        card_number_location_override = _prompt(
            "Card number location override "
            f"manufacturer default = {manufacturer_card_number_location}",
            default=None,
            optional=True
        )

    card_name_location_override = _prompt(
        "Card name location override "
        f"manufacturer default = {manufacturer_card_name_location}",
        default=None,
        optional=True
    )

    card_name_format_override = _prompt(
        "Card name format override "
        f"manufacturer default = {manufacturer_card_name_format}",
        default=None,
        optional=True
    )

    variants_input = _prompt(
        "Known variants (comma-separated, e.g. base,holographic,gold)",
        default="base"
    )
    known_variants = [
        v.strip().lower()
        for v in variants_input.split(",")
        if v.strip()
    ]

    subjects_input = _prompt(
        "Supported subjects (comma-separated, e.g. "
        "player,team_badge,poster,mascot,trophy)"
    )
    supported_subjects = [
        s.strip().lower()
        for s in subjects_input.split(",")
        if s.strip()
    ]

    print("\nSubject prompt map overrides — only needed for subjects "
          "that differ from the manufacturer default.")
    print("Leave blank when done.")
    subject_prompt_map = _collect_subject_prompt_map()

    directory_name_hint = _prompt(
        "Directory name hint (e.g. merlin_1998 — what you'd naturally "
        "name the directory)",
        default=set_key
    )

    config = {
        "display_name": display_name,
        "aliases": aliases,
        "year": int(year),
        "image_mode": image_mode,
        "has_card_number": has_card_number,
        "card_number_format": card_number_format,
        "card_number_range": card_number_range,
        "card_number_location_override": card_number_location_override,
        "card_name_location_override": card_name_location_override,
        "card_name_format_override": card_name_format_override,
        "known_variants": known_variants,
        "supported_subjects": supported_subjects,
        "subject_prompt_map": subject_prompt_map,
        "directory_name_hint": directory_name_hint
    }

    _write_config(
        CONFIG_DIR / manufacturer_key / f"{set_key}.json",
        config
    )

    print(f"\n✓ Created set config for '{manufacturer_key}/{set_key}'")
    print(
        f"  config/manufacturers/{manufacturer_key}/{set_key}.json"
    )


def _collect_subject_prompt_map() -> dict:
    """
    Interactively collects subject → prompt file mappings.
    """
    subject_map = {}
    while True:
        subject = input("  Subject type (or press Enter to finish): ").strip().lower()
        if not subject:
            break
        prompt_file = input(
            f"  Prompt file for '{subject}' "
            f"(e.g. player / player_sticker / poster): "
        ).strip().lower()
        if prompt_file:
            subject_map[subject] = prompt_file
    return subject_map


def _get_existing_manufacturers() -> list[str]:
    """
    Returns list of manufacturer keys that have a manufacturer.json.
    """
    if not CONFIG_DIR.exists():
        return []
    return [
        d.name for d in sorted(CONFIG_DIR.iterdir())
        if d.is_dir() and (d / "manufacturer.json").exists()
    ]


def _write_config(path: Path, config: dict) -> None:
    """
    Writes config dictionary to JSON file, creating directories
    as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _prompt(
    label: str,
    options: list[str] | None = None,
    default: str | None = None,
    optional: bool = False,
    validator=None
) -> str | None:
    """
    Prompts the user for input with optional validation.
    """
    hint_parts = []
    if options:
        hint_parts.append("/".join(options))
    if default is not None:
        hint_parts.append(f"default: {default}")
    elif optional:
        hint_parts.append("optional")

    hint = f" ({', '.join(hint_parts)})" if hint_parts else ""
    prompt_str = f"{label}{hint}: "

    while True:
        value = input(prompt_str).strip()

        if not value:
            if default is not None:
                return default
            if optional:
                return None
            print("  This field is required.")
            continue

        if options and value not in options:
            print(f"  Please enter one of: {', '.join(options)}")
            continue

        if validator:
            error = validator(value)
            if error:
                print(f"  {error}")
                continue

        return value


def _validate_key(value: str) -> str | None:
    """
    Validates that a key uses only lowercase letters, numbers
    and underscores.
    """
    if not all(c.islower() or c.isdigit() or c == "_" for c in value):
        return "Key must use only lowercase letters, numbers and underscores."
    return None


def _validate_year(value: str) -> str | None:
    """
    Validates that a year is a four digit number.
    """
    if not value.isdigit() or len(value) != 4:
        return "Year must be a four digit number."
    return None


if __name__ == "__main__":
    main()