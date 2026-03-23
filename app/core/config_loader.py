# Loads and assembles manufacturer and set config from split files.
# Config is split by manufacturer for maintainability:
#
#   config/
#       manufacturers/
#           panini/
#               manufacturer.json   ← manufacturer-level config
#               world_cup_2002.json ← set-level config
#           topps/
#               manufacturer.json
#               merlin_1998.json
#
# Assembles into the same nested dictionary structure the rest of
# the codebase expects, so no other modules need to change.

import json
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("config/manufacturers")


class ConfigLoadError(Exception):
    """
    Raised when config files cannot be loaded or assembled.
    """
    pass


@lru_cache(maxsize=1)
def load_sets_config() -> dict:
    """
    Loads and assembles the full sets config from split manufacturer
    and set files. Result is cached after first load — config is
    treated as immutable at runtime.

    Returns the assembled config in the standard nested structure:
    {
        "manufacturers": {
            "topps": {
                ...manufacturer config...,
                "sets": {
                    "merlin_1998": { ...set config... }
                }
            }
        }
    }

    Raises ConfigLoadError if any required files are missing or
    cannot be parsed.
    """
    if not CONFIG_DIR.exists():
        raise ConfigLoadError(
            f"Config directory not found: {CONFIG_DIR}. "
            f"Copy config.example/manufacturers to config/manufacturers "
            f"and populate with your sets."
        )

    assembled = {"manufacturers": {}}

    manufacturer_dirs = [
        d for d in CONFIG_DIR.iterdir()
        if d.is_dir()
    ]

    if not manufacturer_dirs:
        raise ConfigLoadError(
            f"No manufacturer directories found in {CONFIG_DIR}."
        )

    for manufacturer_dir in sorted(manufacturer_dirs):
        manufacturer_key = manufacturer_dir.name
        manufacturer_file = manufacturer_dir / "manufacturer.json"

        if not manufacturer_file.exists():
            logger.warning(
                f"No manufacturer.json found in {manufacturer_dir}. "
                f"Skipping."
            )
            continue

        try:
            manufacturer_config = _load_json(manufacturer_file)
        except ConfigLoadError as e:
            logger.error(
                f"Failed to load manufacturer config for "
                f"{manufacturer_key}: {e}. Skipping."
            )
            continue

        manufacturer_config["sets"] = {}

        # load set files — everything except manufacturer.json
        set_files = [
            f for f in manufacturer_dir.iterdir()
            if f.is_file()
            and f.suffix == ".json"
            and f.name != "manufacturer.json"
        ]

        for set_file in sorted(set_files):
            set_key = set_file.stem

            try:
                set_config = _load_json(set_file)
                manufacturer_config["sets"][set_key] = set_config
                logger.debug(
                    f"Loaded set config: {manufacturer_key}/{set_key}"
                )
            except ConfigLoadError as e:
                logger.error(
                    f"Failed to load set config "
                    f"{manufacturer_key}/{set_key}: {e}. Skipping."
                )
                continue

        assembled["manufacturers"][manufacturer_key] = manufacturer_config
        logger.debug(
            f"Loaded manufacturer config: {manufacturer_key} "
            f"({len(manufacturer_config['sets'])} sets)"
        )

    logger.info(
        f"Config loaded: {len(assembled['manufacturers'])} manufacturers, "
        f"{_count_sets(assembled)} sets total"
    )

    return assembled


def get_manufacturer_config(
    sets_config: dict,
    manufacturer_key: str
) -> dict:
    """
    Returns the manufacturer config for a given key.
    Raises ConfigLoadError if the manufacturer is not found.
    """
    manufacturers = sets_config.get("manufacturers", {})
    if manufacturer_key not in manufacturers:
        raise ConfigLoadError(
            f"Manufacturer '{manufacturer_key}' not found in config."
        )
    return manufacturers[manufacturer_key]


def get_set_config(
    sets_config: dict,
    manufacturer_key: str,
    set_key: str
) -> dict:
    """
    Returns the set config for a given manufacturer and set key.
    Raises ConfigLoadError if either is not found.
    """
    manufacturer_config = get_manufacturer_config(
        sets_config, manufacturer_key
    )
    sets = manufacturer_config.get("sets", {})
    if set_key not in sets:
        raise ConfigLoadError(
            f"Set '{set_key}' not found under "
            f"manufacturer '{manufacturer_key}'."
        )
    return sets[set_key]


def reload_config() -> dict:
    """
    Forces a reload of the config, bypassing the cache.
    Useful during testing when config files are modified between runs.
    """
    load_sets_config.cache_clear()
    return load_sets_config()


def _load_json(path: Path) -> dict:
    """
    Loads and parses a JSON file.
    Raises ConfigLoadError if the file cannot be read or parsed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigLoadError(
            f"Invalid JSON in {path}: {e}"
        )
    except OSError as e:
        raise ConfigLoadError(
            f"Could not read {path}: {e}"
        )


def _count_sets(assembled: dict) -> int:
    """
    Returns total number of sets across all manufacturers.
    """
    return sum(
        len(m.get("sets", {}))
        for m in assembled["manufacturers"].values()
    )

