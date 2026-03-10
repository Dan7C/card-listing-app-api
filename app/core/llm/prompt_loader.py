# Loads prompt templates from the prompts/ directory at runtime

from pathlib import Path

# Path to the prompts directory relative to the project root
PROMPTS_DIR = Path("prompts")


class PromptLoadError(Exception):
    """Raised when a prompt template cannot be loaded."""
    pass


def load_prompt(name: str) -> str:
    """
    Loads a prompt template by name from the prompts/ directory.

    Args:
        name: The prompt filename without extension (e.g. "classification")

    Returns:
        The prompt template as a string.

    Raises PromptLoadError if the file does not exist or cannot be read.
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"

    if not prompt_path.exists():
        raise PromptLoadError(
            f"Prompt template '{name}' not found at {prompt_path}. "
            f"Copy prompts.example/ to prompts/ and fill in your prompts."
        )

    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise PromptLoadError(
            f"Failed to read prompt template '{name}': {e}"
        )


def load_prompt_with_variables(name: str, variables: dict) -> str:
    """
    Loads a prompt template and substitutes placeholder variables.

    Placeholders in prompt files use {variable_name} syntax, for example:
        "This card is from {set_display_name} by {manufacturer_display_name}"

    Args:
        name: The prompt filename without extension
        variables: Dictionary of variable names and their values to substitute

    Returns:
        The prompt template with all variables substituted.

    Raises PromptLoadError if the file cannot be loaded or a variable is missing.
    """
    template = load_prompt(name)

    try:
        return template.format(**variables)
    except KeyError as e:
        raise PromptLoadError(
            f"Prompt template '{name}' contains placeholder {e} "
            f"but no value was provided for it. "
            f"Provided variables: {list(variables.keys())}"
        )


def list_available_prompts() -> list[str]:
    """
    Returns a list of available prompt template names in the prompts/ directory.
    Useful for debugging and verifying prompt setup.
    """
    if not PROMPTS_DIR.exists():
        raise PromptLoadError(
            f"Prompts directory not found at '{PROMPTS_DIR}'. "
            f"Copy prompts.example/ to prompts/ and fill in your prompts."
        )

    return [
        p.stem for p in PROMPTS_DIR.glob("*.txt")
    ]