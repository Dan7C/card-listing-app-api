# Groq provider implementation for the LLM client

import os
import json
import logging
from groq import AsyncGroq
from groq import RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from app.core.llm.client import BaseLLMProvider, LLMProviderError
from app.core.pipeline.results import ClassificationResult, ExtractionResult

load_dotenv()

logger = logging.getLogger(__name__)

# Model selected for vision capability on Groq free tier
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class GroqProvider(BaseLLMProvider):
    """
    Groq implementation of BaseLLMProvider.
    Uses the AsyncGroq client to make vision API calls.

    Catches all Groq SDK exceptions and re-raises as LLMProviderError
    with HTTP status codes so LLMClient can apply the correct retry strategy.
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    async def classify(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str
    ) -> ClassificationResult:
        """
        Sends a classification call to Groq.
        Expects the prompt to instruct the model to respond in JSON.

        Args:
            front_image_b64: Base64 encoded front image
            mime_type: MIME type of the image (e.g. "image/png")
            prompt: The classification prompt string

        Returns:
            ClassificationResult parsed from the model response

        Raises:
            LLMProviderError with appropriate status code on failure
        """
        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            _image_block(front_image_b64, mime_type),
                            _text_block(prompt)
                        ]
                    }
                ]
            )
            raw = response.choices[0].message.content
            logger.debug(f"Groq classify raw response: {raw}")
            return _parse_classification(raw)

        except RateLimitError as e:
            raise LLMProviderError(
                str(e),
                status_code=429,
                retry_after=_extract_retry_after(e)
            )
        except APITimeoutError as e:
            raise LLMProviderError(str(e), status_code=408)
        except APIStatusError as e:
            raise LLMProviderError(str(e), status_code=e.status_code)
        except APIConnectionError as e:
            raise LLMProviderError(str(e), status_code=None)

    async def extract(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str,
        back_image_b64: str | None = None
    ) -> ExtractionResult:
        """
        Sends an extraction call to Groq.
        Includes back image if provided.
        Expects the prompt to instruct the model to respond in JSON.

        Args:
            front_image_b64: Base64 encoded front image
            mime_type: MIME type of the image (e.g. "image/png")
            prompt: The extraction prompt string
            back_image_b64: Optional base64 encoded back image

        Returns:
            ExtractionResult parsed from the model response

        Raises:
            LLMProviderError with appropriate status code on failure
        """
        content = [_image_block(front_image_b64, mime_type)]

        if back_image_b64 is not None:
            content.append(_image_block(back_image_b64, mime_type))

        content.append(_text_block(prompt))

        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            raw = response.choices[0].message.content
            logger.debug(f"Groq extract raw response: {raw}")
            return _parse_extraction(raw)

        except RateLimitError as e:
            raise LLMProviderError(
                str(e),
                status_code=429,
                retry_after=_extract_retry_after(e)
            )
        except APITimeoutError as e:
            raise LLMProviderError(str(e), status_code=408)
        except APIStatusError as e:
            raise LLMProviderError(str(e), status_code=e.status_code)
        except APIConnectionError as e:
            raise LLMProviderError(str(e), status_code=None)


def _image_block(image_b64: str, mime_type: str) -> dict:
    """
    Builds a Groq image content block from a base64 encoded image.
    Uses the data URI format required by the Groq vision API.
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_b64}"
        }
    }


def _text_block(text: str) -> dict:
    """
    Builds a Groq text content block.
    """
    return {
        "type": "text",
        "text": text
    }


def _extract_retry_after(error: RateLimitError) -> int | None:
    """
    Attempts to extract the Retry-After header value from a
    RateLimitError response.
    Returns the value in seconds as an integer, or None if not present.
    """
    try:
        value = error.response.headers.get("retry-after")
        return int(value) if value is not None else None
    except Exception:
        return None


def _parse_classification(raw: str) -> ClassificationResult:
    """
    Parses a raw JSON string from the Groq classification response
    into a ClassificationResult.

    Expected JSON fields:
        manufacturer: str | null
        set: str | null
        face: "front" | "back" | null
        subject: "player" | "team_badge" | "trophy" | "squad" |
                 "stadium" | "manager" | "other" | null
        contains_multiple_cards: bool

    If parsing fails, returns a ClassificationResult with all fields
    set to None so the pipeline can route to Discovery mode rather
    than crashing.
    """
    try:
        data = _parse_json(raw)
        return ClassificationResult(
            manufacturer=data.get("manufacturer"),
            set_name=data.get("set"),
            face=data.get("face"),
            subject=data.get("subject"),
            contains_multiple_cards=data.get("contains_multiple_cards", False),
            raw_response=raw
        )
    except Exception as e:
        logger.warning(f"Failed to parse classification response: {e}")
        logger.debug(f"Raw response was: {raw}")
        return ClassificationResult(
            manufacturer=None,
            set_name=None,
            face=None,
            subject=None,
            contains_multiple_cards=False,
            raw_response=raw
        )


def _parse_extraction(raw: str) -> ExtractionResult:
    """
    Parses a raw JSON string from the Groq extraction response
    into an ExtractionResult.

    Expected JSON fields:
        player_name: str | null
        team_name: str | null
        card_number: str | null
        variant: str | null
        condition: str | null
        condition_notes: str | null
        processing_mode: "supported" | "discovery"
        confidence: "high" | "unverified"

    If parsing fails, returns an ExtractionResult with all fields
    set to None and confidence set to "unverified".
    """
    try:
        data = _parse_json(raw)
        return ExtractionResult(
            player_name=data.get("player_name"),
            team_name=data.get("team_name"),
            card_number=data.get("card_number"),
            variant=data.get("variant"),
            condition=data.get("condition"),
            condition_notes=data.get("condition_notes"),
            processing_mode=data.get("processing_mode", ""),
            raw_response=raw,
            confidence=data.get("confidence", "unverified")
        )
    except Exception as e:
        logger.warning(f"Failed to parse extraction response: {e}")
        logger.debug(f"Raw response was: {raw}")
        return ExtractionResult(
            player_name=None,
            team_name=None,
            card_number=None,
            variant=None,
            condition=None,
            condition_notes=None,
            processing_mode="",
            raw_response=raw,
            confidence="unverified"
        )


def _parse_json(raw: str) -> dict:
    """
    Strips markdown code fences from a string and parses it as JSON.
    LLMs sometimes wrap JSON responses in ```json ... ``` blocks
    despite being instructed not to — this handles that gracefully.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())