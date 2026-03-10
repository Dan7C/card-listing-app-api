# Provider-agnostic async LLM API client

import os
import asyncio
import logging
from tenacity import (
    retry,
    wait_exponential,
    wait_fixed,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)
from dotenv import load_dotenv
from app.core.pipeline.results import ClassificationResult, ExtractionResult

load_dotenv()

logger = logging.getLogger(__name__)

# Fixed interval retry for timeouts and unexpected errors
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_INTERVAL_SECONDS = float(os.getenv("LLM_RETRY_INTERVAL_SECONDS", "10"))

# Exponential backoff constants for rate limits and server errors.
# Hardcoded as these are algorithm parameters, not user-facing config.
# Sequence: 2s, 4s, 8s, 16s, 32s - raises exception once max is reached.
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 32.0
BACKOFF_MAX_ATTEMPTS = 5

# Status codes that should not be retried - request is fundamentally invalid
NO_RETRY_STATUS_CODES = {400, 401, 403}

# Status codes that warrant exponential backoff
EXPONENTIAL_STATUS_CODES = {429, 500, 503}

# Status codes that warrant fixed interval retry
FIXED_RETRY_STATUS_CODES = {408, 504}


class LLMClientError(Exception):
    """
    Raised when an LLM API call fails after all retry attempts
    on all configured providers.
    """
    pass


class LLMProviderError(Exception):
    """
    Raised by provider implementations to surface HTTP status codes
    and Retry-After header values to the retry logic in LLMClient.

    Providers should raise this with the HTTP status code where available,
    and retry_after if the server returned a Retry-After header.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: int | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class BaseLLMProvider:
    """
    Abstract base class for LLM providers.
    All providers must implement classify() and extract() as async methods.

    Providers should raise LLMProviderError with the HTTP status code
    and retry_after (if present in the Retry-After header) on failure,
    so that LLMClient can apply the correct retry strategy.
    """

    async def classify(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str
    ) -> ClassificationResult:
        """
        Sends a classification call to the provider.
        Must be implemented by each provider subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement classify()"
        )

    async def extract(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str,
        back_image_b64: str | None = None
    ) -> ExtractionResult:
        """
        Sends an extraction call to the provider.
        Must be implemented by each provider subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract()"
        )


class LLMClient:
    """
    Provider-agnostic async LLM client.
    Wraps a provider instance and adds retry logic, logging,
    and fallback provider support.

    Retry strategy:
        400, 401, 403:  no retry - request is fundamentally invalid
        429, 500, 503:  exponential backoff - rate limit or server error
                        if Retry-After header present, respects it
                        unconditionally
        408, 504:       fixed interval retry - timeout
        unknown errors: fixed interval retry

    Usage:
        client = LLMClient(provider=GroqProvider())
        result = await client.classify(front_image_b64, mime_type, prompt)

        # with fallback provider
        client = LLMClient(
            provider=GroqProvider(),
            fallback_provider=OpenRouterProvider()
        )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None
    ):
        self.provider = provider
        self.fallback_provider = fallback_provider

    async def classify(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str
    ) -> ClassificationResult:
        """
        Sends a classification call with retry and fallback logic.

        Args:
            front_image_b64: Base64 encoded front image
            mime_type: MIME type of the image (e.g. "image/png")
            prompt: The classification prompt string

        Returns:
            ClassificationResult

        Raises:
            LLMClientError if all retry attempts fail on all providers.
        """
        return await self._call_with_retry(
            method_name="classify",
            front_image_b64=front_image_b64,
            mime_type=mime_type,
            prompt=prompt
        )

    async def extract(
        self,
        front_image_b64: str,
        mime_type: str,
        prompt: str,
        back_image_b64: str | None = None
    ) -> ExtractionResult:
        """
        Sends an extraction call with retry and fallback logic.

        Args:
            front_image_b64: Base64 encoded front image
            mime_type: MIME type of the image (e.g. "image/png")
            prompt: The extraction prompt string
            back_image_b64: Optional base64 encoded back image

        Returns:
            ExtractionResult

        Raises:
            LLMClientError if all retry attempts fail on all providers.
        """
        return await self._call_with_retry(
            method_name="extract",
            front_image_b64=front_image_b64,
            mime_type=mime_type,
            prompt=prompt,
            back_image_b64=back_image_b64
        )

    async def _call_with_retry(
        self,
        method_name: str,
        **kwargs
    ) -> ClassificationResult | ExtractionResult:
        """
        Internal method handling retry logic and fallback provider.
        Attempts the call on the primary provider first, then the fallback
        if configured and the primary fails.

        Args:
            method_name: Provider method to call ("classify" or "extract")
            **kwargs: Arguments forwarded to the provider method
        """
        providers_to_try = [self.provider]
        if self.fallback_provider:
            providers_to_try.append(self.fallback_provider)

        last_exception = None

        for provider in providers_to_try:
            provider_name = provider.__class__.__name__
            logger.info(
                f"Attempting {method_name} using {provider_name}"
            )

            try:
                return await self._call_provider(
                    provider=provider,
                    method_name=method_name,
                    **kwargs
                )

            except LLMClientError as e:
                last_exception = e
                logger.error(
                    f"All attempts failed on {provider_name}. "
                    f"{'Trying fallback provider.' if self.fallback_provider and provider == self.provider else 'No more providers to try.'}"
                )
                continue

        raise LLMClientError(
            f"{method_name} failed on all configured providers. "
            f"Last error: {last_exception}"
        )

    async def _call_provider(
        self,
        provider: BaseLLMProvider,
        method_name: str,
        **kwargs
    ) -> ClassificationResult | ExtractionResult:
        """
        Calls a single provider method with the appropriate tenacity
        retry strategy based on the error type received.

        Retry-After and non-retryable status codes are handled before
        tenacity is invoked, keeping tenacity responsible only for
        interval calculation and attempt counting.

        Args:
            provider: The provider instance to call
            method_name: The method to call on the provider
            **kwargs: Arguments forwarded to the provider method

        Raises:
            LLMClientError if all retry attempts are exhausted.
        """
        method = getattr(provider, method_name)
        provider_name = provider.__class__.__name__
        attempt = 0

        while True:
            attempt += 1
            try:
                return await method(**kwargs)

            except LLMProviderError as e:

                # non-retryable - fail immediately
                if e.status_code in NO_RETRY_STATUS_CODES:
                    logger.error(
                        f"Status {e.status_code} is non-retryable "
                        f"on {provider_name}."
                    )
                    raise LLMClientError(
                        f"Non-retryable error {e.status_code} on "
                        f"{provider_name}: {e}"
                    )

                # respect Retry-After unconditionally before tenacity
                if e.retry_after is not None:
                    logger.info(
                        f"Server requested retry after {e.retry_after}s "
                        f"(Retry-After header)"
                    )
                    await asyncio.sleep(e.retry_after)
                    continue

                # exponential backoff via tenacity
                if e.status_code in EXPONENTIAL_STATUS_CODES:
                    try:
                        return await _retry_exponential(method, **kwargs)
                    except Exception as retry_error:
                        raise LLMClientError(
                            f"Exponential backoff exhausted on "
                            f"{provider_name}: {retry_error}"
                        )

                # fixed interval via tenacity for timeouts
                if attempt >= LLM_RETRY_ATTEMPTS:
                    raise LLMClientError(
                        f"Fixed interval retry exhausted on "
                        f"{provider_name}: {e}"
                    )
                try:
                    return await _retry_fixed(method, **kwargs)
                except Exception as retry_error:
                    raise LLMClientError(
                        f"Fixed interval retry exhausted on "
                        f"{provider_name}: {retry_error}"
                    )

            except Exception as e:
                # unexpected error - fixed interval
                if attempt >= LLM_RETRY_ATTEMPTS:
                    raise LLMClientError(
                        f"Unexpected error on {provider_name} after "
                        f"{LLM_RETRY_ATTEMPTS} attempts: {e}"
                    )
                await asyncio.sleep(LLM_RETRY_INTERVAL_SECONDS)


async def _retry_exponential(method, **kwargs):
    """
    Wraps a provider method call with tenacity exponential backoff.
    Used for rate limit (429) and server error (500, 503) responses.
    Retries until the wait interval reaches BACKOFF_MAX_SECONDS.
    """

    @retry(
        wait=wait_exponential(
            multiplier=BACKOFF_BASE_SECONDS,
            max=BACKOFF_MAX_SECONDS
        ),
        stop=stop_after_attempt(BACKOFF_MAX_ATTEMPTS),
        retry=retry_if_exception_type(LLMProviderError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _call():
        return await method(**kwargs)

    return await _call()


async def _retry_fixed(method, **kwargs):
    """
    Wraps a provider method call with tenacity fixed interval retry.
    Used for timeout (408, 504) and unexpected errors.
    Retries up to LLM_RETRY_ATTEMPTS times.
    """

    @retry(
        wait=wait_fixed(LLM_RETRY_INTERVAL_SECONDS),
        stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
        retry=retry_if_exception_type(LLMProviderError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _call():
        return await method(**kwargs)

    return await _call()