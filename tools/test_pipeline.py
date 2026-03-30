# tools/test_pipeline.py
import asyncio
from pathlib import Path
from app.core.config_loader import load_sets_config
from app.core.llm.client import LLMClient
from app.core.llm.providers.groq import GroqProvider
from app.core.pipeline.modes.single_set import SingleSetConfig, run_single_set
from app.utils.file_walker import ImageSortOrder
from app.core.pipeline.pairing import ImageMode

async def main():
    sets_config = load_sets_config()
    provider = GroqProvider()
    client = LLMClient(provider=provider)

    config = SingleSetConfig(
        source_path=Path("test_images/"),
        sets_config=sets_config,
        manufacturer_key="panini",       # or None for discovery
        set_key="your_set_key",          # or None for discovery
        image_mode=ImageMode.FRONT_BACK, # or UNKNOWN to infer
        max_depth=0,
        sort_order=ImageSortOrder.FILESYSTEM,
        output_dir=Path("outputs/test")
    )

    candidates_path, deferred_path = await run_single_set(config, client)
    print(f"Candidates: {candidates_path}")
    print(f"Deferred: {deferred_path}")

if __name__ == "__main__":
    asyncio.run(main())