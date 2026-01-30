"""Provider latency test script.

Examples:
    python scripts/provider_speed_test.py --provider openrouter --prompt "ping"
    python scripts/provider_speed_test.py --provider openai --model gpt-4o-mini --repeat 3
    python scripts/provider_speed_test.py --provider openrouter --use-proxy
"""

from __future__ import annotations

import argparse
import asyncio
import time
from statistics import mean

from llm_adapter import ConfigManager
from llm_adapter.adapter import LLMAdapter


def _pick_model(provider_config) -> str:
    return (
        provider_config.default_model
        or provider_config.models.normal
        or provider_config.models.cheap
        or provider_config.models.premium
        or provider_config.models.multimodal
    )


async def _run_once(adapter, prompt: str, model: str) -> float:
    start = time.perf_counter()
    result = await adapter.generate(prompt, model)
    elapsed = time.perf_counter() - start
    return elapsed, result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test provider latency with optional proxy.")
    parser.add_argument("--provider", required=True, help="Provider name (openai/gemini/openrouter/etc)")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--prompt", default="ping", help="Prompt for the test call")
    parser.add_argument("--repeat", type=int, default=1, help="Number of calls to average")
    parser.add_argument("--use-proxy", action="store_true", help="Force use proxy if available")
    parser.add_argument("--proxy-url", help="Override proxy URL (e.g. http://host:port)")
    args = parser.parse_args()

    config_manager = ConfigManager()
    config_manager.load_env_file()
    provider = args.provider.lower()

    if provider not in LLMAdapter.PROVIDER_ADAPTERS:
        raise SystemExit(f"Unsupported provider: {provider}")

    provider_config = config_manager.get_provider_config(provider)
    adapter_class = LLMAdapter.PROVIDER_ADAPTERS[provider]

    kwargs: dict[str, object] = {}
    if provider_config.base_url:
        kwargs["base_url"] = provider_config.base_url
    if provider_config.account_id:
        kwargs["account_id"] = provider_config.account_id

    proxy_url = args.proxy_url or config_manager.get_proxy_url()
    if args.use_proxy and proxy_url:
        kwargs["proxy_url"] = proxy_url

    adapter = adapter_class(api_key=provider_config.api_key, **kwargs)
    model = args.model or _pick_model(provider_config)
    if not model:
        raise SystemExit("No model available; provide --model or configure models/default_model.")

    times: list[float] = []
    last_result = None
    if args.prompt == "ping":
        args.prompt = "Please respond with just the sentences: 'Hello, world!'"
    for _ in range(max(1, args.repeat)):
        elapsed, last_result = await _run_once(adapter, args.prompt, model)
        times.append(elapsed)

    print("\n--- Provider speed test ---")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Use proxy: {bool(kwargs.get('proxy_url'))}")
    print(f"Repeat: {len(times)}")
    print(f"Latency avg: {mean(times):.3f}s")
    print(f"Latency min: {min(times):.3f}s")
    print(f"Latency max: {max(times):.3f}s")

    if last_result is not None:
        print(f"Input tokens: {last_result.input_tokens}")
        print(f"Output tokens: {last_result.output_tokens}")
        print(f"Text preview: {last_result.text[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
