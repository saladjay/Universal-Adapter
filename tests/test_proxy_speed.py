"""Speed comparison test for proxy vs direct requests.

Enable by setting RUN_PROXY_SPEED_TEST=1.
Optionally set PROXY_TEST_URL (default: https://httpbin.org/get).
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from llm_adapter.config import ConfigManager


@pytest.mark.integration
def test_proxy_speed_comparison() -> None:
    """Compare request latency with and without proxy (manual/integration test)."""
    if os.getenv("RUN_PROXY_SPEED_TEST") != "1":
        pytest.skip("Set RUN_PROXY_SPEED_TEST=1 to enable proxy speed test.")

    url = os.getenv("PROXY_TEST_URL", "https://httpbin.org/get")
    config = ConfigManager().load()
    proxy_url = ConfigManager().get_proxy_url()
    if not proxy_url:
        pytest.skip("Proxy is not enabled in config.yaml.")

    def measure(client: httpx.Client) -> float:
        start = time.perf_counter()
        response = client.get(url)
        response.raise_for_status()
        return time.perf_counter() - start

    with httpx.Client(timeout=30.0) as direct_client:
        direct_time = measure(direct_client)

    with httpx.Client(timeout=30.0, proxies=proxy_url) as proxy_client:
        proxy_time = measure(proxy_client)

    print("\n--- Proxy speed test ---")
    print(f"URL: {url}")
    print(f"Direct: {direct_time:.3f}s")
    print(f"Proxy: {proxy_time:.3f}s")
