"""Third-party tool endpoints (weather, research) over a shared httpx pool."""

import os
from typing import Any

import httpx


class ToolAPIClient:
    def __init__(self):
        base_url = os.environ.get("ELYOS_API_BASE", "")
        api_key = os.environ.get("ELYOS_API_KEY", "")
        if not base_url:
            raise RuntimeError("ELYOS_API_BASE is not set")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": api_key} if api_key else {},
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def __aenter__(self) -> "ToolAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.aclose()

    async def get_weather(self, location: str) -> dict[str, Any]:
        return await self._get("/weather", {"location": location})

    async def research_topic(self, topic: str) -> dict[str, Any]:
        return await self._get("/research", {"topic": topic})

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            resp = await self._client.get(path, params=params)
        except httpx.TimeoutException:
            return {"ok": False, "error": "timeout"}
        except httpx.RequestError as e:
            return {"ok": False, "error": f"network_error: {type(e).__name__}"}

        if 400 <= resp.status_code < 500:
            return {"ok": False, "error": f"client_error: {resp.status_code}", "body": resp.text[:200]}
        if resp.status_code >= 500:
            return {"ok": False, "error": f"server_error: {resp.status_code}"}

        try:
            data = resp.json()
        except ValueError:
            return {"ok": False, "error": "invalid_response: not json"}

        if not isinstance(data, dict):
            return {"ok": False, "error": "invalid_response: not an object"}

        return {"ok": True, "data": data}
