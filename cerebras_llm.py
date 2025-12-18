import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class _CompatResponse:
    """Mimic the original llm_client response shape with a .text attribute."""

    def __init__(self, text: str):
        self.text = text


class CerebrasLLM:
    """
    Compat wrapper that mirrors the old llm_client API but routes
    chat completions to the Cerebras Inference API (OpenAI-compatible).
    """

    def __init__(self):
        self.model = "llama-3.3-70b"
        if not self.model:
            raise EnvironmentError("CEREBRAS_MODEL is required (e.g. llama3.1-70b)")

        self.api_url = os.environ.get("CEREBRAS_API_URL", "https://api.cerebras.ai/v1")
        self.api_key = ""
        if not self.api_key:
            logger.warning("CEREBRAS_API_KEY not set; requests will fail until configured.")

        self.timeout = float(os.environ.get("CEREBRAS_LLM_TIMEOUT", "120"))
        self.max_retries = int(os.environ.get("CEREBRAS_LLM_RETRIES", "3"))
        self.backoff = float(os.environ.get("CEREBRAS_LLM_BACKOFF", "2"))

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> _CompatResponse:
        """Async generation with retry + backoff."""
        attempt = 0
        last_err: Optional[Exception] = None

        while attempt < self.max_retries:
            start = time.time()
            try:
                text, usage = await self._generate_once(
                    prompt=prompt,
                    temperature=temperature,
                    stream=stream,
                )

                latency = time.time() - start
                log_extra: Dict[str, Any] = {
                    "model": self.model,
                    "latency_ms": int(latency * 1000),
                }
                if usage:
                    log_extra.update(
                        {
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                        }
                    )
                logger.info("Cerebras LLM call succeeded", extra=log_extra)

                return _CompatResponse(text)
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                attempt += 1
                sleep_for = self.backoff**attempt
                logger.warning(
                    f"Cerebras LLM attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {sleep_for:.1f}s"
                )
                await asyncio.sleep(sleep_for)

        raise RuntimeError(
            f"Cerebras LLM failed after {self.max_retries} attempts"
        ) from last_err

    def generate_sync(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> _CompatResponse:
        """Synchronous wrapper around the async generate, safe in any thread."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Already in an event loop; spin up a dedicated loop for this sync call.
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(
                        self.generate(prompt, temperature=temperature, stream=stream)
                    )
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(loop)
            # We have a loop object but it's not running; use it directly.
            return loop.run_until_complete(
                self.generate(prompt, temperature=temperature, stream=stream)
            )
        except RuntimeError:
            # No current loop in this thread; create one.
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.generate(prompt, temperature=temperature, stream=stream)
                )
            finally:
                new_loop.close()

    async def _generate_once(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        temperature: float,
        stream: bool,
    ) -> (str, Optional[Dict[str, Any]]):
        
        """Single Cerebras /chat/completions call, returns (text, usage)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.api_url.rstrip('/')}/chat/completions"

        # Normalize messages: accept raw string or already-structured messages.
        messages: List[Dict[str, Any]]
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            # Assume OpenAI-style list of {role, content} or {role, parts[..].text}
            messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content")
                if content is None and "parts" in msg:
                    # Gemini-style: [{"role": "...", "parts": [{"text": "..."}]}]
                    parts = msg["parts"]
                    if parts and isinstance(parts, list):
                        content = parts[0].get("text", "")
                if content is None:
                    content = ""
                messages.append({"role": role, "content": content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            if stream:
                full_text = ""
                usage: Optional[Dict[str, Any]] = None
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line.removeprefix("data:").strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choice = (data.get("choices") or [{}])[0]
                        delta = (choice.get("delta") or {}).get("content") or ""
                        full_text += delta
                        if "usage" in data:
                            usage = data["usage"]
                return full_text, usage

            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            usage = data.get("usage")
            return text, usage


_cerebras_llm = CerebrasLLM()


async def generate(
    prompt: Union[str, List[Dict[str, Any]]],
    temperature: float = 0.0,
    stream: bool = False,
):
    """Public async API matching the old signature (returns object with .text)."""
    return await _cerebras_llm.generate(prompt, temperature=temperature, stream=stream)


def generate_sync(
    prompt: Union[str, List[Dict[str, Any]]],
    temperature: float = 0.0,
    stream: bool = False,
):
    """Public sync API matching the old signature (returns object with .text)."""
    return _cerebras_llm.generate_sync(prompt, temperature=temperature, stream=stream)


