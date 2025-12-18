import os
import json
import httpx
import asyncio
from typing import Dict, Any, Union, List

class LLMClient:
    """
    Robust Production Client for Local LLMs (Ollama).
    - Uses /api/chat for correct templating.
    - Enforces format='json' natively.
    - strict separate sync/async methods to prevent event loop errors.
    """

    def __init__(self, model_name: str = "qwen2.5:3b-instruct-q5_K_M"):
        # Make sure this matches your exact 'ollama list' model name
        self.model = model_name
        self.base_url = "http://localhost:11434/api/chat"
        self.timeout = 180.0

    def _prepare_payload(self, prompt: Union[str, List[Dict]], temperature: float) -> Dict:
        """Constructs the payload for Ollama /api/chat."""
        messages = []
        
        # Handle list of messages (Gemini/OpenAI style)
        if isinstance(prompt, list):
            for msg in prompt:
                role = msg.get("role", "user")
                # Handle Gemini's nested "parts" structure or standard "content"
                content = ""
                if "parts" in msg:
                    content = msg["parts"][0].get("text", "")
                else:
                    content = msg.get("content", "")
                
                messages.append({"role": role, "content": content})
        # Handle raw string prompt
        else:
            messages.append({"role": "user", "content": prompt})

        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",  # NATIVE JSON ENFORCEMENT
            "options": {
                "temperature": temperature,
                "num_ctx": 4096,  # Increased context window
            }
        }

    async def generate(self, prompt, temperature: float = 0.0) -> Any:
        """Async generation."""
        payload = self._prepare_payload(prompt, temperature)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.base_url, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
            except Exception as e:
                print(f"LLM Error: {e}")
                # Return empty valid JSON structure on failure to prevent crashes
                return type('obj', (object,), {'text': '{}'})

    def generate_sync(self, prompt, temperature: float = 0.0) -> Any:
        """
        True synchronous generation using blocking HTTP.
        Avoids asyncio loop conflicts in scripts.
        """
        payload = self._prepare_payload(prompt, temperature)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.base_url, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
        except Exception as e:
            print(f"LLM Sync Error: {e}")
            return type('obj', (object,), {'text': '{}'})

    def _parse_response(self, data: Dict) -> Any:
        """Normalize response to match Gemini's object structure."""
        if not data or "message" not in data:
            return type('obj', (object,), {'text': '{}'})

        content = data["message"]["content"]
        
        # Return object compatible with .text attribute
        return type('obj', (object,), {'text': content})
