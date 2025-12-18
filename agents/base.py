"""Base utilities for agents."""
import logging
import hashlib
import json
from pathlib import Path
from backend.app.config import get_settings
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple file-based cache for API responses."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path("/tmp/runs/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    
    def _hash_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Create a hash key for cache lookup."""
        key_str = json.dumps(
            {"endpoint": endpoint, "params": params},
            sort_keys=True
        )
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached response."""
        hash_key = self._hash_key(endpoint, params)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            logger.warning(f"Cache hit: {endpoint}")
            return cached.get("data")
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        """Store response in cache."""
        hash_key = self._hash_key(endpoint, params)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump({"data": data}, f)
            logger.warning(f"Cache written: {endpoint}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


cache_manager = CacheManager()
