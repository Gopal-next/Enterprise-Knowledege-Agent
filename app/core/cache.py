"""
In-memory Query Caching Layer with TTL, LRU Eviction, and Performance Telemetry.
Provides ultra-low latency (<5ms) responses for repeated enterprise questions,
dramatically slashing LLM token consumption and API costs.
"""

import time
import re
import threading
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict


def normalize_query(query: str) -> str:
    """Normalize query text for cache key lookup (lowercase, stripped whitespace & punctuation)."""
    clean = query.lower().strip()
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean


class QueryCache:
    """Thread-safe In-Memory LRU Cache with Time-To-Live (TTL) support."""

    def __init__(self, max_size: int = 256, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Telemetry metrics
        self.hits = 0
        self.misses = 0
        self.total_saved_ms = 0.0

    def get(self, query: str, context_tag: str = "") -> Optional[Dict[str, Any]]:
        """
        Check if query result is cached and not expired.
        Returns cached payload or None.
        """
        key = f"{context_tag}:{normalize_query(query)}"
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]
            now = time.time()

            # TTL expiration check
            if now > entry["expires_at"]:
                del self._cache[key]
                self.misses += 1
                return None

            # Move to end for LRU policy
            self._cache.move_to_end(key)
            self.hits += 1
            saved_ms = entry.get("generation_time_ms", 1200.0)
            self.total_saved_ms += saved_ms

            return {
                "answer": entry["answer"],
                "tool_used": entry["tool_used"],
                "cached": True,
                "cached_at": entry["cached_at"],
                "saved_time_ms": saved_ms,
                "evaluation": entry.get("evaluation")
            }

    def set(
        self,
        query: str,
        answer: str,
        tool_used: str,
        generation_time_ms: float = 0.0,
        context_tag: str = "",
        ttl: Optional[int] = None,
        evaluation: Optional[Dict[str, Any]] = None
    ):
        """Store query and synthesized response in cache."""
        key = f"{context_tag}:{normalize_query(query)}"
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        now = time.time()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest item (FIFO/LRU head)
                self._cache.popitem(last=False)

            self._cache[key] = {
                "answer": answer,
                "tool_used": tool_used,
                "cached_at": now,
                "expires_at": now + ttl_seconds,
                "generation_time_ms": generation_time_ms,
                "evaluation": evaluation
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health, hit rate, and latency savings metrics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_queries": total,
                "hit_rate_pct": round(hit_rate, 1),
                "cached_entries": len(self._cache),
                "max_size": self.max_size,
                "saved_latency_sec": round(self.total_saved_ms / 1000.0, 2)
            }

    def clear(self):
        """Purge all cache entries."""
        with self._lock:
            self._cache.clear()


# Global Singleton instance
global_cache = QueryCache(max_size=256, default_ttl=3600)
