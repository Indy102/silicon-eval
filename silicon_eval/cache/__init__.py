"""Content-addressed result cache."""

from silicon_eval.cache.store import ResultCache, cache_key, default_cache_dir

__all__ = ["ResultCache", "cache_key", "default_cache_dir"]
