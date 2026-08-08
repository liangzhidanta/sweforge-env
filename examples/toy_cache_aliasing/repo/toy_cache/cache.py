"""A tiny keyed cache. Entries are indexed by their computed value (bug)."""

_CACHE: dict[object, object] = {}


def get_or_compute(key: str, compute) -> object:
    """Return a cached value for ``key``, computing it once if absent."""
    value = compute()
    if value not in _CACHE:
        _CACHE[value] = value
    return _CACHE[value]
