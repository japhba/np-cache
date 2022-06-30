# np-cache

This module provides a "cache-by-value" decorator for `numpy.ndarrays` similar to the Python Standard Library's `@lru_cache` function decorator.

```python
import numpy as np
from np_cache import np_lru_cache

@np_lru_cache
def cached_inverse(arr: np.ndarray):
    return np.linalg.inv(arr)

@np_lru_cache(maxsize=16)
def cached_inverse(arr: np.ndarray):
    return np.linalg.inv(arr)
```

`@np_lru_cache` uses `xxhash`, a lightning-fast hashing library, to make cache keys by value. Much like `@lru_cache`, users can set the maximum size of the `@np_lru_cache` and examine/clear the cache using `func.cache_info()` and `func.cache_clear()`. Unlike `@lru_cache`, however, `@np_lru_cache` is not thread-safe.

`@np_lru_cache` is completely transparent to annotations, and is therefore fully compatible with type hints.