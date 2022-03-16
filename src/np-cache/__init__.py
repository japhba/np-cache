from collections import namedtuple, OrderedDict
from functools import wraps

import numpy as np
from xxhash import xxh3_64


class HashedSeq(list):
    __slots__ = "hashvalue"

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = _hasher(tup)

    def __hash__(self):
        return self.hashvalue


def _hasher(tup):
    hasher = xxh3_64()

    for item in tup:
        try:
            hasher.update(bytes(item))
        except TypeError:
            hasher.update(bytes(np.array(item, dtype=object)))

    return hasher.intdigest()


def hash_key(*args, **kwargs):
    key = args
    for kwarg in kwargs.items():
        key += kwarg
    return HashedSeq(key)


_NpCacheInfo = namedtuple("NpCacheInfo", ["hits", "misses", "maxsize", "currsize"])


def np_lru_cache(user_function=None, *, maxsize=128):
    if isinstance(maxsize, int):
        if maxsize < 0:
            maxsize = 0

    def actual_np_cache(user_function):
        cache = OrderedDict()
        hits = misses = 0
        cache_len = cache.__len__
        full = False

        if maxsize is None:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal hits, misses
                key = hash_key(*args, **kwargs)
                if key not in cache:
                    misses += 1
                    cache[key] = user_function(*args, **kwargs)
                else:
                    hits += 1
                return cache[key]

        elif maxsize == 0:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal misses
                misses += 1
                return user_function(*args, **kwargs)

        else:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal hits, misses, full
                key = hash_key(*args, **kwargs)
                if key not in cache:
                    misses += 1
                    cache[key] = user_function(*args, **kwargs)
                    if full:
                        cache.popitem(last=False)
                    else:
                        full = cache_len() >= maxsize
                else:
                    hits += 1
                    cache.move_to_end(key, last=True)
                return cache[key]

        def cache_info():
            return _NpCacheInfo(hits, misses, maxsize, cache_len())

        def cache_clear():
            nonlocal hits, misses, full
            cache.clear()
            hits = misses = 0
            full = False

        _np_cache_wrapper.cache_info = cache_info
        _np_cache_wrapper.cache_clear = cache_clear
        _np_cache_wrapper.cache = cache

        return _np_cache_wrapper

    if user_function:
        return actual_np_cache(user_function)

    return actual_np_cache


__all__ = ["np_lru_cache"]
