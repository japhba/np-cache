from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from functools import singledispatch, wraps
from typing import Callable, Optional, TypeVar, cast

import numpy as np
from xxhash import xxh3_128

__all__ = ["np_lru_cache"]

TCallable = TypeVar("TCallable", bound=Callable)


def np_lru_cache(
    user_function: TCallable = None, *, maxsize: Optional[int] = 16
) -> TCallable:
    """Wrapper similar to functool's lru_cache, but can handle caching numpy arrays.
    Uses xxhash to hash the raw bytes of the argument array(s) + shape information
    to prevent collisions on arrays with identical data but different dimensions.

    Intentionally has a smaller default maxsize than lru_cache - if you're using this
    wrapper, you are likely trying to avoid some slow computations on large arrays.
    There's no reason to hold onto 128 of these in memory unless you have to.

    Does not have the thread-safety features of lru_cache.

    Parameters
    ----------
    user_function : TCallable, optional
    maxsize : int, optional
        Max number of entries in the cache. None for no limit, by default 16

    Returns
    -------
    TCallable
        Wrapped function. Should by mypy-compliant.
    """

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
        return cast(TCallable, actual_np_cache)(user_function)

    return cast(TCallable, actual_np_cache)


class HashedSeq(list):
    __slots__ = "hashvalue"

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = _hasher(tup)

    def __hash__(self):
        return self.hashvalue


def _hasher(tup):
    hasher = xxh3_128()

    for item in tup:
        hasher.update(hashable_representation(item))

    return hasher.intdigest()


@singledispatch
def hashable_representation(obj):
    # this is the generic path and it may result in collisions if the
    # repr of an object omits information, i.e. np arrays
    return str(obj)


@hashable_representation.register
def _(obj: np.ndarray):
    hasher = xxh3_128()
    # tobytes() does not include dimension information, but we want to
    # avoid collisions there
    hasher.update(bytes(str(obj.shape), encoding="UTF-8"))
    hasher.update(obj.tobytes())

    return str(hasher.hexdigest())


@hashable_representation.register
def _(obj: str):
    return obj


@hashable_representation.register
def _(obj: Iterable):
    return b" ".join(hashable_representation(subobj) for subobj in obj)


def hash_key(*args, **kwargs):
    key = args
    for kwarg in kwargs.items():
        key += kwarg
    return HashedSeq(key)


_NpCacheInfo = namedtuple("NpCacheInfo", ["hits", "misses", "maxsize", "currsize"])
