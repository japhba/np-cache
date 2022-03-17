import unittest
from copy import deepcopy
from itertools import product
from unittest import TestCase

import numpy as np
from np_cache import cacheimpl
from numpy.testing import assert_almost_equal


class TestMakeHashKey(TestCase):
    def test_make_hash_key(self):
        """Test several combinations of args and kwargs and check that
        they produce the same hash.
        """
        test_args = (
            ("a",),
            (1,),
            (
                np.array(
                    [1.214],
                )
            ),
            ("a", 1),
            (2, 2.574),
            (np.array([[1, 1], [1, 2], [1, 3]]), "hello", 3.44),
        )

        test_kwargs = (
            {"a": 2},
            {"key": 3.14, "arr": np.array([2, 5.55, "1"])},
            {},
            {"a": "a", "b": "b", "c": np.array([1, 1])},
        )

        for args, kwargs in product(test_args, test_kwargs):

            copy_args = deepcopy(args)
            copy_kwargs = deepcopy(kwargs)

            orig_hash = cacheimpl._make_hash_key(*args, **kwargs)
            copy_hash = cacheimpl._make_hash_key(*copy_args, **copy_kwargs)

            self.assertEqual(orig_hash.hashvalue, copy_hash.hashvalue)


class TestNpLruCache(TestCase):
    def test_single_arg_func(self):

        test_arr = np.eye(1000, dtype=np.float64)

        expected = np.linalg.inv(test_arr)

        cached_inv = cacheimpl.np_lru_cache(np.linalg.inv)
        # burn the first computation
        cached_inv(test_arr)
        generated = cached_inv(test_arr)

        assert_almost_equal(generated, expected)

        # check that result came from cache
        self.assertEqual(cached_inv.cache_info().hits, 1)

    def test_double_arg_func(self):

        X = np.eye(3).repeat(3, axis=0)
        y = np.random.random(size=len(X))

        expected = np.linalg.lstsq(X, y)

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq)
        # burn the first computation
        cached_lstsq(X, y)
        generated = cached_lstsq(X, y)

        for gen, exp in zip(generated, expected):
            assert_almost_equal(gen, exp)

        # check that result came from cache
        self.assertEqual(cached_lstsq.cache_info().hits, 1)

        # check that providing a different optional arg results
        # in a cache miss, counting the prior miss
        cached_lstsq(X, y, rcond=-1)
        self.assertEqual(cached_lstsq.cache_info().misses, 2)

    def test_wrapper_args(self):

        X = np.eye(3).repeat(3, axis=0)
        y = np.random.random(size=len(X))

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq, maxsize=None)
        cached_lstsq(X, y)
        cached_lstsq(X, y)

        self.assertEqual(cached_lstsq.cache_info().maxsize, None)
        self.assertEqual(cached_lstsq.cache_info().currsize, 1)

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq, maxsize=128)
        cached_lstsq(X, y)

        self.assertEqual(cached_lstsq.cache_info().maxsize, 128)
        self.assertEqual(cached_lstsq.cache_info().currsize, 1)

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq, maxsize=-10)
        cached_lstsq(X, y)

        self.assertEqual(cached_lstsq.cache_info().maxsize, 0)
        self.assertEqual(cached_lstsq.cache_info().currsize, 0)

    def test_cache_manipulation(self):

        X = np.eye(3).repeat(3, axis=0)
        y = np.random.random(size=len(X))

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq, maxsize=None)
        cached_lstsq(X, y)

        self.assertEqual(cached_lstsq.cache_info().currsize, 1)

        cached_lstsq.cache_clear()
        self.assertEqual(cached_lstsq.cache_info().currsize, 0)

    def test_cache_eviction(self):

        X = np.eye(3).repeat(3, axis=0)
        y = np.random.random(size=len(X))
        z = np.random.random(size=len(X))

        cached_lstsq = cacheimpl.np_lru_cache(np.linalg.lstsq, maxsize=1)
        cached_lstsq(X, y)

        self.assertEqual(cached_lstsq.cache_info().currsize, 1)

        out = cached_lstsq(X, z)
        self.assertEqual(cached_lstsq.cache_info().currsize, 1)
        # make sure our newest result kicked out the old result
        self.assertIn(out, cached_lstsq.cache.values())


if __name__ == "__main__":
    unittest.main()
