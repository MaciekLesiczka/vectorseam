"""Database-free checks for the tuner anchor harness."""

import unittest

from seam_harness.anchor import fnv1a64, is_train


class SeamAnchorHarnessTest(unittest.TestCase):
    """Checks the independent Python FNV-1a split reference."""

    def test_fnv1a64_matches_frozen_reference_values(self) -> None:
        self.assertEqual(fnv1a64(b""), 0xCBF29CE484222325)
        self.assertEqual(fnv1a64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(fnv1a64(b"foobar"), 0x85944171F73967E8)

    def test_split_membership_is_content_stable(self) -> None:
        vector_hash = 0x85944171F73967E8
        self.assertEqual(is_train(vector_hash), is_train(vector_hash))


if __name__ == "__main__":
    unittest.main()
