"""Database-free checks for the tuner anchor harness."""

import unittest

from seam_harness import anchor


class SeamAnchorHarnessTest(unittest.TestCase):
    """Checks the independent Python FNV-1a split reference."""

    def test_fnv1a64_matches_frozen_reference_values(self) -> None:
        self.assertEqual(anchor.fnv1a64(b""), 0xCBF29CE484222325)
        self.assertEqual(anchor.fnv1a64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(anchor.fnv1a64(b"foobar"), 0x85944171F73967E8)

    def test_split_membership_is_content_stable(self) -> None:
        vector_hash = 0x85944171F73967E8
        self.assertEqual(
            anchor.is_train(vector_hash), anchor.is_train(vector_hash)
        )

    def test_product_selection_checks_grid_maximum_before_lower_efs(
        self,
    ) -> None:
        class Analyze:
            @staticmethod
            def _p10_for_subset(
                rows: list[dict[str, object]],
                dataset: str,
                ef_search: int,
                query_ids: set[int],
            ) -> float:
                return 1.0

        train_ids = set(range(22))
        test_ids = set(range(100, 122))
        rows = []
        for ef_search in (10, 20, 40, 80, 160):
            for query_id in train_ids:
                rows.append(
                    {
                        "ef": ef_search,
                        "query_id": query_id,
                        "recall": 1.0 if ef_search == 20 else 0.0,
                    }
                )
            for query_id in test_ids:
                rows.append(
                    {
                        "ef": ef_search,
                        "query_id": query_id,
                        "recall": 1.0,
                    }
                )

        observed = anchor._product_calibration(
            Analyze(), rows, train_ids, test_ids
        )

        self.assertEqual(observed["recommended_ef"], 160)
        self.assertLess(observed["train_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
