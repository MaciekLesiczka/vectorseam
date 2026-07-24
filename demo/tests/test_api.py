"""Tests for the demo search API helpers."""

import unittest

import numpy as np

from demo.api import app


class _FakeModel:
    """Records one model encode call."""

    def __init__(self) -> None:
        self.inputs = None
        self.kwargs = None

    def encode(self, inputs, **kwargs):
        self.inputs = inputs
        self.kwargs = kwargs
        return np.arange(
            app.EMBEDDING_DIMENSION, dtype=np.float32
        ).reshape(1, -1)


class ApiTest(unittest.TestCase):
    """Verifies benchmark-compatible embedding and startup settings."""

    def test_embed_query_matches_benchmark_options(self) -> None:
        model = _FakeModel()

        vector = app._embed_query(model, "disk recovery")

        self.assertEqual(["disk recovery"], model.inputs)
        self.assertEqual(
            {
                "batch_size": app.MODEL_BATCH_SIZE,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
            model.kwargs,
        )
        self.assertEqual((app.EMBEDDING_DIMENSION,), vector.shape)
        self.assertEqual(np.dtype("<f4"), vector.dtype)
        self.assertTrue(vector.flags.c_contiguous)

    def test_settings_use_m1_defaults(self) -> None:
        settings = app.Settings.from_environment({})

        self.assertEqual("127.0.0.1", settings.collector_host)
        self.assertEqual(7737, settings.collector_port)
        self.assertEqual(100, settings.ef_search)

    def test_settings_reject_invalid_environment_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "COLLECTOR_PORT must be an integer"
        ):
            app.Settings.from_environment({"COLLECTOR_PORT": "invalid"})
        with self.assertRaisesRegex(
            ValueError, "DEMO_EF_SEARCH must be between"
        ):
            app.Settings.from_environment({"DEMO_EF_SEARCH": "0"})


if __name__ == "__main__":
    unittest.main()
