"""Tests for the demo search API helpers."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from pydantic import ValidationError

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

    def test_search_request_selects_supported_cohort(self) -> None:
        default_request = app.SearchRequest(query="disk recovery")
        reddit_request = app.SearchRequest(
            query="today I learned", cohort="reddit"
        )

        self.assertEqual(app.CohortName.SUPERUSER, default_request.cohort)
        self.assertEqual(app.CohortName.REDDIT, reddit_request.cohort)
        self.assertEqual(
            "docs_reddit", app.COHORT_TABLES[reddit_request.cohort]
        )

    def test_search_request_rejects_unknown_cohort(self) -> None:
        with self.assertRaises(ValidationError):
            app.SearchRequest(query="query", cohort="unknown")

    def test_search_captures_and_queries_selected_cohort(self) -> None:
        settings = app.Settings.from_environment({})
        producer = object()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    model=_FakeModel(),
                    producer=producer,
                    settings=settings,
                )
            )
        )
        payload = app.SearchRequest(query="today I learned", cohort="reddit")
        with (
            mock.patch.object(app, "capture_vector") as capture_vector,
            mock.patch.object(
                app, "_search_database", return_value=([], 1.25)
            ) as search_database,
        ):
            response = app.search(payload, request)

        captured_args = capture_vector.call_args
        self.assertEqual("reddit", captured_args.args[0])
        self.assertIs(producer, captured_args.kwargs["producer"])
        self.assertEqual(
            app.CohortName.REDDIT, search_database.call_args.args[3]
        )
        self.assertEqual(app.CohortName.REDDIT, response.cohort)


if __name__ == "__main__":
    unittest.main()
