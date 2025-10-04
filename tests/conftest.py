"""Pytest configuration and fixtures."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from tests.cache_utils import cached_encode

# Import benchmark plugin
pytest_plugins = ['tests.benchmark_plugin']


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file for all tests."""
    load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def enable_backend_cache():
    """Enable filesystem cache for all backend API calls in tests."""
    patches = []

    try:
        from embedsim.backends import OpenAIBackend
        patches.append(patch.object(OpenAIBackend, "encode", cached_encode(OpenAIBackend.encode)))
    except ImportError:
        pass

    try:
        from embedsim.backends import SentenceTransformerBackend
        patches.append(patch.object(SentenceTransformerBackend, "encode", cached_encode(SentenceTransformerBackend.encode)))
    except ImportError:
        pass

    # Start all patches
    for p in patches:
        p.start()

    yield

    # Stop all patches
    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
def corpus_data():
    """Load corpus test data from JSON file."""
    fixture_path = Path(__file__).parent / "fixtures" / "corpus_data.json"
    with open(fixture_path) as f:
        return json.load(f)


def pytest_generate_tests(metafunc):
    """Parameterize test_corpus.py tests with multiple models."""
    if "test_corpus" in str(metafunc.definition.path):
        if "model_id" in metafunc.fixturenames:
            from embedsim.embedsim import MODEL_CONFIGS

            # Get all model IDs (all backends are now cached)
            all_models = list(MODEL_CONFIGS.keys())

            metafunc.parametrize("model_id", all_models)


