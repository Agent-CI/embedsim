"""Tests for group coherence similarity."""

import embedsim


def test_groupsim_returns_list():
    """Test that groupsim returns a list of floats."""
    texts = [
        "First sentence about embeddings",
        "Second sentence about vectors",
        "Third sentence about similarity"
    ]

    scores = embedsim.groupsim(texts)

    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_groupsim_coherent_texts():
    """Test coherence analysis with semantically similar texts."""
    texts = [
        "Redis is an in-memory data store",
        "Redis supports vector similarity search",
        "You can store embeddings in Redis",
        "Redis provides fast key-value access"
    ]

    scores = embedsim.groupsim(texts)

    # Should have high coherence for similar texts
    assert all(s > 0.7 for s in scores)


def test_groupsim_with_outlier():
    """Test coherence analysis with an outlier text."""
    texts = [
        "Redis is an in-memory data store",
        "Redis supports vector similarity search",
        "You can store embeddings in Redis",
        "The weather in Paris is rainy today"  # Outlier
    ]

    scores = embedsim.groupsim(texts)

    # Last score should be noticeably lower
    assert scores[3] < min(scores[:3])


def test_groupsim_identical_texts():
    """Test coherence with identical texts."""
    texts = [
        "This is a test sentence",
        "This is a test sentence",
        "This is a test sentence"
    ]

    scores = embedsim.groupsim(texts)

    # Should have very similar scores
    assert max(scores) - min(scores) < 0.01


def test_groupsim_two_texts():
    """Test coherence with just two texts."""
    texts = [
        "Machine learning is a subset of AI",
        "Artificial intelligence includes machine learning"
    ]

    scores = embedsim.groupsim(texts)

    # Should work with minimal texts
    assert len(scores) == 2
    assert all(s > 0.6 for s in scores)
    # Both scores should be identical (equidistant from centroid)
    assert abs(scores[0] - scores[1]) < 0.01


def test_groupsim_varied_topics():
    """Test coherence with completely varied topics."""
    texts = [
        "Python is a programming language",
        "The Eiffel Tower is in Paris",
        "Photosynthesis converts light to energy",
        "Basketball is played with five players"
    ]

    scores = embedsim.groupsim(texts)

    # Should return valid scores even for diverse topics
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_groupsim_unrelated_sentence():
    """Test that unrelated sentences have lower coherence."""
    texts = [
        "The cat sat on the mat",
        "A dog played in the park",
        "Quantum physics involves subatomic particle interactions"
    ]

    scores = embedsim.groupsim(texts)

    # Physics sentence should have lower coherence than animal sentences
    assert scores[2] < max(scores[0], scores[1])
