"""Tests for pairwise similarity."""

import embedsim


def test_pairsim_returns_float():
    """Test that pairsim returns a single float."""
    score = embedsim.pairsim(
        "The cat sat on the mat",
        "A feline rested on the rug"
    )

    assert isinstance(score, float)


def test_pairsim_similar_texts():
    """Test pairwise similarity with similar texts."""
    score = embedsim.pairsim(
        "The cat sat on the mat",
        "A feline rested on the rug"
    )

    # Should be moderately high for similar meanings
    assert score > 0.6


def test_pairsim_identical_texts():
    """Test pairwise similarity with identical texts."""
    text = "the quick brown dog jumped over the lazy dog"
    score = embedsim.pairsim(text, text)

    # Should be very close to 1.0
    assert score > 0.99
    assert score <= 1.01


def test_pairsim_unrelated_texts():
    """Test pairwise similarity with unrelated texts."""
    score = embedsim.pairsim(
        "Python is a programming language",
        "Pizza is a popular food"
    )

    # Should be low for unrelated texts
    assert score < 0.5


def test_pairsim_range():
    """Test that pairsim returns score in valid range."""
    score = embedsim.pairsim(
        "Machine learning uses neural networks",
        "Deep learning is a subset of ML"
    )

    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0
