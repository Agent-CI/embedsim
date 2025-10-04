"""Tests for larger corpus analysis using fixtures."""

import numpy as np
import embedsim


def test_small_coherent(corpus_data, model_id, benchmark):
    """Test coherence within small ML-focused texts (all rewritten variations)."""
    texts = corpus_data["small_coherent"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics for benchmarking
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark performance criteria
    if not all(s > 0.65 for s in scores):
        benchmark.add_failure(f"Low coherence: min={min(scores):.4f}")

    if round(score_range, 2) > 0.10:
        benchmark.add_failure(f"High variance: range={score_range:.4f}")


def test_small_mixed(corpus_data, model_id, benchmark):
    """Test small corpus with mixed unrelated topics."""
    texts = corpus_data["small_mixed"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark: mixed topics should have variance
    if round(score_range, 2) < 0.10:
        benchmark.add_failure(f"Low variance: range={score_range:.4f}")


def test_small_outlier(corpus_data, model_id, benchmark):
    """Test outlier detection in small corpus (ML texts + rabbit story)."""
    texts = corpus_data["small_outlier"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    outlier_score = scores[4]
    ml_scores = scores[:4]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    benchmark.add_metric("outlier_score", outlier_score)
    benchmark.add_metric("ml_min", min(ml_scores))
    benchmark.add_metric("ml_mean", sum(ml_scores) / len(ml_scores))
    benchmark.add_metric("std_dev", std_score)

    # Benchmark: outlier detection
    if outlier_score >= mean_score - 1.5 * std_score:
        benchmark.add_failure(f"Outlier not detected: {outlier_score:.4f} vs threshold {mean_score - 1.5 * std_score:.4f}")

    if not all(s > 0.65 for s in ml_scores):
        benchmark.add_failure(f"Low ML coherence: min={min(ml_scores):.4f}")


def test_medium_coherent(corpus_data, model_id, benchmark):
    """Test coherence within medium-length ML texts (all rewritten variations)."""
    texts = corpus_data["medium_coherent"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark performance
    if not all(s > 0.65 for s in scores):
        benchmark.add_failure(f"Low coherence: min={min(scores):.4f}")

    if round(score_range, 2) > 0.10:
        benchmark.add_failure(f"High variance: range={score_range:.4f}")


def test_medium_mixed(corpus_data, model_id, benchmark):
    """Test medium corpus with diverse topics."""
    texts = corpus_data["medium_mixed"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark: mixed should have variance
    if round(score_range, 2) < 0.10:
        benchmark.add_failure(f"Low variance: range={score_range:.4f}")


def test_medium_outlier(corpus_data, model_id, benchmark):
    """Test outlier detection in medium corpus (ML + library description)."""
    texts = corpus_data["medium_outlier"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    outlier_score = scores[4]
    ml_scores = scores[:4]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    benchmark.add_metric("outlier_score", outlier_score)
    benchmark.add_metric("ml_min", min(ml_scores))
    benchmark.add_metric("ml_mean", sum(ml_scores) / len(ml_scores))
    benchmark.add_metric("std_dev", std_score)

    # Benchmark: outlier detection
    if outlier_score >= mean_score - 1.5 * std_score:
        benchmark.add_failure(f"Outlier not detected: {outlier_score:.4f} vs threshold {mean_score - 1.5 * std_score:.4f}")

    if not all(s > 0.65 for s in ml_scores):
        benchmark.add_failure(f"Low ML coherence: min={min(ml_scores):.4f}")


def test_large_coherent(corpus_data, model_id, benchmark):
    """Test coherence within large ML texts (all rewritten variations)."""
    texts = corpus_data["large_coherent"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark performance
    if not all(s > 0.65 for s in scores):
        benchmark.add_failure(f"Low coherence: min={min(scores):.4f}")

    if round(score_range, 2) > 0.10:
        benchmark.add_failure(f"High variance: range={score_range:.4f}")


def test_large_mixed(corpus_data, model_id, benchmark):
    """Test large corpus with diverse topics."""
    texts = corpus_data["large_mixed"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    score_range = max(scores) - min(scores)
    benchmark.add_metric("min_score", min(scores))
    benchmark.add_metric("max_score", max(scores))
    benchmark.add_metric("range", score_range)
    benchmark.add_metric("mean_score", sum(scores) / len(scores))

    # Benchmark: mixed should have variance
    if round(score_range, 2) < 0.10:
        benchmark.add_failure(f"Low variance: range={score_range:.4f}")


def test_large_outlier(corpus_data, model_id, benchmark):
    """Test outlier detection in large corpus (ML + lighthouse story)."""
    texts = corpus_data["large_outlier"]
    scores = embedsim.groupsim(texts, model_id=model_id)

    # Sanity checks
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    outlier_score = scores[4]
    ml_scores = scores[:4]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    benchmark.add_metric("outlier_score", outlier_score)
    benchmark.add_metric("ml_min", min(ml_scores))
    benchmark.add_metric("ml_mean", sum(ml_scores) / len(ml_scores))
    benchmark.add_metric("std_dev", std_score)

    # Benchmark: outlier detection
    if outlier_score >= mean_score - 1.5 * std_score:
        benchmark.add_failure(f"Outlier not detected: {outlier_score:.4f} vs threshold {mean_score - 1.5 * std_score:.4f}")

    if not all(s > 0.65 for s in ml_scores):
        benchmark.add_failure(f"Low ML coherence: min={min(ml_scores):.4f}")


def test_pairwise_long_texts(corpus_data, model_id, benchmark):
    """Test pairwise similarity with long texts."""
    # Same domain - both ML
    score_ml = embedsim.pairsim(
        corpus_data["large_coherent"][0],  # Transformers
        corpus_data["large_coherent"][1],  # Gradient Descent
        model_id=model_id
    )

    # Different domains
    score_cross = embedsim.pairsim(
        corpus_data["large_coherent"][0],  # Transformers (ML)
        corpus_data["large_mixed"][1],     # Cardiovascular (Biology)
        model_id=model_id
    )

    # Sanity checks
    assert 0.0 <= score_ml <= 1.0
    assert 0.0 <= score_cross <= 1.0

    # Record metrics
    diff = abs(score_ml - score_cross)
    benchmark.add_metric("ml_similarity", score_ml)
    benchmark.add_metric("cross_similarity", score_cross)
    benchmark.add_metric("difference", diff)

    # Benchmark: should distinguish domains
    if round(diff, 2) < 0.10:
        benchmark.add_failure(f"Low domain distinction: diff={diff:.4f}")


def test_text_length_scaling(corpus_data, model_id, benchmark):
    """Test that similarity works consistently across text lengths."""
    small_scores = embedsim.groupsim(corpus_data["small_coherent"], model_id=model_id)
    medium_scores = embedsim.groupsim(corpus_data["medium_coherent"], model_id=model_id)
    large_scores = embedsim.groupsim(corpus_data["large_coherent"], model_id=model_id)

    # Sanity checks
    for scores in [small_scores, medium_scores, large_scores]:
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)

    # Record metrics
    benchmark.add_metric("small_mean", sum(small_scores) / len(small_scores))
    benchmark.add_metric("medium_mean", sum(medium_scores) / len(medium_scores))
    benchmark.add_metric("large_mean", sum(large_scores) / len(large_scores))

    # Benchmark: all should maintain coherence
    if not all(s > 0.65 for s in small_scores):
        benchmark.add_failure(f"Low small coherence: min={min(small_scores):.4f}")

    if not all(s > 0.65 for s in medium_scores):
        benchmark.add_failure(f"Low medium coherence: min={min(medium_scores):.4f}")

    if not all(s > 0.65 for s in large_scores):
        benchmark.add_failure(f"Low large coherence: min={min(large_scores):.4f}")
