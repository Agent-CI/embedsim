"""Pytest plugin for benchmarking model performance instead of pass/fail."""

import pytest


# Global storage for benchmark results
benchmark_results = {}


class BenchmarkResult:
    """Store benchmark metrics for a test."""
    def __init__(self, test_name, model_id):
        self.test_name = test_name
        self.model_id = model_id
        self.metrics = {}
        self.passed = True
        self.failures = []

    def add_metric(self, name, value):
        """Add a metric (e.g., 'coherence_range', 'min_score', 'max_score')."""
        self.metrics[name] = value

    def add_failure(self, assertion_msg):
        """Record an assertion failure."""
        self.passed = False
        self.failures.append(assertion_msg)


@pytest.fixture
def benchmark(request):
    """Fixture to record benchmark metrics instead of failing tests."""
    test_name = request.node.name.split('[')[0]  # Remove parameter suffix
    model_id = request.node.callspec.params.get('model_id', 'unknown')

    result = BenchmarkResult(test_name, model_id)

    # Store in global dict
    if test_name not in benchmark_results:
        benchmark_results[test_name] = {}
    benchmark_results[test_name][model_id] = result

    return result


def pytest_sessionfinish(session, exitstatus):
    """Generate benchmark report after all tests complete."""
    if not benchmark_results:
        return

    print(f"\n{'='*100}")
    print("MODEL PERFORMANCE BENCHMARK")
    print(f"{'='*100}\n")

    # Collect all models and tests
    all_models = set()
    for models in benchmark_results.values():
        all_models.update(models.keys())
    all_models = sorted(all_models)

    # Print model comparison table
    print("PERFORMANCE SUMMARY")
    print("-" * 100)
    print(f"{'Model':<35} | {'Tests Passed':<15} | {'Avg Range':<12} | {'Avg Score':<12}")
    print("-" * 100)

    for model_id in all_models:
        total_tests = 0
        passed_tests = 0
        total_range = 0
        total_score = 0

        for test_name, models in benchmark_results.items():
            if model_id in models:
                result = models[model_id]
                total_tests += 1
                if result.passed:
                    passed_tests += 1
                if "range" in result.metrics:
                    total_range += result.metrics["range"]
                if "mean_score" in result.metrics:
                    total_score += result.metrics["mean_score"]

        avg_range = total_range / total_tests if total_tests > 0 else 0
        avg_score = total_score / total_tests if total_tests > 0 else 0
        pass_rate = f"{passed_tests}/{total_tests}"

        print(f"{model_id:<35} | {pass_rate:<15} | {avg_range:<12.4f} | {avg_score:<12.4f}")

    # Detailed results per test
    print(f"\n{'='*100}")
    print("DETAILED RESULTS BY TEST")
    print(f"{'='*100}\n")

    for test_name, models in sorted(benchmark_results.items()):
        print(f"\n{test_name}")
        print("-" * 100)

        # Sort by mean_score descending to show best performers first
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].metrics.get("mean_score", 0),
            reverse=True
        )

        for model_id, result in sorted_models:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            metrics_str = _format_metrics_compact(result.metrics)
            print(f"  {status} {model_id:<30} {metrics_str}")

            if result.failures:
                for failure in result.failures:
                    # Truncate long failure messages
                    failure_short = failure if len(failure) < 80 else failure[:77] + "..."
                    print(f"       └─ {failure_short}")

    print(f"\n{'='*100}\n")


def _format_metrics(metrics):
    """Format metrics dict as a readable string."""
    if not metrics:
        return "No metrics"

    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    return " | ".join(parts)


def _format_metrics_compact(metrics):
    """Format metrics in compact form for table display."""
    if not metrics:
        return ""

    parts = []
    # Show most important metrics first
    if "mean_score" in metrics:
        parts.append(f"score={metrics['mean_score']:.4f}")
    if "range" in metrics:
        parts.append(f"range={metrics['range']:.4f}")

    return " | ".join(parts)
