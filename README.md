# `embedsim`

**Measure semantic similarity and detect outliers in text collections using embeddings.**

`embedsim` is a lightweight Python library that helps you understand how well texts relate to each other. It provides two core functions: pairwise similarity for comparing two texts, and group coherence for analyzing collections.

**Use cases:**
- **Content moderation**: Find off-topic comments or reviews
- **Document clustering**: Identify outliers before grouping
- **Quality assurance**: Verify generated content stays on topic
- **Search relevance**: Score how well results match a query theme
- **Duplicate detection**: Compare documents for similarity

## Installation

For OpenAI models:
```bash
uv add embedsim[openai]
export OPENAI_API_KEY=your-key-here
```

For local models:
```bash
uv add embedsim[sentence-transformers]
```

## Quick Start

### Pairwise Similarity

Compare two texts directly:

```python
import embedsim

# Similar texts
score = embedsim.pairsim(
    "The cat sat on the mat",
    "A feline rested on the rug"
)
print(score)  # 0.89

# Dissimilar texts
score = embedsim.pairsim(
    "The cat sat on the mat",
    "Python is a programming language"
)
print(score)  # 0.21
```

### Group Coherence

Analyze a collection and find outliers:

```python
import embedsim

texts = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning uses neural networks",
    "Pizza is a popular food"  # This doesn't belong
]

scores = embedsim.groupsim(texts)
# [0.76, 0.73, 0.71, 0.28]
#                    ~~~~ Outlier detected!
```

## API Reference

### `pairsim(text_a, text_b, model_id=None, **config) → float`

Compute similarity between two texts.

- Converts both texts to embeddings
- Computes cosine similarity
- Returns a single similarity score (0-1, higher = more similar)

### `groupsim(texts, model_id=None, **config) → list[float]`

Compute coherence scores for a collection of texts.

- Converts all texts to embeddings
- Calculates the centroid (average) of all embeddings
- Measures how close each text is to the centroid
- Returns coherence scores (0-1, higher = more coherent)

This centroid-based approach gives you a score per text showing how well it fits with the group's semantic theme.

## Models

`embedsim` supports both OpenAI's API and local sentence-transformer models.

See [MODELS.md](MODELS.md) for detailed model comparison and selection guide.

**OpenAI (default, requires API key):**
```python
# Best for production - fast, accurate, no model downloads
score = embedsim.pairsim(text_a, text_b)  # uses openai-3-small
scores = embedsim.groupsim(texts, model_id="openai-3-large")
```

**Local models (privacy, offline):**
```python
# Run entirely on your machine
score = embedsim.pairsim(text_a, text_b, model_id="jina-v2-base")
scores = embedsim.groupsim(texts, model_id="all-MiniLM-L6-v2")
```

## Environment Configuration

```bash
# Set default model
export EMBEDSIM_MODEL=jina-v2-base

# Use custom OpenAI key
export EMBEDSIM_OPENAI_API_KEY=sk-...
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests and benchmarks
make test
```

## License

MIT

## Links

- [Model comparison](MODELS.md) - Detailed guide to choosing the right embedding model
