# API Response Cache

This directory contains cached OpenAI API responses for test fixtures.

## Purpose

- Speeds up test execution by avoiding repeated API calls
- Reduces API costs during development
- Ensures consistent test results
- Version controlled to allow running tests without API keys

## Structure

Each cache file is named using a SHA-256 hash of:
- Model name
- Sorted list of input texts

Cache files are JSON with the format:
```json
{
  "model": "text-embedding-3-small",
  "texts": ["text1", "text2"],
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

## Clearing Cache

To regenerate cached responses (e.g., after changing test data):
```bash
rm tests/.cache/*.json
```

The next test run will make fresh API calls and repopulate the cache.
