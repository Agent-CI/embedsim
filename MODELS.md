# Embedding Models for `embedsim`

`embedsim` supports both local sentence-transformer models and OpenAI's embedding API. Below are the currently configured models and their specifications.

## Benchmark Performance Summary

Based on comprehensive testing across coherent texts, mixed topics, outlier detection, and text length scaling:

| Model | Tests Passed | Avg Score | Avg Range | Best For |
|-------|--------------|-----------|-----------|----------|
| **e5-large** | 7/11 | 0.4983 | 0.0114 | **Highest coherence, lowest variance** |
| **bge-large** | 8/11 | 0.4455 | 0.0301 | **Best overall pass rate and consistency** |
| **openai-3-small** | 11/11 | 0.3489 | 0.0512 | **Perfect pass rate, distinguishing mixed topics** |
| **openai-3-large** | 7/11 | 0.3507 | 0.0439 | **High accuracy, good topic distinction** |
| **jina-v2-base** | 3/11 | 0.3750 | 0.0729 | **Long documents, struggles with coherence** |
| **all-MiniLM-L6-v2** | 5/11 | 0.3413 | 0.0724 | **Fast development, high variance** |

**Key Findings:**
- **e5-large** and **bge-large** excel at maintaining high coherence scores with minimal variance
- **openai-3-small** achieves perfect pass rate and is best at distinguishing mixed/unrelated topics
- **jina-v2-base** shows lower coherence scores despite long context window capabilities
- **all-MiniLM-L6-v2** useful for rapid prototyping but less reliable for production


## OpenAI Models (API-based)

### openai-3-small (text-embedding-3-small)

Excels at distinguishing between mixed and unrelated topics, making it ideal when you need to differentiate diverse content. With cost-effective pricing at $0.02 per million tokens and excellent reliability across test scenarios, it's the best choice for production use where topic discrimination matters more than raw similarity scores. Choose this when you need API-based processing with cost efficiency.

- Model ID: `openai-3-small`
- Embedding dim: 1,536
- Max tokens: 8,191
- Cost: $0.02 per million tokens

### openai-3-large (text-embedding-3-large)

OpenAI's highest accuracy embedding model delivers strong topic distinction with larger embedding dimensions (3,072). Significantly outperforms ada-002 with 54.9% on MIRACL (vs 31.4%) and 64.6% on MTEB (vs 61.0%). Best for critical applications where accuracy justifies the premium cost of $0.13 per million tokens - about 6.5x more expensive than openai-3-small.

- Model ID: `openai-3-large`
- Embedding dim: 3,072
- Max tokens: 8,191
- Cost: $0.13 per million tokens

**Note:** Both OpenAI models support dimension flexibility - you can trade-off performance and cost by shortening embeddings without losing concept-representing properties.

## Sentence-Transformer Models (Local)

### e5-large (intfloat/e5-large-v2)

Achieves the highest coherence scores (avg 0.4983) with the lowest variance (0.0114) across all models, making it exceptional for detecting similarity in coherent texts. The most consistent local model for applications where texts should score high similarity. Choose this when you need maximum coherence detection and consistency in similarity scores, with the tradeoff that it may not distinguish mixed topics as effectively.

- Model ID: `e5-large`
- Embedding dim: 1,024
- Max tokens: 1,024

### bge-large (BAAI/bge-large-en-v1.5)

Most reliable local model across diverse scenarios, balancing strong coherence scores (avg 0.4455) with reasonable variance (0.0301). Excellent for retrieval tasks and general-purpose use when you need a well-tested, proven baseline. The best overall local option when you want consistent performance across different types of content without the tradeoffs of more specialized models.

- Model ID: `bge-large`
- Embedding dim: 1,024
- Max tokens: 1,024

### all-MiniLM-L6-v2

Lightweight and fast model ideal for rapid prototyping and development. With only 384 dimensions, it runs quickly for testing and iteration, though with higher variance (0.0724) and limited context window of 256 tokens (≈1000 characters). Choose this for fast local inference during development when you can tolerate less reliable results and shorter context limits.

- Model ID: `all-MiniLM-L6-v2`
- Embedding dim: 384
- Max tokens: 256

### jina-v2-base (jinaai/jina-embeddings-v2-base-en)

Offers the longest context window for local models at 8,192 tokens, purpose-built for chunked long documents. However, it struggles with coherence thresholds (avg score 0.3750) and outlier detection in benchmarks. Choose this when you need very long context windows for document processing and coherence thresholds are not critical to your application, or when local processing is required for privacy.

- Model ID: `jina-v2-base`
- Embedding dim: 768
- Max tokens: 8,192

## Detailed Benchmark Results

```
====================================================================================================
MODEL PERFORMANCE BENCHMARK
====================================================================================================

PERFORMANCE SUMMARY
----------------------------------------------------------------------------------------------------
Model                               | Tests Passed    | Avg Range    | Avg Score
----------------------------------------------------------------------------------------------------
all-MiniLM-L6-v2                    | 5/11            | 0.0724       | 0.3413
bge-large                           | 8/11            | 0.0301       | 0.4455
e5-large                            | 7/11            | 0.0114       | 0.4983
jina-v2-base                        | 3/11            | 0.0729       | 0.3750
openai-3-large                      | 7/11            | 0.0439       | 0.3507
openai-3-small                      | 11/11           | 0.0512       | 0.3489

====================================================================================================
DETAILED RESULTS BY TEST
====================================================================================================


test_large_coherent
----------------------------------------------------------------------------------------------------
  ✓ PASS e5-large                       score=0.9223 | range=0.0217
  ✓ PASS bge-large                      score=0.8528 | range=0.0659
  ✓ PASS openai-3-small                 score=0.7257 | range=0.0509
  ✗ FAIL all-MiniLM-L6-v2               score=0.7139 | range=0.1762
       └─ Low coherence: min=0.5953
       └─ High variance: range=0.1762
  ✗ FAIL jina-v2-base                   score=0.7023 | range=0.1143
       └─ Low coherence: min=0.6395
       └─ High variance: range=0.1143
  ✗ FAIL openai-3-large                 score=0.6923 | range=0.0766
       └─ Low coherence: min=0.6480

test_large_mixed
----------------------------------------------------------------------------------------------------
  ✗ FAIL e5-large                       score=0.8876 | range=0.0301
       └─ Low variance: range=0.0301
  ✗ FAIL bge-large                      score=0.7619 | range=0.0642
       └─ Low variance: range=0.0642
  ✓ PASS jina-v2-base                   score=0.6950 | range=0.1220
  ✗ FAIL openai-3-large                 score=0.5570 | range=0.0893
       └─ Low variance: range=0.0893
  ✓ PASS openai-3-small                 score=0.5515 | range=0.1087
  ✓ PASS all-MiniLM-L6-v2               score=0.4883 | range=0.1799

test_large_outlier
----------------------------------------------------------------------------------------------------
  ✓ PASS all-MiniLM-L6-v2
  ✗ FAIL jina-v2-base
       └─ Outlier not detected: 0.7086 vs threshold 0.6349
       └─ Low ML coherence: min=0.6256
  ✓ PASS bge-large
  ✓ PASS e5-large
  ✓ PASS openai-3-small
  ✓ PASS openai-3-large

test_medium_coherent
----------------------------------------------------------------------------------------------------
  ✓ PASS e5-large                       score=0.9252 | range=0.0156
  ✓ PASS bge-large                      score=0.8653 | range=0.0406
  ✓ PASS openai-3-large                 score=0.7318 | range=0.0596
  ✓ PASS openai-3-small                 score=0.7132 | range=0.0817
  ✗ FAIL jina-v2-base                   score=0.7015 | range=0.1146
       └─ Low coherence: min=0.6391
       └─ High variance: range=0.1146
  ✗ FAIL all-MiniLM-L6-v2               score=0.6923 | range=0.1077
       └─ High variance: range=0.1077

test_medium_mixed
----------------------------------------------------------------------------------------------------
  ✗ FAIL e5-large                       score=0.9051 | range=0.0210
       └─ Low variance: range=0.0210
  ✗ FAIL bge-large                      score=0.7779 | range=0.0450
       └─ Low variance: range=0.0450
  ✓ PASS jina-v2-base                   score=0.7046 | range=0.1937
  ✗ FAIL openai-3-large                 score=0.5739 | range=0.0640
       └─ Low variance: range=0.0640
  ✓ PASS openai-3-small                 score=0.5497 | range=0.1124
  ✗ FAIL all-MiniLM-L6-v2               score=0.5378 | range=0.0913
       └─ Low variance: range=0.0913

test_medium_outlier
----------------------------------------------------------------------------------------------------
  ✗ FAIL all-MiniLM-L6-v2
       └─ Low ML coherence: min=0.6090
  ✗ FAIL jina-v2-base
       └─ Outlier not detected: 0.6106 vs threshold 0.5917
       └─ Low ML coherence: min=0.6195
  ✓ PASS bge-large
  ✓ PASS e5-large
  ✓ PASS openai-3-small
  ✓ PASS openai-3-large

test_pairwise_long_texts
----------------------------------------------------------------------------------------------------
  ✓ PASS all-MiniLM-L6-v2
  ✓ PASS jina-v2-base
  ✓ PASS bge-large
  ✗ FAIL e5-large
       └─ Low domain distinction: diff=0.0578
  ✓ PASS openai-3-small
  ✓ PASS openai-3-large

test_small_coherent
----------------------------------------------------------------------------------------------------
  ✓ PASS e5-large                       score=0.9376 | range=0.0120
  ✓ PASS bge-large                      score=0.8754 | range=0.0454
  ✗ FAIL all-MiniLM-L6-v2               score=0.7754 | range=0.1077
       └─ High variance: range=0.1077
  ✓ PASS openai-3-large                 score=0.7415 | range=0.0846
  ✓ PASS openai-3-small                 score=0.7294 | range=0.0772
  ✗ FAIL jina-v2-base                   score=0.6664 | range=0.1958
       └─ Low coherence: min=0.5594
       └─ High variance: range=0.1958

test_small_mixed
----------------------------------------------------------------------------------------------------
  ✗ FAIL e5-large                       score=0.9033 | range=0.0252
       └─ Low variance: range=0.0252
  ✗ FAIL bge-large                      score=0.7672 | range=0.0704
       └─ Low variance: range=0.0704
  ✗ FAIL jina-v2-base                   score=0.6551 | range=0.0618
       └─ Low variance: range=0.0618
  ✓ PASS openai-3-small                 score=0.5682 | range=0.1318
  ✓ PASS openai-3-large                 score=0.5612 | range=0.1093
  ✓ PASS all-MiniLM-L6-v2               score=0.5469 | range=0.1340

test_small_outlier
----------------------------------------------------------------------------------------------------
  ✓ PASS all-MiniLM-L6-v2
  ✗ FAIL jina-v2-base
       └─ Outlier not detected: 0.6467 vs threshold 0.5726
       └─ Low ML coherence: min=0.5869
  ✓ PASS bge-large
  ✓ PASS e5-large
  ✓ PASS openai-3-small
  ✓ PASS openai-3-large

test_text_length_scaling
----------------------------------------------------------------------------------------------------
  ✗ FAIL all-MiniLM-L6-v2
       └─ Low large coherence: min=0.5953
  ✗ FAIL jina-v2-base
       └─ Low small coherence: min=0.5594
       └─ Low medium coherence: min=0.6391
       └─ Low large coherence: min=0.6395
  ✓ PASS bge-large
  ✓ PASS e5-large
  ✓ PASS openai-3-small
  ✗ FAIL openai-3-large
       └─ Low large coherence: min=0.6480

====================================================================================================
```
