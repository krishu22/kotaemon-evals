# Evaluation operations

Live pipeline and application results include an `operations` object. Saved-answer
evaluators preserve source-run measurements under `operations.originating_run` and
record their own evaluation time and calls separately.

## Timing boundaries

- `application_request` uses a monotonic clock from execution of Kotaemon's
  top-level pipeline stream, after indexing and runner setup, until the final
  answer stream is exhausted. Failed requests retain elapsed time when the
  request had started.
- `pipeline` starts at the first retrieval-stage entry and ends when the final
  answer stream is exhausted. It can be slightly shorter than request latency.
- `application_ttft` starts at that same boundary and ends at the first
  non-empty `chat`-channel answer chunk. `generator_ttft` starts at the answer
  model stream and ends at the first such chunk. Internal reasoning and
  citation-model output are not answer tokens.
- `retrieval`, `query_embedding`, `search`, `reranking`, `evidence_preparation`,
  `answer_generation`, `answer_stream`, and `citation_processing` are component
  timings. Nested or parallel stages can overlap; do not sum them to derive
  request latency.
- `evaluation`, `result_serialization`, `trace_upload_wait`, and
  `langsmith_usage_fetch` are separate overhead fields, not part of request
  latency. Setup indexing is stored once in batch summaries under
  `operations.setup_accounted_once`.

## Model usage and cost

Calls are recorded at locally observed OpenAI SDK leaf endpoints, tagged with a
component, purpose, transport, model, provider-reported usage, retry coverage,
and failure/timeout state. Parent/child LangSmith spans are not added as
additional calls. Streaming usage is recorded when the provider returns usage
chunks. LangSmith leaf usage, when enabled, is kept separately as
`langsmith_trace_usage_non_additive`.

The `application` and `pipeline` component totals are rollups of the leaf
retriever, generator, and application calls. They are non-additive with those
leaf totals.

No token or cost estimates are substituted for missing provider data. Missing
usage and pricing remain `null` and are reflected in coverage fields. Reranking
provider cost is reported separately when exposed; otherwise it is unavailable.
Calculated costs require explicit per-million-token rates:

```sh
export KOTAEMON_EVAL_PRICING_CURRENCY=USD
export KOTAEMON_EVAL_PRICING_JSON='{"gpt-4o-mini":{"input_per_1m":0,"output_per_1m":0},"text-embedding-3-large":{"input_per_1m":0}}'
```

Replace the example rates with the rates applicable to the deployment. The JSON
records the configured rates and source; no default price is assumed.

## Batch and timing checks

```sh
uv run python -m evals.pipeline.run_eval \
  --dataset data/golden/rlpm.json \
  --question-id RLPM-39 \
  --output-dir results/output/operations/live_rlpm39

uv run python -m evals.pipeline.run_eval \
  --dataset data/golden/rlpm.json \
  --question-ids RLPM-01 RLPM-02 \
  --output-dir results/output/operations/rlpm_small_batch

uv run python -m pytest -q tests/test_evals_operations.py
```

The focused test inserts a delay after request completion and verifies that it
increases evaluation duration without changing application request latency.
