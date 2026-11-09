# Kotaemon evaluation case study

Based on https://github.com/Cinnamon/kotaemon.
Original source is preserved at the baseline/kotaemon-original tag.

Goal: evaluate document QA correctness, evidence retrieval, grounding,
citation support, and abstention using a reviewed QASPER subset.

Paper content is indexed in Kotaemon. Reference answers and evaluation
annotations are kept outside the application's retrieval corpus.

## Inspecting retrieved evidence

From the repository root with the project virtual environment active, inspect
one golden question before answer generation:

```bash
python -m evals.retriever.inspect_evidence \
  --dataset data/golden/rlpm.json \
  --question-id RLPM-39 \
  --output results/retriever/RLPM-39_evidence.json
```

The JSON dump contains the ranked chunks and the final formatted generation
context, including character/token counts and available source and score
metadata. Capture stops before the answer-generation pipeline is invoked.

## Extracting retrieval claims

Generate a reviewable preview for the RLPM pilot with the project virtual
environment active:

```bash
python -m evals.retriever.extract_claims \
  --dataset data/golden/rlpm.json \
  --question-ids RLPM-39 RLPM-24 \
  --preview-output results/retriever/rlpm_claims_preview.json
```

The script uses `OPENAI_API_KEY` and `OPENAI_API_BASE` from the environment or
root `.env`, with `gpt-4o-mini` as its default model. Review or edit the saved
preview before applying it to the selected dataset entries:

```bash
python -m evals.retriever.extract_claims \
  --dataset data/golden/rlpm.json \
  --question-ids RLPM-39 RLPM-24 \
  --apply-preview results/retriever/rlpm_claims_preview.json
```

Applying does not call the model. If a selected entry already has
`retrieval_claims`, pass `--overwrite` to replace it.

## Judging retrieval evidence

After reviewing the stored claims and capturing evidence dumps for both pilot
questions, run the text-only judge from the repository root:

```bash
python -m evals.retriever.judge_evidence \
  --dataset data/golden/rlpm.json \
  --question-ids RLPM-39 RLPM-24 \
  --evidence results/retriever/RLPM-39_evidence.json \
            results/retriever/RLPM-24_evidence.json \
  --k 5 \
  --report results/retriever/rlpm_judgment.json
```

The judge uses the first `k` mapped chunks per saved rank order, makes one
structured OpenAI call per non-empty claim list, and calculates recall and
precision in Python. It does not rerun or reindex Kotaemon. The RLPM-24 pilot
preflight requires its reviewed claims to contain only the positive
bioprotocol/sequencing-graph statement; corpus-omission claims must not be
included in this retrieval judgment.
