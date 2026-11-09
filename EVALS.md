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
