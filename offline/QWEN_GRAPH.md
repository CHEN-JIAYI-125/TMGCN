# Qwen embeddings inside TMGCN

This integrates frozen **per-utterance** Qwen features into the existing graph
model; the separate conversation-classifier baseline is unchanged. Global IDs,
node deduplication, labels, edges, split files and last-sentence evaluation are
preserved. No labels are included in prompts. Qwen is frozen; only the input
projection and graph model are trained. This first experiment isolates the
encoder change; contextual prefix embeddings are a separate future ablation.

Sentence features and the existing four target views use the same Qwen encoder,
last-token pooling, L2 normalization and shared trainable projection to the graph
hidden width. Original 768-dimensional configs retain an identity projection.
The baseline and reply-distillation model paths both support projection.

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-8B

Use a local Qwen embedding model with transformers >= 4.51, torch, torch-geometric,
numpy, pandas, scikit-learn and PyYAML. Model files and feature caches are not Git
artifacts. The extractor refuses a nonempty destination; failed caches must be
kept separate from a fresh rerun. `manifest.json` is written only on completion
and records source hashes, encoder arguments and the number of truncated texts.
Default max length is 512; inspect truncation before interpreting results.

```bash
export PYTHON=/path/to/environment/bin/python
export MODEL=$HOME/models/qwen3-embedding-8b
export GPU=1
export RUN_DIR=$HOME/tmgcn_runs/qwen_nodes_001
mkdir -p "$RUN_DIR"
nohup bash scripts/run_qwen_graph.sh > "$RUN_DIR/pipeline.log" 2>&1 &
tail -f "$RUN_DIR/pipeline.log"
```

The script stops on any failure, first runs mapping and forward/backward tests,
builds a separate cache and a derived YAML, then trains. Generated config keeps
the original optimizer, seed, epochs, grounding and channel switches. Select
checkpoints using dev metrics as in the existing trainer; do not tune on test
F1. Compare against a fresh original-config run on the same split/seed, and
report both dev-selected final test results. Improved F1 is an experimental
outcome, not guaranteed by adding an LLM.
