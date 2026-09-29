# Lab 6 — Deep Learning for NLP

**Name:** Hadi Muneer Abu Allairat
**Student ID:** 2230005761

## Tasks

1. **Represent the reviews with Word2Vec** (`vector_size=200`, `window=5`, `min_count=1`, `sg=1`)
   instead of TF-IDF, and convert to PyTorch tensors.
2. **Build a deeper network** — input → 128 → 64 → 32 → 1, `BCEWithLogitsLoss`, Adam at `lr=0.001`,
   15 epochs — and report the test accuracy.

## Contents

| File | |
| --- | --- |
| `Lab6_Deep_Learning_for_NLP.ipynb` | The lab notebook, executed with all outputs saved |
| `dataset/07-yelp-dataset.txt` | The dataset, unchanged from the original lab repository |

## Results

| features | hidden layers | epochs | lr | test accuracy |
| --- | --- | --- | --- | --- |
| TF-IDF (1000 sparse) | 64, 32 | 10 | 0.01 | **0.755** |
| Word2Vec avg (200 dense) | 128, 64, 32 | 15 | 0.001 | **0.490** |
| Word2Vec avg (200 dense) | 128, 64, 32 | 500 | 0.001 | **0.675** |

- Task 2 as specified scores **0.490** — no better than guessing. The loss sits at 0.693, which is
  exactly `-ln(0.5)`, and the model predicts one class for everything.
- **Most of that is undertraining, not the representation.** The loop is full-batch, so 15 epochs is
  15 weight updates at `lr=0.001`. Training the identical model for 500 epochs lifts it to **0.675**.
  The TF-IDF baseline only reached 0.755 in 10 epochs because its learning rate was 10× larger —
  equal epochs is not equal training.
- Feature scale was measured and **ruled out** as a cause: mean row norm is 1.00 (TF-IDF) vs 1.04
  (Word2Vec).
- The remaining 0.675 vs 0.755 gap is the representation. 8,000 training tokens is far too little to
  learn useful embeddings, and averaging discards the word order that `not good` depends on.

Two bugs in the lab sheet are corrected in the notebook, both of which fail silently:

- `preprocess_text` has its `df[...] = df[...].apply(...)` lines indented **below the `return`**, so
  preprocessing never runs and the model trains on raw text.
- The training loop omits `optimizer.zero_grad()`, so gradients accumulate across epochs. Worth
  0.755 vs 0.740 here.

## Running it

```bash
pip install pandas numpy scikit-learn gensim torch notebook
jupyter notebook Lab6_Deep_Learning_for_NLP.ipynb
```

The dataset is committed alongside the notebook, so it runs as-is with no download step.
