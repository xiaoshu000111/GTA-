# DetectRL-ZH data used by this reproduction

Source: <https://github.com/NLP2CT/NLPCC-2025-Task1>

The official task repository publicly provides the training, development, and
test data. This copy is included only to make the Colab feasibility experiment
reproducible. The original dataset authors retain their rights; this repository
does not grant any additional license to the data.

Included splits:

| File | Samples | Intended use |
|---|---:|---|
| `train.json` | 32,400 | Model training |
| `dev.json` | 2,800 | Model selection and tuning only |
| `test_with_label.json` | 11,000 | Final analysis/evaluation only |

Important protocol constraint: the official task documentation states that the
development split must not be used for model training. The labeled test split
must likewise remain isolated from optimization and threshold selection in this
reproduction.

The SHA-256 digests in `SHA256SUMS` were generated from the local copies before
their first commit.
