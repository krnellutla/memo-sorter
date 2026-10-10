# memo-sorter

Tags a short note as `idea`, `todo`, `reminder` or `journal`. Part of [tinkeryard.dev](https://tinkeryard.dev).

## Data
- `data/train.csv`: 360 notes generated from templates (`src/make_data.py`).
- `data/test.csv`: 48 hand-written notes with different wording, never seen in training.

## Results
| Model | Test accuracy |
|---|---|
| TF-IDF + logistic regression | 66.7% (32 of 48) |

## Run
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python src/make_data.py && python src/baseline.py
```
