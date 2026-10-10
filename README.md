# memo-sorter

Tags a short note as `idea`, `todo`, `reminder` or `journal`. Part of [tinkeryard.dev](https://tinkeryard.dev).

## Data
- `data/train.csv`: 360 notes generated from templates (`src/make_data.py`).
- `data/test.csv`: 48 hand-written notes with different wording, never seen in training.

## Results
| Model | Test accuracy |
|---|---|
| TF-IDF + logistic regression | 66.7% (32 of 48) |
| MiniLM-L6 fine-tuned, 8 epochs, 36 s on CPU | 83.3% (40 of 48) |

Journal notes: 12 of 12. Most misses are ideas read as todos (5 of 12), since "build a tiny voice assistant" and "fix the fence" are both imperative. The test set is small, so one note moves the score by 2 points.

## Run
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python src/make_data.py && python src/baseline.py && python src/finetune.py
```
