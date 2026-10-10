"""Fine-tune MiniLM-L6 (22M params) on CPU and compare against the baseline."""
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS, BATCH, LR, MAXLEN = 8, 16, 8e-5, 48

random.seed(7); np.random.seed(7); torch.manual_seed(7)
torch.set_num_threads(8)

train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")
labels = sorted(train.label.unique())
l2i = {l: i for i, l in enumerate(labels)}

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i
)


def encode(texts):
    return tok(list(texts), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


def predict(texts):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            out.append(model(**encode(texts[i:i + 64])).logits.argmax(-1))
    return torch.cat(out).numpy()


y_train = torch.tensor([l2i[l] for l in train.label])
y_test = np.array([l2i[l] for l in test.label])
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    idx = torch.randperm(len(train))
    total = 0.0
    for i in range(0, len(train), BATCH):
        b = idx[i:i + BATCH]
        batch = encode(train.text.iloc[b.numpy()])
        loss = model(**batch, labels=y_train[b]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        total += loss.item() * len(b)
    acc = (predict(test.text.tolist()) == y_test).mean()
    print(f"epoch {ep + 1}/{EPOCHS}  loss {total / len(train):.3f}  test acc {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

pred = predict(test.text.tolist())
print(classification_report(y_test, pred, target_names=labels, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, pred), index=labels, columns=labels))
wrong = test[pred != y_test].assign(predicted=[labels[p] for p in pred[pred != y_test]])
print("\nmisses:\n", wrong.to_string(index=False))

out = ROOT / "models" / "minilm-memo-sorter"
model.save_pretrained(out); tok.save_pretrained(out)
print("saved", out)
