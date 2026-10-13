"""Shopping review sentiment: fine-tune a Korean transformer on Naver shopping reviews.

Data: 200,000 reviews with a star rating (1, 2, 4 or 5). Ratings 4-5 are positive, 1-2 negative.
Usage: python review_sentiment/train.py [--train-size 60000] [--epochs 2]
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data' / 'reviews' / 'naver_shopping.txt'
RESULTS = ROOT / 'results'
LABELS = ['negative', 'positive']


def load_reviews():
    texts, labels = [], []
    for line in DATA.read_text(encoding='utf-8').splitlines():
        rating, _, text = line.partition('\t')
        if not text.strip():
            continue
        texts.append(text.strip())
        labels.append(1 if int(rating) >= 4 else 0)
    return texts, labels


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        return enc['input_ids'][0], enc['attention_mask'][0], torch.tensor(self.labels[idx])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, probs = [], []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        p = torch.softmax(logits, dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p[:, 1].cpu().tolist())
    model.train()
    return preds, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='klue/roberta-small')
    ap.add_argument('--train-size', type=int, default=60000)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-len', type=int, default=64)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-5)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    texts, labels = load_reviews()
    order = list(range(len(texts)))
    random.shuffle(order)
    # 20,000 reviews are held out for the test and never seen in training; 2,000 more track the curve
    test_idx, val_idx, train_idx = order[:20000], order[20000:22000], order[22000:22000 + args.train_size]
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx])
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = pick(train_idx), pick(val_idx), pick(test_idx)
    print(f'reviews: {len(texts)} | train {len(train_x)} | val {len(val_x)} | test {len(test_x)} | positive share {np.mean(labels):.3f}')

    # baseline: character n-gram TF-IDF + logistic regression on the same split
    started = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=3, sublinear_tf=True)
    baseline = LogisticRegression(max_iter=1000, C=4.0)
    baseline.fit(vectorizer.fit_transform(train_x), train_y)
    base_pred = baseline.predict(vectorizer.transform(test_x))
    base = {'accuracy': accuracy_score(test_y, base_pred), 'f1': f1_score(test_y, base_pred), 'seconds': round(time.time() - started, 1)}
    print('baseline:', base)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    train_loader = DataLoader(ReviewDataset(train_x, train_y, tokenizer, args.max_len), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(ReviewDataset(val_x, val_y, tokenizer, args.max_len), batch_size=128)
    test_loader = DataLoader(ReviewDataset(test_x, test_y, tokenizer, args.max_len), batch_size=128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    history = {'step': [], 'train_loss': [], 'val_accuracy': []}
    started, step, running = time.time(), 0, []
    model.train()
    for epoch in range(args.epochs):
        for ids, mask, y in train_loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running.append(out.loss.item()); step += 1
            if step % 250 == 0 or step == total_steps:
                val_pred, _ = evaluate(model, val_loader, device)
                history['step'].append(step); history['train_loss'].append(float(np.mean(running))); history['val_accuracy'].append(accuracy_score(val_y, val_pred))
                print(f'epoch {epoch + 1} step {step}/{total_steps} loss {np.mean(running):.4f} val acc {history["val_accuracy"][-1]:.4f} ({time.time() - started:.0f}s)', flush=True)
                running = []
    train_seconds = round(time.time() - started, 1)

    test_pred, test_prob = evaluate(model, test_loader, device)
    cm = confusion_matrix(test_y, test_pred)
    metrics = {
        'model': args.model, 'parameters': params, 'device': str(device),
        'train_size': len(train_x), 'test_size': len(test_x), 'epochs': args.epochs, 'max_len': args.max_len,
        'batch': args.batch, 'lr': args.lr, 'train_seconds': train_seconds,
        'accuracy': accuracy_score(test_y, test_pred), 'f1': f1_score(test_y, test_pred),
        'confusion_matrix': cm.tolist(), 'baseline_tfidf_logreg': base,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (RESULTS / 'history.json').write_text(json.dumps(history))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # test reviews with the model's answer: three hits and two misses per class, short enough to read
    def take(label, hit, count):
        rows = [i for i in range(len(test_x)) if test_y[i] == label and (test_pred[i] == label) == hit and 18 <= len(test_x[i]) <= 60]
        return rows[:count]
    chosen = take(1, True, 3) + take(0, True, 3) + take(1, False, 2) + take(0, False, 2)
    examples = [{'text': test_x[i], 'label': LABELS[test_y[i]], 'predicted': LABELS[test_pred[i]], 'p_positive': round(test_prob[i], 3)} for i in chosen]
    (RESULTS / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history['step'], history['train_loss'], color='#2563eb'); ax[0].set_title('Training loss'); ax[0].set_xlabel('step')
    ax[1].plot(history['step'], history['val_accuracy'], color='#16a34a'); ax[1].axhline(base['accuracy'], color='#9ca3af', ls='--', label='TF-IDF baseline (test)')
    ax[1].set_title('Validation accuracy'); ax[1].set_xlabel('step'); ax[1].legend()
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(RESULTS / 'curves.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else '#111', fontsize=13)
    ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS); ax.set_xlabel('predicted'); ax.set_ylabel('actual'); ax.set_title('Test set, 20,000 reviews')
    fig.tight_layout(); fig.savefig(RESULTS / 'confusion_matrix.png', dpi=160); plt.close(fig)

    out = ROOT.parent / 'runs' / 'review_sentiment'
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    print('saved to', out)


if __name__ == '__main__':
    main()
