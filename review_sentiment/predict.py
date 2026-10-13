"""Score reviews with the fine-tuned model: python review_sentiment/predict.py "배송 빠르고 좋아요" "사이즈가 안 맞아요" """
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = Path(__file__).resolve().parent.parent / 'runs' / 'review_sentiment'


def main():
    texts = sys.argv[1:] or ['배송 빠르고 좋아요', '사이즈가 안 맞아요']
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    enc = tokenizer(texts, truncation=True, max_length=64, padding=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
    for text, p in zip(texts, probs.tolist()):
        print(f'{"positive" if p >= 0.5 else "negative"}  {p:.3f}  {text}')


if __name__ == '__main__':
    main()
