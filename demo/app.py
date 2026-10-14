"""Local demo for both models: python demo/app.py, then open http://127.0.0.1:7860

Standard library server only; the models load once at start.
"""
import json
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v2
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PAGE = (Path(__file__).resolve().parent / 'index.html').read_bytes()
SAMPLES = ROOT / 'data' / 'food-101' / 'images'
# sample photos come from Food-101's test list, so the model has never seen them
_firsts = {}
for _line in (ROOT / 'data' / 'food-101' / 'meta' / 'test.txt').read_text().splitlines():
    _folder = _line.split('/')[0]
    if _folder in ('pizza', 'hamburger', 'french_fries', 'fried_rice', 'cheesecake', 'hot_dog', 'chocolate_cake') and len(_firsts.setdefault(_folder, [])) < 2:
        _firsts[_folder].append(f'{_line}.jpg')
SAMPLE_FILES = [f for files in _firsts.values() for f in files]
MAX_UPLOAD = 15 * 1024 * 1024

# Menu Detector
saved = torch.load(ROOT / 'runs' / 'menu_detector' / 'menu_detector.pt', map_location='cpu')
CLASSES = saved['classes']
menu = mobilenet_v2()
menu.classifier[1] = nn.Linear(menu.classifier[1].in_features, len(CLASSES))
menu.load_state_dict(saved['state_dict'])
menu.eval()
to_tensor = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Review sentiment
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'runs' / 'review_sentiment')
review = AutoModelForSequenceClassification.from_pretrained(ROOT / 'runs' / 'review_sentiment').eval()


def classify_photo(data: bytes):
    image = Image.open(io.BytesIO(data)).convert('RGB')
    with torch.no_grad():
        probs = torch.softmax(menu(to_tensor(image).unsqueeze(0)), dim=-1)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda x: -x[1])
    return [{'label': label, 'p': round(p, 4)} for label, p in ranked]


def score_reviews(texts):
    texts = [t.strip() for t in texts if t.strip()][:20]
    if not texts:
        return []
    enc = tokenizer(texts, truncation=True, max_length=64, padding=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.softmax(review(**enc).logits, dim=-1)[:, 1].tolist()
    return [{'text': t, 'p_positive': round(p, 4), 'label': 'positive' if p >= 0.5 else 'negative'} for t, p in zip(texts, probs)]


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, kind='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            return self.send(200, PAGE, 'text/html; charset=utf-8')
        if self.path == '/api/samples':
            return self.send(200, json.dumps([f for f in SAMPLE_FILES if (SAMPLES / f).exists()]).encode())
        if self.path.startswith('/samples/'):
            name = self.path[len('/samples/'):]
            if name in SAMPLE_FILES and (SAMPLES / name).exists():
                return self.send(200, (SAMPLES / name).read_bytes(), 'image/jpeg')
        self.send(404, b'{"error": "not found"}')

    def do_POST(self):
        size = int(self.headers.get('Content-Length') or 0)
        if size <= 0 or size > MAX_UPLOAD:
            return self.send(413, b'{"error": "empty or too large"}')
        body = self.rfile.read(size)
        try:
            if self.path == '/api/menu':
                return self.send(200, json.dumps(classify_photo(body)).encode())
            if self.path == '/api/review':
                return self.send(200, json.dumps(score_reviews(json.loads(body).get('texts', [])), ensure_ascii=False).encode())
        except Exception as error:  # a bad image or bad json is the caller's mistake, not a crash
            return self.send(400, json.dumps({'error': type(error).__name__}).encode())
        self.send(404, b'{"error": "not found"}')

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    print('demo on http://127.0.0.1:7860')
    ThreadingHTTPServer(('127.0.0.1', 7860), Handler).serve_forever()
