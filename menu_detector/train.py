"""Menu Detector: recognise a dish from a photo with MobileNetV2 transfer learning.

Continues menu_detector_model.ipynb: the same pretrained backbone, loss and optimiser, now trained
to the end and measured on Food-101's own test split.
Usage: python menu_detector/train.py [--head-epochs 2] [--epochs 6]   (--eval-only redraws the figures from the saved weights)
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
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

ROOT = Path(__file__).resolve().parent
FOOD101 = ROOT.parent / 'data' / 'food-101'
RESULTS = ROOT / 'results'

# Food-101 folder -> menu class; two cakes share one label (class consolidation, as in the notebook)
CLASS_MAPPING = {
    'hamburger': 'hamburger',
    'hot_dog': 'hot_dog',
    'pizza': 'pizza',
    'french_fries': 'french_fries',
    'fried_rice': 'fried_rice',
    'chocolate_cake': 'dessert',
    'cheesecake': 'dessert',
}
CLASSES = ['hamburger', 'hot_dog', 'pizza', 'french_fries', 'fried_rice', 'dessert']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def read_split(name):
    """Food-101 ships its own train/test lists: 750 and 250 photos per folder."""
    items = []
    for line in (FOOD101 / 'meta' / f'{name}.txt').read_text().splitlines():
        folder = line.split('/')[0]
        if folder in CLASS_MAPPING:
            items.append((FOOD101 / 'images' / f'{line}.jpg', CLASS_TO_IDX[CLASS_MAPPING[folder]]))
    return items


class FoodDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert('RGB')), label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, confs, labels = [], [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=-1)
        conf, pred = p.max(-1)
        preds.extend(pred.cpu().tolist()); confs.extend(conf.cpu().tolist()); labels.extend(y.tolist())
    model.train()
    return np.array(preds), np.array(confs), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head-epochs', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_items, test_items = read_split('train'), read_split('test')
    random.shuffle(train_items)
    val_items, train_items = train_items[:500], train_items[500:]
    print(f'train {len(train_items)} | val {len(val_items)} | test {len(test_items)} | classes {CLASSES}')
    train_loader = DataLoader(FoodDataset(train_items, train_tf), batch_size=args.batch, shuffle=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(FoodDataset(val_items, test_tf), batch_size=64, num_workers=4)
    test_loader = DataLoader(FoodDataset(test_items, test_tf), batch_size=64, num_workers=4)

    model = mobilenet_v2(weights='IMAGENET1K_V1')  # pretrained on ImageNet, 1000 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    history = {'epoch': [], 'phase': [], 'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    started = time.time()

    def run(epochs, phase, optimizer):
        for _ in range(epochs):
            losses, hits, seen = [], 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item()); hits += (out.argmax(-1) == y).sum().item(); seen += len(y)
            preds, _, labels = evaluate(model, val_loader, device)
            history['epoch'].append(len(history['epoch']) + 1); history['phase'].append(phase)
            history['train_loss'].append(float(np.mean(losses))); history['train_accuracy'].append(hits / seen)
            history['val_accuracy'].append(float((preds == labels).mean()))
            print(f'epoch {history["epoch"][-1]} [{phase}] loss {np.mean(losses):.4f} train acc {hits / seen:.4f} val acc {history["val_accuracy"][-1]:.4f} ({time.time() - started:.0f}s)', flush=True)

    saved = ROOT.parent / 'runs' / 'menu_detector' / 'menu_detector.pt'
    if args.eval_only:
        model.load_state_dict(torch.load(saved, map_location=device)['state_dict'])
        history = json.loads((RESULTS / 'history.json').read_text())
        train_seconds = json.loads((RESULTS / 'metrics.json').read_text())['train_seconds']
    else:
        # 1) the backbone stays frozen while the new head learns the six classes
        for p in model.features.parameters():
            p.requires_grad = False
        run(args.head_epochs, 'head only', torch.optim.Adam(model.classifier.parameters(), lr=1e-3))
        # 2) then the whole network is fine-tuned at a tenth of the rate
        for p in model.features.parameters():
            p.requires_grad = True
        run(args.epochs, 'fine-tune', torch.optim.Adam(model.parameters(), lr=1e-4))
        train_seconds = round(time.time() - started, 1)

    preds, confs, labels = evaluate(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    per_class = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}
    metrics = {
        'model': 'MobileNetV2 (ImageNet weights), new 6-class head', 'parameters': params, 'device': str(device),
        'train_size': len(train_items), 'val_size': len(val_items), 'test_size': len(test_items),
        'head_epochs': args.head_epochs, 'fine_tune_epochs': args.epochs, 'batch': args.batch, 'train_seconds': train_seconds,
        'test_accuracy': float((preds == labels).mean()), 'per_class_accuracy': per_class, 'confusion_matrix': cm.tolist(),
        'head_only_val_accuracy': history['val_accuracy'][args.head_epochs - 1],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (RESULTS / 'history.json').write_text(json.dumps(history))
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history['epoch'], history['train_loss'], marker='o', color='#2563eb'); ax[0].set_title('Training loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(history['epoch'], history['train_accuracy'], marker='o', color='#9ca3af', label='train')
    ax[1].plot(history['epoch'], history['val_accuracy'], marker='o', color='#16a34a', label='validation')
    ax[1].set_title('Accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
    for a in ax:
        a.axvline(args.head_epochs + 0.5, color='#f59e0b', ls='--', lw=1); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(RESULTS / 'curves.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(cm, cmap='Blues')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else '#111', fontsize=11)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha='right'); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('predicted'); ax.set_ylabel('actual'); ax.set_title(f'Test set, {len(test_items):,} photos')
    fig.tight_layout(); fig.savefig(RESULTS / 'confusion_matrix.png', dpi=160); plt.close(fig)

    # twelve test photos with the model's answer: ten taken at random, plus two it got wrong
    rng = random.Random(7)
    wrong = [i for i in range(len(preds)) if preds[i] != labels[i]]
    chosen = rng.sample(range(len(preds)), 10) + rng.sample(wrong, min(2, len(wrong)))
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.2))
    for a, i in zip(axes.flat, chosen):
        a.imshow(Image.open(test_items[i][0]).convert('RGB').resize((256, 256))); a.axis('off')
        ok = preds[i] == labels[i]
        a.set_title(f'{CLASSES[preds[i]]} {confs[i] * 100:.0f}%' + ('' if ok else f'\n(is {CLASSES[labels[i]]})'), color='#15803d' if ok else '#b91c1c', fontsize=11)
    fig.tight_layout(h_pad=2.2); fig.savefig(RESULTS / 'predictions.png', dpi=150); plt.close(fig)

    if not args.eval_only:
        saved.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'classes': CLASSES}, saved)
        print('saved to', saved.parent)

if __name__ == '__main__':
    main()
