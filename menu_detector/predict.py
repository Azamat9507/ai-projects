"""Name the dish in a photo: python menu_detector/predict.py photo.jpg [more.jpg ...]"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v2

WEIGHTS = Path(__file__).resolve().parent.parent / 'runs' / 'menu_detector' / 'menu_detector.pt'


def main():
    saved = torch.load(WEIGHTS, map_location='cpu')
    classes = saved['classes']
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(saved['state_dict'])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for path in sys.argv[1:]:
        with torch.no_grad():
            probs = torch.softmax(model(transform(Image.open(path).convert('RGB')).unsqueeze(0)), dim=-1)[0]
        conf, idx = probs.max(0)
        print(f'{classes[idx]}  {conf:.3f}  {path}')


if __name__ == '__main__':
    main()
