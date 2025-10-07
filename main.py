import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR(Dataset):
    def __init__(
        self, 
        dir: Path = Path('data/cifar10/CIFAR-10-C'),
        corruption: str = "brightness",
    ):
        images = np.load(dir / (corruption + '.npy'))
        images = images.astype(np.float32).transpose(0, 3, 1, 2) / 255   # (channels, x, y)
        self.images = torch.from_numpy(images)
        self.labels = np.load(dir / 'labels.npy')
        self.batch_size = len(self.labels)
        self.num_batches = 1

    def load(self, batch_size: int = None):
        if not batch_size: return images, labels

        self.batch_size = batch_size
        self.num_batches = len(self.labels) // self.batch_size
        images = self.images.reshape(self.num_batches, self.batch_size, *self.images.shape[1:])
        labels = torch.from_numpy(self.labels).reshape(self.num_batches, self.batch_size)

        for i in range(self.num_batches):
            yield images[i], labels[i]

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def eval(model, dataset):
    data = dataset.load(batch_size=16)

    with torch.no_grad():
        correct = 0
        start_time = time.time()
        for images, labels in data:
            preds = model(images.to(device)).argmax(1)
            correct += (preds == labels.to(device)).float().sum()
        time_taken = time.time() - start_time
        acc = correct / len(dataset)
    return acc, time_taken

def main():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    dataset = CIFAR(dir=Path('data/cifar10/CIFAR-10-C'))
    acc, time_taken = eval(model, dataset)
    print(f'Accuracy: {acc}, Time taken: {time_taken}')
    print(f'Dataset length: {len(dataset)}')


if __name__ == "__main__":
    main()
