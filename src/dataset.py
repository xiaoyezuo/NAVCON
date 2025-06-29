import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class NavconDataset(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotation = annotation
        self.max_len = 4
        self.data = []

        with open(annotation, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                video_path = os.path.join(self.root_dir, line[0])
                if not os.path.exists(video_path):
                    continue
                label = int(line[-1])
                self.data.append((video_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        images = []
        for filename in os.listdir(video_path):
            image_path = os.path.join(video_path, filename)
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            image = self.transform(image)
            images.append(image)
        if len(images) > self.max_len:
            indices = torch.linspace(0, len(images)-1, self.max_len, dtype=int)
            images = [images[i] for i in indices]
        images = torch.stack(images, dim=0)
        return images, len(images), label
