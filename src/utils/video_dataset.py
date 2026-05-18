import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        self.transform = transform

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for video_dir in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_dir)
                self.samples.append((video_path, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_name = self.samples[idx]
        frames = []
        for frame_name in sorted(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame_name)
            frame = Image.open(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Dimension: (C, T, H, W)
        frames = torch.stack(frames, dim=1)

        class_idx = self.classes.index(class_name)
        return frames, class_idx, video_path