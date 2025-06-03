import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

class HAT(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [
            'XYH-6-X', 'XYH-6-Y', 'ios', 'lavalier',
            'XYH-6-X+XYH-6-Y', 'XYH-6-X+ios','XYH-6-X+lavalier',
            'XYH-6-Y+ios', 'XYH-6-Y+lavalier', 'ios+lavalier'
        ]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        waveform, sr = torchaudio.load(filepath)
        if self.transform:
            waveform = self.transform(waveform)
        # print(f'Loaded {filepath} with label {label}')
        return waveform, label

def custom_collate_fn(batch):
    max_length = max([waveform.size(1) for waveform, _ in batch])
    padded_waveforms = []
    labels = []
    for waveform, label in batch:
        if waveform.size(1) < max_length:
            padding = max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(waveform)
        labels.append(label)
    return torch.stack(padded_waveforms), torch.tensor(labels)

class HAT_DM(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = HAT(os.path.join(self.data_dir, 'train'))
        self.val_dataset = HAT(os.path.join(self.data_dir, 'test'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=custom_collate_fn)

if __name__ == '__main__':
    data_dir = '/share/nas169/jethrowang/URSA-GAN/data/HAT'
    batch_size = 32
    num_workers = 10

    tat_dm = HAT_DM(data_dir, batch_size=batch_size, num_workers=num_workers)

    tat_dm.setup()

    train_loader = tat_dm.train_dataloader()
    val_loader = tat_dm.val_dataloader()

    print("Train Dataset:")
    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}: waveforms shape = {waveforms.shape}, labels shape = {labels.shape}")
        if batch_idx == 1:
            break

    print("Validation Dataset:")
    for batch_idx, (waveforms, labels) in enumerate(val_loader):
        print(f"Batch {batch_idx + 1}: waveforms shape = {waveforms.shape}, labels shape = {labels.shape}")
        if batch_idx == 1:
            break
