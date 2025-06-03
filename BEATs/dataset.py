import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

class HAT_ESC(Dataset):
    def __init__(self, data_dir, max_waveform_length=None):
        self.data_dir = data_dir
        self.labels = sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else x)
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        self.audio_files = self.get_audio_files()
        
        # If max_waveform_length is provided, use it; otherwise calculate it.
        self.max_waveform_length = max_waveform_length or self.find_max_waveform_length()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = os.path.basename(os.path.dirname(audio_file))
        label_id = self.label_to_index[label]

        # Load audio and pad it to max_waveform_length
        waveform, _ = torchaudio.load(audio_file)
        if waveform.shape[1] < self.max_waveform_length:
            waveform = F.pad(waveform, (0, self.max_waveform_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_waveform_length]  # Truncate if necessary

        return waveform, label_id

    def get_audio_files(self):
        audio_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def find_max_waveform_length(self):
        max_length = 0
        for audio_file in self.audio_files:
            try:
                waveform, _ = torchaudio.load(audio_file)
                max_length = max(max_length, waveform.shape[1])
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
        return max_length
        