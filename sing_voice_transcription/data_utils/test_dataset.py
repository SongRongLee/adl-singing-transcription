from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .helper_func import label_to_value
import librosa
import os
import numpy as np
import random


class TestDataset(Dataset):
    """
    A TestDataset contains ALL frames for all songs, which is different from AudioDataset.
    """

    def __init__(self, data_dir, is_test=False):
        self.data_instances = []

        for song_dir in tqdm(sorted(Path(data_dir).iterdir())):
            wav_path = song_dir / 'Vocal.wav'
            song_id = song_dir.stem

            # Load song and extract features
            y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 16000)
            frame512 = np.abs(librosa.core.stft(y, n_fft=512, hop_length=512, center=True))
            frame1024 = np.abs(librosa.core.stft(y, n_fft=1024, hop_length=512, center=True))
            frame2048 = np.abs(librosa.core.stft(y, n_fft=2048, hop_length=512, center=True))
            features = np.concatenate((frame512, frame1024, frame2048), axis=0)

            # For each frame, combine adjacent frames as a data_instance
            feature_size, frame_num = features.shape[0], features.shape[1]
            for frame_idx in range(frame_num):
                concated_feature = torch.empty(feature_size, 7)
                for frame_window_idx in range(frame_idx - 3, frame_idx + 4):
                    # Boundary check
                    if frame_window_idx < 0:
                        choosed_idx = 0
                    elif frame_window_idx >= frame_num:
                        choosed_idx = frame_num - 1
                    else:
                        choosed_idx = frame_window_idx

                    concated_feature[:, frame_window_idx - frame_idx + 3] = torch.tensor(features[:, choosed_idx])

                self.data_instances.append((concated_feature, song_id))

        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
