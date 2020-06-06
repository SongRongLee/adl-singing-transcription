from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# from .helper_func import label_to_value
import librosa
import os
import numpy as np
import random


class TestRNNDataset(Dataset):
    """
    A TestDataset contains ALL frames for all songs, which is different from AudioDataset.
    """

    def __init__(self, data_dir, is_test=False, chunk_size=150):
        self.data_instances = []
        self.chunk_size = chunk_size
        sr = 16000

        for song_dir in tqdm(sorted(Path(data_dir).iterdir())):
            wav_path = song_dir / 'Vocal.wav'
            song_id = song_dir.stem

            # Load song and extract features
            y, _ = librosa.core.load(wav_path, sr=sr, mono=True)
            frame512 = np.abs(librosa.core.stft(y, n_fft=512, hop_length=512, center=True))
            frame1024 = np.abs(librosa.core.stft(y, n_fft=1024, hop_length=512, center=True))
            frame2048 = np.abs(librosa.core.stft(y, n_fft=2048, hop_length=512, center=True))
            features = np.concatenate((frame512, frame1024, frame2048), axis=0)

            # For each chunk, combine adjacent frames for each frame in chunk as a data_instance
            # Append the last chunk (< chunk size) with zeros
            feature_size, frame_num = features.shape[0], features.shape[1]
            for start_frame_idx in range(0, frame_num, self.chunk_size):
                chunk_feature = torch.zeros(self.chunk_size, feature_size, 7, dtype=torch.float)
                
                # Compute end frame idx
                if start_frame_idx + self.chunk_size < frame_num:
                    end_frame_idx = start_frame_idx + self.chunk_size
                else:
                    end_frame_idx = frame_num - 1
                
                for frame_chunk_idx in range(start_frame_idx, end_frame_idx):
                    for frame_window_idx in range(frame_chunk_idx - 3, frame_chunk_idx + 4):
                        # Boundary check
                        if frame_window_idx < 0:
                            choosed_idx = 0
                        elif frame_window_idx >= frame_num:
                            choosed_idx = frame_num - 1
                        else:
                            choosed_idx = frame_window_idx

                        chunk_feature[frame_chunk_idx - start_frame_idx, :, frame_window_idx - frame_chunk_idx + 3] = \
                            torch.tensor(features[:, choosed_idx])
                
                self.data_instances.append((chunk_feature, song_id, start_frame_idx))

        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)

if __name__ == "__main__":
    dataset = TestRNNDataset(data_dir="../../../data/ismir2014")
    r = dataset.__getitem__(0)

    print(r[0])
    print(r[1])