from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random


def preprocess(gt_data, frame_num):
    """
    Label for each frame depends on <song_id>_grountruth.txt

    1st dimension denotes onset: 1 if True, otherwise 0
    2nd dimension denotes offset: 1 if True, otherwise 0
    3rd dimension denotes pitch class: 0~48 and 49 for no picth
    """

    frame_gt_list = []

    note_num = len(gt_data)
    cur_note_idx = 0
    cur_note_onset, cur_note_offset, cur_note_pitch = gt_data[cur_note_idx]

    for i in range(frame_num):
        cur_frame_time = i * 0.032 + 0.016

        if abs(cur_frame_time - cur_note_onset) < 0.0161:
            # First dim : onset
            # Second dim : If the pitch is zero (No pitch) (After offset)
            if i == 0 or frame_gt_list[-1][0] != 1:
                label = [1, 0, int(max(0, min(cur_note_pitch - 36, 48)))]
            else:
                label = [0, 0, int(max(0, min(cur_note_pitch - 36, 48)))]
            
        elif cur_frame_time < cur_note_onset or cur_note_idx >= note_num:
            # For the frame that doesn't belong to any note
            label = [0, 0, 49]

        elif abs(cur_frame_time - cur_note_offset) < 0.0161:
            # For the offset frame
            label = [0, 1, int(max(0, min(cur_note_pitch - 36, 48)))]

            cur_note_idx += 1
            if cur_note_idx < note_num:
                cur_note_onset, cur_note_offset, cur_note_pitch = gt_data[cur_note_idx]

                if abs(cur_frame_time - cur_note_onset) < 0.0161:
                    label[0] = 1

        else:
            # For the voiced frame
            label = [0, 0, int(max(0, min(cur_note_pitch - 36, 48)))]

        frame_gt_list.append(np.array(label))

    return torch.tensor(frame_gt_list, dtype=torch.long)


class AudioRNNDataset(Dataset):
    """
    A AudioRNNDataset returns 150 frames around 5 sec for all songs, which is different from AudioDataset.
    """
    def __init__(self, data_dir, is_test=False, chunk_size=150):
        self.data_instances = []
        self.chunk_size = chunk_size
        sr = 16000

        for song_dir in tqdm(sorted(Path(data_dir).iterdir())):
            song_id = song_dir.stem
            wav_path = song_dir / "Vocal.wav"
            gt_path = song_dir / f"{song_id}_groundtruth.txt"

            # Load song and extract features
            y, _ = librosa.core.load(wav_path, sr=sr, mono=True)
            frame512 = np.abs(librosa.core.stft(y, n_fft=512, hop_length=512, center=True))
            frame1024 = np.abs(librosa.core.stft(y, n_fft=1024, hop_length=512, center=True))
            frame2048 = np.abs(librosa.core.stft(y, n_fft=2048, hop_length=512, center=True))
            features = np.concatenate((frame512, frame1024, frame2048), axis=0)
            
            feature_size, frame_num = features.shape[0], features.shape[1]

            # Load ground truth and create label for each frame
            gt_data = np.loadtxt(gt_path)
            frame_gt_data = preprocess(gt_data, frame_num)
            
            self.data_instances.append((features, frame_gt_data))

        print("Dataset initialized from {}.".format(data_dir))

    def __getitem__(self, idx):
        feature_size, frame_num = len(self.data_instances[idx][0]), len(self.data_instances[idx][1])

        # Random the start frame
        start_frame_idx = int(random.random() * (frame_num - self.chunk_size - 1))

        # Get T to T+chunk_size frames and extract t-3 to t+3 for each frame t
        chunk_feature = torch.empty(self.chunk_size, feature_size, 7, dtype=torch.float)
        
        for frame_chunk_idx in range(start_frame_idx, start_frame_idx+self.chunk_size):
            for frame_window_idx in range(frame_chunk_idx - 3, frame_chunk_idx + 4):
                # Boundary check
                if frame_window_idx < 0:
                    choosed_idx = 0
                elif frame_window_idx >= frame_num:
                    choosed_idx = frame_num - 1
                else:
                    choosed_idx = frame_window_idx

                chunk_feature[frame_chunk_idx - start_frame_idx, :, frame_window_idx - frame_chunk_idx + 3] = \
                    torch.tensor(self.data_instances[idx][0][:, choosed_idx])

        # Concat T to T+window_width frame label
        frame_labels = tuple(self.data_instances[idx][1][frame_idx] 
                            for frame_idx in range(start_frame_idx, start_frame_idx+self.chunk_size))
        label = torch.stack(frame_labels)

        return (chunk_feature, label)

    def __len__(self):
        return len(self.data_instances)

if __name__ == "__main__":
    dataset = AudioRNNDataset(data_dir="../../../data/MIRST500_wav/valid")
    r = dataset.__getitem__(0)
    
    print(r[0].shape)
    print(r[1].shape)

        # print(r[0].shape)
        # [150, 1795, 7]
        # print(r[1].shape)
        # [150, 3]
