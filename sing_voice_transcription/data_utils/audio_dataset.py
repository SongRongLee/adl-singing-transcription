from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .helper_func import label_to_value
import librosa
import os
import numpy as np
import random


def preprocess(gt_data, length):

    new_label = []

    cur_note = 0
    cur_note_onset = gt_data[cur_note][0]
    cur_note_offset = gt_data[cur_note][1]
    cur_note_pitch = gt_data[cur_note][2]

    for i in range(length):
        cur_time = i * 0.032 + 0.016

        if abs(cur_time - cur_note_onset) < 0.0161:
            # First dim : onset
            # Second dim : If the pitch is zero (No pitch) (After offset)
            if i == 0 or new_label[-1][0] != 1:
                label = [1, 0, int(max(0, min(cur_note_pitch - 36, 48)))]
                new_label.append(np.array(label))
            else:
                label = [0, 0, int(max(0, min(cur_note_pitch - 36, 48)))]
                new_label.append(np.array(label))

        elif cur_time < cur_note_onset or cur_note >= len(gt_data):
            # For the frame that doesn't belong to any note
            label = [0, 1, 49]
            new_label.append(np.array(label))

        elif abs(cur_time - cur_note_offset) < 0.0161:
            # For the offset frame
            label = [0, 1, int(max(0, min(cur_note_pitch - 36, 48)))]

            cur_note = cur_note + 1
            if cur_note < len(gt_data):
                cur_note_onset = gt_data[cur_note][0]
                cur_note_offset = gt_data[cur_note][1]
                cur_note_pitch = gt_data[cur_note][2]
                if abs(cur_time - cur_note_onset) < 0.0161:
                    label[0] = 1

            new_label.append(np.array(label))

        else:
            # For the voiced frame
            label = [0, 0, int(max(0, min(cur_note_pitch - 36, 48)))]
            new_label.append(np.array(label))

    return np.array(new_label)


class AudioDataset(Dataset):
    def __init__(self, data_dir, is_test=False):
        self.data_instances = []
        count = 0
        for the_dir in os.listdir(data_dir):
            wav_path = data_dir + "/" + the_dir + "/Vocal.wav"
            gt_path = data_dir + "/" + the_dir + "/" + the_dir + "_groundtruth.txt"
            sr = 16000
            y, _ = librosa.core.load(wav_path, sr=sr, mono=True)
            frame512 = np.abs(librosa.core.stft(y, n_fft=512, hop_length=512, center=True))
            frame1024 = np.abs(librosa.core.stft(y, n_fft=1024, hop_length=512, center=True))
            frame2048 = np.abs(librosa.core.stft(y, n_fft=2048, hop_length=512, center=True))
            data = np.concatenate((frame512.T, frame1024.T, frame2048.T), axis=1)

            gt_data = np.loadtxt(gt_path)
            answer_data = preprocess(gt_data, data.shape[0])
            #print (data.shape)
            print(answer_data.shape)
            #print (answer_data[1000])
            self.data_instances.append((data, answer_data))
            count = count + 1
            print("%d songs processed" % (count))

        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        rand_num = int(random.random() * (len(self.data_instances[idx][0]) - 1))
        # get (rand_num-3 to rand_num+ 3) frames
        feature = torch.empty(7, len(self.data_instances[idx][0][0]), dtype=torch.float)

        for i in range(rand_num - 3, rand_num + 4):
            if i < 0:
                num = 0
            elif i > len(self.data_instances[idx][0]) - 1:
                num = len(self.data_instances[idx][0]) - 1
            else:
                num = i
            feature[i - rand_num + 3, :] = torch.tensor(self.data_instances[idx][0][num])

        return (feature, torch.tensor(self.data_instances[idx][1][rand_num], dtype=torch.long))

    def __len__(self):
        return len(self.data_instances)
