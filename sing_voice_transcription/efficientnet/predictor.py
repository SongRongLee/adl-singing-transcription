import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # nopep8
from data_utils import value_to_label

from net import EffNet
import math

FRAME_LENGTH = librosa.frames_to_time(1, sr=16000, hop_length=512)


class EffNetPredictor:
    def __init__(self, model_path=None):
        """
        Params:
        model_path: Optional pretrained model file
        """
        # Initialize model
        self.model = EffNet().cuda()

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print('Model read from {}.'.format(model_path))

        print('Predictor initialized.')

    def fit(self, train_dataset_path, valid_dataset_path, model_dir, **training_args):
        """
        train_dataset_path: The path to the training dataset.pkl
        valid_dataset_path: The path to the validation dataset.pkl
        model_dir: The directory to save models for each epoch
        training_args:
          - batch_size
          - valid_batch_size
          - epoch
          - lr
          - save_every_epoch
        """
        # Set paths
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Set training params
        self.batch_size = training_args['batch_size']
        self.valid_batch_size = training_args['valid_batch_size']
        self.epoch = training_args['epoch']
        self.lr = training_args['lr']
        self.save_every_epoch = training_args['save_every_epoch']

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.onset_criterion = nn.BCEWithLogitsLoss()
        self.offset_criterion = nn.BCEWithLogitsLoss()
        self.pitch_criterion = nn.CrossEntropyLoss()

        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))
        with open(self.train_dataset_path, 'rb') as f:
            self.training_dataset = pickle.load(f)
        with open(self.valid_dataset_path, 'rb') as f:
            self.validation_dataset = pickle.load(f)

        # Setup dataloader and initial variables
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        self.valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.valid_batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        start_time = time.time()
        training_loss_list = []
        valid_loss_list = []

        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        for epoch in range(1, self.epoch + 1):
            self.model.train()

            # Run iterations
            total_training_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # Parse batch data
                input_tensor = batch[0].permute(0, 2, 1).unsqueeze(1).cuda()
                osnet_prob, offset_prob, pitch_class = batch[1][:, 0].float().cuda(), batch[1][:, 1].float().cuda(), batch[1][:, 2].cuda()

                # Forward model
                onset_logits, offset_logits, pitch_logits = self.model(input_tensor)

                # Calculate loss
                loss = self.onset_criterion(onset_logits, osnet_prob) \
                    + self.offset_criterion(offset_logits, offset_prob) \
                    + self.pitch_criterion(pitch_logits, pitch_class)

                loss.backward()
                self.optimizer.step()

                total_training_loss += loss.item()

                # Free GPU memory
                # torch.cuda.empty_cache()

            if epoch % self.save_every_epoch == 0:
                # Perform validation
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    for batch_idx, batch in enumerate(self.valid_loader):
                        # Parse batch data
                        input_tensor = batch[0].permute(0, 2, 1).unsqueeze(1).cuda()
                        onset_prob, offset_prob, pitch_class = batch[1][:, 0].float().cuda(), batch[1][:, 1].float().cuda(), batch[1][:, 2].cuda()

                        # Forward model
                        onset_logits, offset_logits, pitch_logits = self.model(input_tensor)

                        # Calculate loss
                        loss = self.onset_criterion(onset_logits, onset_prob) \
                            + self.offset_criterion(offset_logits, offset_prob) \
                            + self.pitch_criterion(pitch_logits, pitch_class)
                        total_valid_loss += loss.item()

                        # Free GPU memory
                        # torch.cuda.empty_cache()

                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_dir) / 'e_{}'.format(epoch)
                torch.save(save_dict, target_model_path)

                # Save loss list
                training_loss_list.append((epoch, total_training_loss/self.iters_per_epoch))
                valid_loss_list.append((epoch, total_valid_loss/len(self.valid_loader)))

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        training_loss_list[-1][1],
                        valid_loss_list[-1][1],
                        time.time()-start_time,
                    )
                )

        # Save loss to file
        with open('./plotting/data/loss.pkl', 'wb') as f:
            pickle.dump({'train': training_loss_list, 'valid': valid_loss_list}, f)

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def _parse_frame_info(self, frame_info):
        """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""
        onset_thres = 0.3
        offset_thres = 0.3

        result = []
        current_onset = None
        pitch_counter = Counter()
        last_onset= 0.0
        for idx, info in enumerate(frame_info):
            current_time = FRAME_LENGTH*idx + FRAME_LENGTH/2

            if info[0] >= onset_thres:  # If is onset
                if current_onset is None:
                    current_onset = current_time
                    last_onset = info[0]
                elif info[0] >= onset_thres:
                    # If current_onset exists, make this onset a offset and the next current_onset
                    if pitch_counter.most_common(1)[0][0] != 49:
                        result.append([current_onset, current_time, pitch_counter.most_common(1)[0][0] + 36])
                    current_onset = current_time
                    last_onset = info[0]
                    pitch_counter.clear()
            elif info[1] >= offset_thres:  # If is offset
                if current_onset is not None:
                    if pitch_counter.most_common(1)[0][0] != 49:
                        result.append([current_onset, current_time, pitch_counter.most_common(1)[0][0] + 36])
                    current_onset = None
                    pitch_counter.clear()

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                pitch_counter[info[2]] += 1

        return result

    def predict(self, test_dataset):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 500
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        results = []
        self.model.eval()
        with torch.no_grad():
            print('Forwarding model...')
            song_frames_table = {}
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].unsqueeze(1).cuda()
                song_ids = batch[1]

                # Forward model
                onset_logits, offset_logits, pitch_logits = self.model(input_tensor)
                onset_probs, offset_probs, pitch_logits = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu(), pitch_logits.cpu()

                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_logits[bid]).item())
                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)

            # Parse frame info into output format for every song
            results = {}
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = self._parse_frame_info(frame_info)

        return results
