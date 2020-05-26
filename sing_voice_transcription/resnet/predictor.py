import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
import pickle
from tqdm import tqdm

from net import ResNet
from ..data_utils import value_to_label


class ResNetPredictor:
    def __init__(self, model_path=None):
        """
        Params:
        model_path: Optional pretrained model file
        """
        # Initialize model
        self.model = ResNet().cuda()

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
          - print_every_iter
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
        self.print_every_iter = training_args['print_every_iter']

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Read the datasets
        print('Reading datasets...')
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
        self.iters_per_epoch = len(self.train_loader)
        for epoch in range(1, self.epoch + 1):
            self.model.train()

            # Run iteration
            total_training_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # Parse batch data
                input_tensor = batch[0].cuda()
                gt_tensor = batch[1].cuda()

                # Forward model
                output = self.model(input_tensor)

                loss = self.criterion(output, gt_tensor)
                loss.backward()

                self.optimizer.step()

                total_training_loss += loss.item()

                # Show training message
                if batch_idx % self.print_every_iter == 0 or batch_idx == self.iters_per_epoch-1:
                    print(
                        '| Epoch [{:4d}/{:4d}] Iter[{:4d}/{:4d}] Loss {:.4f} Time {:.1f}'.format(
                            epoch,
                            self.epoch,
                            batch_idx+1,
                            self.iters_per_epoch,
                            loss.item(),
                            time.time()-start_time,
                        ),
                        end='\r',
                    )

                # Free GPU memory
                # torch.cuda.empty_cache()

                if batch_idx >= self.iters_per_epoch-1:
                    break

            # Perform validation
            self.model.eval()
            with torch.no_grad():
                total_valid_loss = 0
                for batch_idx, batch in enumerate(self.valid_loader):
                    # Parse batch data
                    input_tensor = batch[0].cuda()
                    gt_tensor = batch[1].cuda()

                    # Forward model
                    input_tensor, gt_tensor = input_tensor.cuda(), gt_tensor.cuda()
                    output = self.model(input_tensor)

                    loss = self.criterion(output, gt_tensor)
                    total_valid_loss += loss.item()

                    # Free GPU memory
                    # torch.cuda.empty_cache()

            # Save loss list
            training_loss_list.append(total_training_loss/self.iters_per_epoch)
            valid_loss_list.append(total_valid_loss/len(self.valid_loader))

            # Safe model
            save_dict = self.model.state_dict()

            target_model_path = Path(self.model_dir) / 'e_{}'.format(epoch)
            torch.save(save_dict, target_model_path)

            # Epoch done message
            print(
                '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                    epoch,
                    self.epoch,
                    training_loss_list[-1],
                    valid_loss_list[-1],
                    time.time()-start_time,
                ),
            )

        # Save loss to file
        with open('./plotting/data/loss.pkl', 'wb') as f:
            pickle.dump({'train': training_loss_list, 'valid': valid_loss_list}, f)

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def predict(self, test_dataset):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 100
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        results = []
        self.model.eval()
        with torch.no_grad():
            print('Forwarding model...')
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # TODO: Parse model output

        return results
