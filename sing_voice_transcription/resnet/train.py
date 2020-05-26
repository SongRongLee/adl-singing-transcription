import torch
import torch.nn as nn
import argparse
from predictor import ResNetPredictor


def main(args):
    predictor = ResNetPredictor(model_path=args.model_path)
    predictor.fit(
        train_dataset_path=args.training_dataset,
        valid_dataset_path=args.validation_dataset,
        model_dir=args.model_dir,
        batch_size=64,
        valid_batch_size=200,
        epoch=30,
        lr=1e-3,
        print_every_iter=10,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('model_dir')
    parser.add_argument('--model-path')

    args = parser.parse_args()

    main(args)
