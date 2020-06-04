import torch
import torch.nn as nn
import argparse
from predictor import RNNPredictor


def main(args):
    pretrain_cnn_path_list = [args.resnet_model_path, args.effnet_model_path,
                                args.alexnet_model_path]
    
    predictor = RNNPredictor(pretrain_cnn_path_list=pretrain_cnn_path_list,
                                model_path=args.model_path)
    predictor.fit(
        train_dataset_path=args.training_dataset,
        valid_dataset_path=args.validation_dataset,
        model_dir=args.model_dir,
        batch_size=50,
        valid_batch_size=50,
        epoch=500,
        lr=1e-3,
        save_every_epoch=10,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('model_dir')
    parser.add_argument('--resnet-model-path')
    parser.add_argument('--alexnet-model-path')
    parser.add_argument('--effnet-model-path')
    parser.add_argument('--model-path')
    
    args = parser.parse_args()

    main(args)