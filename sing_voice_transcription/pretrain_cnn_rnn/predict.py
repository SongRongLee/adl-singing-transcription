import argparse
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from data_utils import TestRNNDataset, TestRNNDfDataset
from predictor import RNNPredictor


def main(args):
    pretrain_cnn_path_list = [args.resnet_model_path, args.effnet_model_path,
                                args.alexnet_model_path]

    # Create predictor
    predictor = RNNPredictor(pretrain_cnn_path_list=pretrain_cnn_path_list, 
                                model_path=args.model_path)

    if args.discard_frame == True:
        # Read from test_dir
        print('Creating testing dataset...')
        test_dataset = TestRNNDfDataset(args.test_dir, chunk_size=150)

        # Feed dataset to the model
        print('Predicting {}...'.format(args.test_dir))
        results = predictor.predict(test_dataset, chunk_size=150)
    else:
        # Read from test_dir
        print('Creating testing dataset...')
        test_dataset = TestRNNDataset(args.test_dir, chunk_size=150)

        # Feed dataset to the model
        print('Predicting {}...'.format(args.test_dir))
        results = predictor.predict(test_dataset, chunk_size=150)

    # Write results to target file
    with open(args.predict_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    print('Prediction done. File writed to: {}'.format(args.predict_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')
    parser.add_argument('--discard-frame', action='store_true', help='Discard remaining frames that can not form a complete chunk.')
    parser.add_argument('--resnet-model-path')
    parser.add_argument('--alexnet-model-path')
    parser.add_argument('--effnet-model-path')

    args = parser.parse_args()

    main(args)
