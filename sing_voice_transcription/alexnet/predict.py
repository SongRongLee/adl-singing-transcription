import argparse
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from data_utils import TestDataset
from predictor import AlexNetPredictor


def main(args):
    # Create predictor
    predictor = AlexNetPredictor(model_path=args.model_path)

    # Read from test_dir
    print('Creating testing dataset...')
    test_dataset = TestDataset(args.test_dir)

    # Feed dataset to the model
    print('Predicting {}...'.format(args.test_dir))
    results = predictor.predict(test_dataset)

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

    args = parser.parse_args()

    main(args)
