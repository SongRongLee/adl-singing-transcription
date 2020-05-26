import argparse

from ..data_utils import AudioDataset
from predictor import ResNetPredictor


def main(args):
    # Create predictor
    predictor = ResNetPredictor(model_path=args.model_path)

    # Read from test_dir
    print('Creating testing dataset...')
    test_dataset = AudioDataset(args.test_dir, is_test=True)

    # Feed dataset to the model
    print('Predicting {}...'.format(args.test_dir))
    results = predictor.predict(test_dataset)

    # Write output_results to target file
    with open(args.predict_file, 'w') as f:
        # TODO: Write predicted data

    print('Prediction done. File writed to: {}'.format(args.predict_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')

    args = parser.parse_args()

    main(args)
