import argparse
import pickle
import json
from pathlib import Path
from sing_voice_transcription.data_utils import AudioDataset


def main(args):
    # Create dataset instances
    print('Generating dataset...')
    print('Using directory: {}'.format(args.data_dir))
    dataset = AudioDataset(args.data_dir)

    # Write the datasets into binary files
    filename_trail = '_dataset.pkl'
    target_path = Path(args.output_dir) / (Path(args.data_dir).stem + filename_trail)
    with open(target_path, 'wb') as f:
        pickle.dump(dataset, f)
    print('Dataset generated at {}.'.format(target_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script will read from a data directory and generate custom dataset class instance into a binary file.")
    parser.add_argument('data_dir')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    main(args)
