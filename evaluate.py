import argparse


def main(args):
    # TODO: Perform evaluation.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('predicted_file')

    args = parser.parse_args()

    main(args)
