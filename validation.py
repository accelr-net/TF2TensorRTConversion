import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

from base.benchmark import Benchmark


def main(args):
    """
    Main function for running inference for benchmark on a TensorFlow, TF-TRT or TensorRT model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """

    # Convert batch size to an integer
    batch_size = int(args.batch_size)
    # Default pattern for validation data
    pattern = 'validation*'

    # Create an instance of the Benchmark class and generate inference results
    results = Benchmark(args.model_path, args.input, pattern, batch_size, args.type)
    results.generate_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path",
                        help="The model path for tensorflow saved model or the TensorRT engine to infer with")
    parser.add_argument("-i", "--input",
                        help="The input to infer, either a single image path, or a directory of images or the "
                             "tfrecord path")
    parser.add_argument("-t", "--type", help=" The model type (Tensorflow or TensorRT)")
    parser.add_argument("-b", "--batch_size", help="The batch size")

    args = parser.parse_args()
    if not all([args.model_path, args.input, args.type, args.batch_size]):
        parser.print_help()
        print("\nThese arguments are required: --model_path --input --type and --batch_size")
        sys.exit(1)
    main(args)
