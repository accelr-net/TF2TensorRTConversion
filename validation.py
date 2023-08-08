import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

from base.benchmark import Benchmark


def main(model_path: str, input_path: str, model_type: str, batch_size: int):
    """
    Main function for running inference benchmark on a TensorFlow, TF-TRT, or TensorRT model.

    Args:
        model_path (str): Path to the model (TensorFlow SavedModel or TensorRT engine).
        input_path (str): Path to the input data (single image, directory of images, or tfrecord path).
        model_type (str): Type of the model ('Tensorflow' or 'TensorRT').
        batch_size (int): Batch size for inference.

    Returns:
        None
    """

    # Convert batch size to an integer
    batch_size = int(args.batch_size)
    # Default pattern for validation data
    pattern = 'validation*'

    try:
        # Create an instance of the Benchmark class and generate inference results
        results = Benchmark(model_path, input_path, pattern, batch_size, model_type)
        results.generate_results()
    except Exception as e:
        print("An error occurred while loading the benchmark module:", str(e))


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
    main(args.model_path, args.input, args.type, args.batch_size)
