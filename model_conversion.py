import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

from modelbuilder.tftrt_builder import TFTRTBuilder


def main(model_path: str, output_path: str, precision: str, batch_size: int):
    """
    Main function for converting a TensorFlow saved model to a TensorFlow-TensorRT (TF-TRT) optimized model.

    Args:
        model_path (str): The path to the original TensorFlow SavedModel.
        output (str): The directory where the optimized TF-TRT model will be saved.
        precision (str): The precision mode for optimization ('FP16' or 'FP32').
        batch_size (int): The batch size for inference.

    Returns:
        None
    """

    if precision not in ["FP16", "FP32"]:
            raise ValueError("Invalid precision mode. Must be one of: FP16 or FP32")
    
    if not os.path.exists(model_path):
            raise FileNotFoundError("Provided TensorFlow model path does not exist.")

    if not os.path.isdir(output_path):
            os.makedirs(output_path)
    

    try:
        # Create an instance of TFTRTBuilder and convert the model to TF-TRT
        tftrt_builder = TFTRTBuilder()
        tftrt_builder.convert_and_save_model(model_path, output_path, precision, batch_size)
    except Exception as e:
        print("An error occurred while creating the model:", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", help="The model path for tensorflow saved model ")
    parser.add_argument("-o", "--output", help="The out path for model")
    parser.add_argument("-fp", "--precision", help=" The precision for output model ")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="The batch size for inference.")

    args = parser.parse_args()
    if not all([args.model_path, args.output]):
        parser.print_help()
        print("\nThese arguments are required: --model_path and --output ")
        sys.exit(1)
    main(args.model_path, args.output, args.precision, args.batch_size)
