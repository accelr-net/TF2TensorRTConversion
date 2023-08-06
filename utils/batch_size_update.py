import os 
import sys

import onnx
import argparse



def main(args):
    
    """
    Main function for modifying the batch size of an ONNX model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """

    onnx_model = onnx.load_model(args.model_file)

    batch_size = args.batch_size
    inputs = onnx_model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = batch_size

    model_name = args.name
    onnx.save_model(onnx_model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_file", help = "The path of onnx model file ")
    parser.add_argument("-n", "--name", help="The name to save the updated onnx model")
    parser.add_argument("-b", "--batch_size", help="The batch size")

    args = parser.parse_args()
    if not all([args.model_file, args.name, args.batch_size]):
        parser.print_help()
        print("\nThese arguments are required: --model_file --name and --batch_size")
        sys.exit(1)
    main(args)
