import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import argparse


from modelbuilder.tftrt_builder import TFTRTBuilder


def main (args):
    """
    Main function for converting a TensorFlow saved model to a TensorFlow-TensorRT (TF-TRT) optimized model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """   
    precision = args.precision
   
    # Create an instance of TFTRTBuilder and convert the model to TF-TRT  
    tftrt_builder = TFTRTBuilder(args.model_path, args.output, precision, args.batch_size)
    tftrt_builder.create_tftrt()
        

    	
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", help = "The model path for tensorflow saved model ")
    parser.add_argument("-o", "--output", help="The outpath for model")
    parser.add_argument("-fp","--precision", help=" The presicion for output model ")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="The batch size for inference.")

    args = parser.parse_args()
    if not all([args.model_path, args.output]):
        parser.print_help()
        print("\nThese arguments are required: --model_path and --output ")
        sys.exit(1)
    main(args)
