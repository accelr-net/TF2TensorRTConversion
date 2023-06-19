import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import argparse


from modelbuilder.tftrt_builder import TFTRTBuilder


def main (args):
    
    
    
    precision = args.precision

   
        
    tftrt_builder = TFTRTBuilder(args.model_path, args.output, precision, batch_size)
    tftrt_builder.create_tftrt()
        

    	
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", help = "The model path for tensorflow saved model ")
    parser.add_argument("-o", "--output", help="The outpath for model")
    parser.add_argument("-fp","--precision", help=" The presicion for output model ")
    

    args = parser.parse_args()
    if not all([args.model_path, args.output]):
        parser.print_help()
        print("\nThese arguments are required: --model_path and --output ")
        sys.exit(1)
    main(args)
