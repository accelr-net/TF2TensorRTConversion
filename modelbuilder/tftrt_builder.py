import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants


class TFTRTBuilder:

    """
    Class to create a TensorFlow-TensorRT (TF-TRT) optimized model from an existing TensorFlow SavedModel.
    """

    def __init__(self, tf_model_path: str, output_path: str, precision: str, batch_size: int) -> None:
        
        """
        Initialize the TFTRTBuilder object.

        Args:
            tf_model_path (str): The path to the original TensorFlow SavedModel.
            output_path (str): The directory where the optimized TF-TRT model will be saved.
            precision (str): The precision mode for optimization ('FP16' or 'FP32').
            batch_size (int): The batch size for inference.
        """
        self.tf_model_path = tf_model_path
        self.output_path = output_path
        self.precision = precision
        self.batch_size = batch_size

       


    def create_tftrt (self) -> None:
        
        """
        Create the TensorFlow-TensorRT optimized model and save it to the output directory.
        """

        if(self.precision == "FP16"):
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
        else:
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=self.tf_model_path, conversion_params=conversion_params)
        converter.convert()
        converter.save(self.output_path +'/TFTRT_' + self.precision + '/')
