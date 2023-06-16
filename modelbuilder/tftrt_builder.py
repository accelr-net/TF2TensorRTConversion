import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants


class TFTRTBuilder:
    def __init__(self, tf_model_path: str, output_path: str, precision: str, batch_size: int) -> None:
        
        self.tf_model_path = tf_model_path
        self.output_path = output_path
        self.precision = precision
        self.batch_size = batch_size

       


    def create_tftrt (self):

        if(self.precision == "FP16"):
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
        else:
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=self.tf_model_path, conversion_params=conversion_params)
        converter.convert()
        converter.save(self.output_path +'/TFTRT_' + self.precision + '/')
