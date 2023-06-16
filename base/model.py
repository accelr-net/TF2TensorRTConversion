import typing
import numpy as np
import tensorflow as tf



class Model:

    def __init__(self, saved_model_dir: str, batch_size: int) -> None:

        self.saved_model_dir = saved_model_dir
        self.batch_size = batch_size

    
    def input_spec(self):
        raise NotImplementedError

    def output_spec(self):
        raise NotImplementedError

    def infer(self,batch):
        raise NotImplementedError