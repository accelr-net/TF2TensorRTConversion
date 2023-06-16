import os
import sys


from PIL import Image
import numpy as np
import tensorflow as tf

class Data:

    def __init__(self, data_dir: str, filename_pattern: str, batch_size: int ) -> None:
        
        self.data_dir = data_dir
        self.filename_pattern= filename_pattern
        self.batch_size = batch_size

    def get_batch(self):
        raise NotImplementedError
