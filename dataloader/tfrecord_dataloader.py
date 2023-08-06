import os
import sys
import typing

from PIL import Image
import numpy as np
import tensorflow as tf

from base.data import Data


class TfRecorder(Data):

    """
    Class to load and preprocess data from TFRecord files and provide it as a TensorFlow dataset for model inference.
    """

    def __init__(self, data_dir: str, filename_pattern: str, batch_size: int):

        """
        Initialize the TfRecorder object.

        Args:
            data_dir (str): The directory containing the TFRecord files.
            filename_pattern (str): The pattern for the TFRecord file names.
            batch_size (int): The batch size for data loading.
        """
        super().__init__(data_dir, filename_pattern, batch_size)     
                    
    def deserialize_image_record(self,record: tf.Tensor) -> tuple:

        """
        Deserialize an image record from the TFRecord.

        Args:
            record: The TFRecord to be deserialized.

        Returns:
            tuple: A tuple containing the image data, label, bounding box, and text from the TFRecord.
        """

        # Feature map to parse the TFRecord
        feature_map = {
            'image/encoded':          tf.io.FixedLenFeature([ ], tf.string, ''),
            'image/class/label':      tf.io.FixedLenFeature([1], tf.int64,  0),
            'image/class/text':       tf.io.FixedLenFeature([ ], tf.string, ''),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
        }
        with tf.name_scope('deserialize_image_record'):
            obj = tf.io.parse_single_example(record, feature_map)
            imgdata = obj['image/encoded']
            label   = tf.cast(obj['image/class/label'], tf.int32)
            bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                                for x in ['ymin', 'xmin', 'ymax', 'xmax']])
            bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
            text    = obj['image/class/text']
            return imgdata, label, bbox, text

    def preprocessing (self, image: tf.Tensor, height: int, width: int) -> tf.Tensor:

        """
        Preprocess the image by normalizing and resizing it.

        Args:
            image (tf.Tensor): The input image tensor.
            height (int): The target height for resizing.
            width (int): The target width for resizing.

        Returns:
            tf.Tensor: The preprocessed image tensor.
        """
    
        image = image/255
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)               
        image = tf.image.resize(image, [height, width])
        return image

    def preprocess(self,record: tf.Tensor) -> tuple:

        """
        Preprocess a single record from the TFRecord.

        Args:
            record (tf.Tensor): The TFRecord to be preprocessed.

        Returns:
            tuple: A tuple containing the preprocessed image and its label.
        """
        # Parse TFRecord
        imgdata, label, _, _ = self.deserialize_image_record(record)
        #label -= 1 # Change to 0-based if not using background class
        try:    
            image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: 
            image = tf.image.decode_png(imgdata, channels=3)

        image = self.preprocessing(image, 224, 224)
        
        return image, label
    

    def get_batch(self) -> tf.data.Dataset:

        """
        Load and preprocess the TFRecord files and return the data as a TensorFlow dataset.

        Returns:
            tf.data.Dataset: The preprocessed data as a TensorFlow dataset.
        """

        if self.data_dir == None:
            return []
        files = tf.io.gfile.glob(os.path.join(self.data_dir, self.filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with '
                         'pattern "{}"'.format(self.data_dir, self.filename_pattern))
        dataset = tf.data.TFRecordDataset(files)   
        dataset = dataset.map(map_func=self.preprocess, num_parallel_calls=20)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)

        return dataset

    @staticmethod
    def create_instance(config: typing.Dict[str, typing.Any]) -> object:
        """
        Create an instance of the TfRecorder class.

        Args:
            config (dict): A dictionary containing the configuration parameters for TfRecorder.

        Returns:
            TfRecorder: An instance of the TfRecorder class.
        """
        return TfRecorder(config["data_dir"], config["filename_pattern"], config["batch_size"])