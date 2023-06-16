import os
import sys
import typing

from PIL import Image
import numpy as np
import tensorflow as tf

from base.data import Data


class TfRecorder(Data):

    def __init__(self, data_dir: str, filename_pattern: str, batch_size: int):

        super().__init__(data_dir, filename_pattern, batch_size)     
                    
    def deserialize_image_record(self,record):
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

    def preprocessing (self, image, height, width):
    
        image = image/255
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)               
        image = tf.image.resize(image, [height, width])
        return image

    def preprocess(self,record):
        # Parse TFRecord
        imgdata, label, bbox, text = self.deserialize_image_record(record)
        #label -= 1 # Change to 0-based if not using background class
        try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: image = tf.image.decode_png(imgdata, channels=3)

        image = self.preprocessing(image, 224, 224)
        
        return image, label

    def get_batch(self):

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
    def create_instance(config: typing.Dict[str, typing.Any]) -> Data:
        return TfRecorder(config["data_dir"], config["filename_pattern"], config["batch_size"])