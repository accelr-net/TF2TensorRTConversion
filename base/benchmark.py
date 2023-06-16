import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import numpy as np
import typing

from dataloader.tfrecord_dataloader import TfRecorder
#from dataloader.image_dataloader import Images
from modelloader.tensorflow_modelloader import TensorflowInfer
from modelloader.tensorrt_modelloader import TensorRTInfer
from modelloader.tftrt_modelloader import TFTRTInfer

class Benchmark:

    def __init__(self, model_path: str, data_dir: str, filename_pattern: str, batch_size: int, type: str) -> None:

        self.model_path = model_path
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size
        self.type = type

    def load_model(self):
        config = {
                "saved_model_dir": self.model_path,
                "batch_size": self.batch_size
                }
        if self.type == "Tensorflow":            
            model = TensorflowInfer.create_instance(config)
        elif self.type == "TensorRT":
            model = TensorRTInfer.create_instance(config)
        elif self.type == "TFTRT":
            model = TFTRTInfer.create_instance(config)
        
        return model

    
    def load_data(self):

        config = {
            "data_dir": self.data_dir,
            "filename_pattern": self.filename_pattern,
            "batch_size": self.batch_size
        }

        data = TfRecorder.create_instance(config)

        return data

    def generate_results(self):

        inference = self.load_model()
        data = self.load_data()
        dataset = data.get_batch()
  
        print('Warming up for 50 batches...')
        
        cnt = 0
        for x, y in dataset:
            print("preds.shape {}".format(x.shape))
            labeling = inference.infer(x)
            cnt += 1
            if cnt == 50:
                break

        print('Benchmarking inference engine...')
        num_hits = 0
        num_predict = 0
        start_time = time.time()
        
        for x, y in dataset:
            yLabels = []
            labeling = inference.infer(x)
            print("preds.shape {}".format(x.shape))
            
            
            for x in labeling: 
                x+=1
                yLabels.append(x)
                       
            
            y = np.squeeze(y)         
            k = (yLabels == y)        
            num_hits += np.sum(yLabels == y)
            num_predict += np.shape(yLabels)[0]
            
                
        print(num_hits)
        print(num_predict)
        print('Accuracy: %.2f%%'%(100*num_hits/num_predict))
        print('Inference speed: %.2f samples/s'%(num_predict/(time.time()-start_time)))

    