import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np

from dataloader.tfrecord_dataloader import TfRecorder
from modelloader.tensorflow_modelloader import TensorflowInfer
from modelloader.tensorrt_modelloader import TensorRTInfer
from modelloader.tftrt_modelloader import TFTRTInfer


class Benchmark:
    """
    A class for benchmarking inference performance and accuracy of a model on a dataset.
    """

    def __init__(self, model_path: str, data_dir: str, filename_pattern: str, batch_size: int, type: str) -> None:

        """
        Initialize the Benchmark object.

        Args:
            model_path (str): The path to the pre-trained model.
            data_dir (str): The directory containing the dataset.
            filename_pattern (str): The pattern for data file names.
            batch_size (int): The batch size for inference.
            type (str): The type of the model (Tensorflow, TensorRT, or TFTRT).
        """

        self.model_path = model_path
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size
        self.type = type

    def load_model(self):

        """
        Load the pre-trained model based on the specified type.

        Returns:
            model: An instance of the loaded model.
        """
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

        """
        Load the dataset for inference.

        Returns:
            data: An instance of the loaded dataset.
        """

        config = {
            "data_dir": self.data_dir,
            "filename_pattern": self.filename_pattern,
            "batch_size": self.batch_size
        }

        data = TfRecorder.create_instance(config)

        return data

    def generate_results(self):

        """
        Perform benchmarking of the model's inference speed and accuracy on the dataset.
        """

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
            ylabels = []
            labeling = inference.infer(x)
            print("preds.shape {}".format(x.shape))

            for m in labeling:
                m += 1
                ylabels.append(m)

            y = np.squeeze(y)
            k = (ylabels == y)
            num_hits += np.sum(ylabels == y)
            num_predict += np.shape(ylabels)[0]

        print(num_hits)
        print(num_predict)
        print('Accuracy: %.2f%%' % (100 * num_hits / num_predict))
        print('Inference speed: %.2f samples/s' % (num_predict / (time.time() - start_time)))
