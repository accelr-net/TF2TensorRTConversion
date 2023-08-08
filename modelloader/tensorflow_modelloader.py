import typing

import numpy as np
import tensorflow as tf

from base.model import Model


class TensorflowInfer(Model):
    """
    Class for performing inference using a TensorFlow SavedModel.
    """

    def __init__(self, saved_model_dir: str, batch_size: int) -> None:

        """
        Initialize the TensorflowInfer object.

        Args:
            saved_model_dir (str): The path to the directory containing the TensorFlow SavedModel.
            batch_size (int): The batch size for inference.
        """

        super().__init__(saved_model_dir, batch_size)

        # Configure GPU memory growth to avoid memory allocation issues
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load the SavedModel and get the prediction function
        self.model = tf.saved_model.load(saved_model_dir)
        self.pred_fn = self.model.signatures["serving_default"]

        # Setup I/O bindings for the model
        self.inputs = []
        fn_inputs = self.pred_fn.structured_input_signature[1]
        for i, input in enumerate(list(fn_inputs.values())):
            self.inputs.append(
                {
                    "index": i,
                    "name": input.name,
                    "dtype": np.dtype(input.dtype.as_numpy_dtype()),
                    "shape": input.shape.as_list(),
                }
            )
        self.outputs = []
        fn_outputs = self.pred_fn.structured_outputs
        for i, output in enumerate(list(fn_outputs.values())):
            self.outputs.append(
                {
                    "index": i,
                    "name": output.name,
                    "dtype": np.dtype(output.dtype.as_numpy_dtype()),
                    "shape": output.shape.as_list(),
                }
            )

    def input_spec(self) -> tuple:
        """
        Get the input specification of the model.

        Returns:
            tuple: A tuple containing the shape and dtype of the input.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self) -> tuple:
        """
        Get the output specification of the model.

        Returns:
            tuple: A tuple containing the shape and dtype of the output.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """
        Perform inference using the TensorFlow SavedModel.

        Args:
            batch (np.ndarray): The input batch for inference.

        Returns:
            np.ndarray: The inferred classes for the input batch.
        """

        # Process I/O and execute the network
        input = {self.inputs[0]["name"]: tf.convert_to_tensor(batch)}
        output = self.pred_fn(**input)
        output = output[self.outputs[0]["name"]].numpy()

        # Read and process the results
        classes = np.argmax(output, axis=1)

        return classes

    @staticmethod
    def create_instance(config: typing.Dict[str, typing.Any]) -> Model:
        """
        Create an instance of the TensorflowInfer class.

        Args:
            config (dict): A dictionary containing the configuration parameters for TensorflowInfer.

        Returns:
            TensorflowInfer: An instance of the TensorflowInfer class.
        """
        return TensorflowInfer(config["saved_model_dir"], config["batch_size"])
