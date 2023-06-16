
import typing

import numpy as np
import tensorflow as tf


from base.model import Model

class TFTRTInfer(Model):
    
    def __init__(self, saved_model_dir: str, batch_size: int) -> None:

        super().__init__(saved_model_dir, batch_size)

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model = tf.saved_model.load(saved_model_dir)
        self.pred_fn = self.model.signatures["serving_default"]

        # Setup I/O bindings
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
                    "shape": (-1, 1001),
                }
            )

    def input_spec(self):
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch):
        # Process I/O and execute the network
        input = {self.inputs[0]["name"]: tf.convert_to_tensor(batch)}
        output = self.pred_fn(**input)
        output = output[self.outputs[0]["name"]].numpy()

        # Read and process the results
        classes = np.argmax(output, axis=1)
        scores = np.max(output, axis=1)

        return classes

    @staticmethod
    def create_instance(config: typing.Dict[str, typing.Any]) -> Model:
        return TFTRTInfer(config["saved_model_dir"], config["batch_size"])
