import sys
import os
import argparse
import typing

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda

try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from base.model import Model

class TensorRTInfer(Model):
    
    def __init__(self, saved_model_dir: str, batch_size: int) -> None:

        super().__init__(saved_model_dir, batch_size)
        
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(saved_model_dir, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
      
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch):
     
        # Prepare the output data
        output = np.zeros(*self.output_spec())
        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])

        # Process the results
        classes = np.argmax(output, axis=1)
        scores = np.max(output, axis=1)
        return classes
    
    @staticmethod
    def create_instance(config: typing.Dict[str, typing.Any]) -> Model:
        return TensorRTInfer(config["saved_model_dir"], config["batch_size"])