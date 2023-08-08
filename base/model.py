import numpy as np


class Model:
    """
    Abstract base class for defining models for inference.
    """

    def __init__(self, saved_model_dir: str, batch_size: int) -> None:
        """
        Initialize the Model object.

        Args:
            saved_model_dir (str): The directory containing the saved model.
            batch_size (int): The batch size for inference.
        """

        self.saved_model_dir = saved_model_dir
        self.batch_size = batch_size

    def input_spec(self) -> tuple:
        """
        Abstract method to define the input specification for the model.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """

        raise NotImplementedError

    def output_spec(self) -> tuple:
        """
        Abstract method to define the output specification for the model.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """
        raise NotImplementedError

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform inference using the model.

        Args:
            batch: The input data batch for inference.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """
        raise NotImplementedError
