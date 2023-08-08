class Data:

    """
    Abstract base class for providing data to  models during inference.
    """

    def __init__(self, data_dir: str, filename_pattern: str, batch_size: int) -> None:
        
        """
        Initialize the Data object.

        Args:
            data_dir (str): The directory containing the dataset.
            filename_pattern (str): The pattern for data file names.
            batch_size (int): The batch size for data loading.
        """
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size

    def get_batch(self):

        """
        Abstract method to get a batch of data from the dataset.

        This method should be implemented in the subclass to provide functionality
        for loading and preprocessing data and returning it as a suitable dataset for required model.

        Raises:
            NotImplementedError: The method is not implemented in the subclass.
        """

        raise NotImplementedError
