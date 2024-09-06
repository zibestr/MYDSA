import numpy as np


class ModelInterface:
    """_summary_
    """
    def __init__(self,
                 serialized_filename: str | None = None,
                 data_columns: list[str] | None = None):
        """_summary_

        Args:
            serialized_filename (str | None, optional): Path to binary model file. Defaults to None.
            data_columns (list[str] | None, optional): List of columns, which will be used for model. Defaults to None.

        Raises:
            ValueError: _description_  TODO: do message
        """
        if serialized_filename is None:
            self.__max = None
            self.__min = None
        else:
            self.__model = None  # TODO: load model from binary
            self.__max_features = None
            self.__min_features = None
            self.__max_target = None
            self.__min_target = None
        self._columns = data_columns

    def __normalize(self, data):
        return np.hstack(
            [(data[:, i] - self.__min[i]) /
             (self.__max_features[i] - self.__min_features[i])
             for i in range(data.shape[0])]
        )

    def __denormalize(self, data):
        return (data * (self.__max_target - self.__min_target)
                + self.__min_target)

    def __predict_one(self, data) -> float:
        if len(data) != len(self._columns):
            raise ValueError('Data shape mismatch train data shape')
        data = self.__normalize(data)
        pred = self.__model.predict(data)
        return self.__denormalize(pred)
