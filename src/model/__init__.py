import datetime as dt
import warnings

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from joblib import dump, load
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


class ModelInterface:
    """Programming interface for interacting with XGBoost model.\\
    Realize train, predict and incremental learning
    """
    def __init__(self,
                 model_filename: str | None = None,
                 label_encoder: str | None = None,
                 feature_encoder: str | None = None,
                 target_encoder: str | None = None):
        """Programming interface for interacting with XGBoost model.\\
           Realize train, predict and incremental learning

        Args:
            **model_filename** (str | None, optional): path to serialized xgb
            model. Defaults to None.
            **label_encoder** (str | None, optional): path to serialized label
            encoder. Defaults to None.
            **feature_encoder** (str | None, optional): path to serialized
            scaler for feature columns. Defaults to None.
            **target_encoder** (str | None, optional): path to serialized
            scaler for target column. Defaults to None.
        """
        if model_filename is None:
            self.__model = None
            self.__label_encoder = None
            self.__feature_encoder = None
            self.__target_encoder = None
        else:
            self.__model = load(model_filename)
            self.__label_encoder = load(label_encoder)
            self.__feature_encoder = load(feature_encoder)
            self.__target_encoder = load(target_encoder)
            self.__filenames = [model_filename,
                                label_encoder,
                                feature_encoder,
                                target_encoder]
        # Required columns for model training and prediction
        self._columns = ['capacity_bytes', 'smart_1_raw', 'smart_4_raw',
                         'smart_5_raw', 'smart_7_raw', 'smart_12_raw',
                         'smart_190_raw', 'smart_192_raw', 'smart_193_raw',
                         'smart_194_raw', 'smart_240_raw', 'smart_241_raw',
                         'smart_242_raw']
        self._target_column = 'smart_9_raw'
        self._cat_column = 'model'

    def __open_dataframe(self, filename: str) -> tuple[pd.DataFrame,
                                                       dt.date,
                                                       list[str]]:
        """Open csv table

        Args:
            **filename** (str): csv table filename

        Raises:
            **ValueError**: raises if the table don't contains
            the required columns.

        Returns:
            tuple[DataFrame, datetime.date, list[str]]: dataframe,
            data aggregation date, unique disk names
        """
        df = pd.read_csv(filename, sep=',')
        try:
            df = df.loc[df['failure'] == 0]
            date = dt.date(*[int(part)
                             for part in df['date'].iloc[0].split('-')])
            ids = df['serial_number'].to_list()
            df = df[self._columns + [self._target_column,
                                     self._cat_column]]
            return df, date, ids
        except IndexError as exc:
            raise ValueError(f'Unknown rows in dataset: {filename}') from exc

    def __to_datetime(self,
                      date: dt.datetime,
                      pred: np.ndarray) -> np.ndarray:
        """Convert predicted time from hours to dates

        Args:
            **date** (dt.datetime): data aggregation date
            **pred** (np.ndarray): predicted time before failure
        Returns:
            ndarray: failure dates
        """
        relu = np.vectorize(lambda x: x if x >= 0 else 0)
        date = np.datetime64(date)
        return relu(pred).astype('timedelta64[h]') + date

    def __interpret_predictions(self,
                                date: dt.datetime,
                                ids: list[str],
                                pred: np.ndarray) -> pd.DataFrame:
        """Convert predictions in dataframe

        Args:
            **date** (dt.datetime): data aggregation date
            **ids** (list[str]): serial numbers of devices
            **pred** (np.ndarray): predicted time before failure

        Returns:
            DataFrame: formatted dataframe
        """
        datetimes = self.__to_datetime(date,
                                       pred)
        df = pd.DataFrame()
        df['serial_numbers'] = ids
        df['failure_date'] = datetimes
        return df

    def predict(self, filename: str) -> pd.DataFrame:
        """Predict failure dates of devices in given file

        Args:
            **filename** (str): csv table filename

        Raises:
            ValueError: raises if call predict before training model

        Returns:
            DataFrame: formatted dataframe
        """
        if self.__model is None:
            raise ValueError('model must be trained before predictions')
        data, date, ids = self.__open_dataframe(filename)
        X = self.__get_X(data)
        times = data['smart_9_raw'].to_numpy().reshape(-1, 1)
        X = xgb.DMatrix(X)
        pred = self.__target_encoder.inverse_transform(
            self.__model.predict(X).reshape(-1, 1)
        ).astype(int)
        return self.__interpret_predictions(date,
                                            ids,
                                            times - pred)

    def __get_failures(self, filename: str) -> tuple[pd.DataFrame,
                                                     list[str]]:
        """_summary_

        Args:
            filename (str): _description_

        Returns:
            tuple[DataFrame, list[str]]: _description_
        """
        df = pd.read_csv(filename, sep=',')
        disks = df[self._cat_column].unique().tolist()
        df = df.loc[df['failure'] == 1]
        df = df.loc[:, df.columns.isin(self._columns
                                       + [self._target_column,
                                          self._cat_column])]
        return df, disks

    def __merge_dataset(self,
                        filenames: list[str]) -> tuple[pd.DataFrame,
                                                       list[str]]:
        """Automation building dataset from given files.\\
        Dataset is built on the basis of failed disks.

        Args:
            filenames (list[str]): names of csv files

        Raises:
            ValueError: raises if none of the files contain the
            required columns.

        Returns:
            tuple[pd.DataFrame, list[str]]: dataset, list of unique model disks
        """
        df, disks = None, None
        for filename in tqdm.tqdm(filenames):
            try:
                new_df, new_disks = self.__get_failures(filename)
                if df is None:
                    df, disks = new_df, new_disks
                for disk in new_disks:
                    if disk not in disks:
                        disks.append(disk)
                df = pd.concat([df, new_df])
            except IndexError:
                continue
        if df is None:
            raise ValueError('Аt least one file must contain right columns')
        df = df.fillna(0)
        df = df.loc[df[self._target_column] > 1000]
        disks.append('unknown')
        return df, disks

    def __get_X_y(self, df: pd.DataFrame,
                  disks: list[str]) -> tuple[np.ndarray,
                                             np.ndarray]:
        """Return data and labels from dataset

        Args:
            df (pd.DataFrame): dataset
            disks (list[str]): unique names of disks models

        Returns:
            tuple[np.ndarray, np.ndarray]: X, y
        """
        if self.__label_encoder is None:
            self.__label_encoder = LabelEncoder().fit([[disk]
                                                       for disk in disks])
        # get only numerical features
        X = df.loc[:, ~df.columns.isin([self._target_column,
                                        self._cat_column])].to_numpy()
        # stack numerical features with categorical feature
        X = np.hstack([X,
                       self.__label_encoder.transform(
                           df[self._cat_column].to_numpy().reshape(-1, 1)
                       ).reshape(-1, 1)])
        y = df[self._target_column].to_numpy().reshape(-1, 1)
        if self.__feature_encoder is None:
            self.__feature_encoder = MinMaxScaler().fit(X)
            self.__target_encoder = MinMaxScaler().fit(y)
        # normalize numerical data
        X = self.__feature_encoder.transform(X)
        y = self.__target_encoder.transform(
            y.reshape(-1, 1)
        )
        return (X, y)

    def __get_X(self, df: pd.DataFrame) -> np.ndarray:
        """Return X from dataset

        Args:
            df (pd.DataFrame): dataset

        Returns:
            np.ndarray: X
        """
        # fill unknown models of devices
        for i in range(len(df[self._cat_column])):
            if df[self._cat_column].iloc[i] \
               not in self.__label_encoder.classes_:
                df[self._cat_column].iloc[i] = 'unknown'
        # get only numerical features
        X = df.loc[:, ~df.columns.isin([self._target_column,
                                        self._cat_column])].to_numpy()
        # stack numerical features with categorical feature
        X = np.hstack([X,
                       self.__label_encoder.transform(
                           df[self._cat_column].to_numpy().reshape(-1, 1)
                       ).reshape(-1, 1)])
        # normalize numerical data
        X = self.__feature_encoder.transform(X)
        return X

    def train(self,
              filenames: list[str],
              model_filename: str,
              label_encoder: str,
              feature_encoder: str,
              target_encoder: str) -> tuple[float, float]:
        """Training a model from scratch

        Args:
            filenames (list[str]): names of csv files
            model_filename (str): filename to serialize the model
            label_encoder (str): filename to serialize the label encoder
            feature_encoder (str): filename to serialize the feature scaler
            target_encoder (str): filename to serialize the target scaler

        Returns:
            tuple[float, float]: **R<sup>2</sup>** score, **MSE**
        """
        # best xgboost parameters for current task
        seed = 67947
        params = {
            'random_state': seed,
            'tree_method': 'exact',
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'lambda': 0.697324690805717,
            'alpha': 9.172742855494088e-05,
            'subsample': 0.9121489394519637,
            'colsample_bytree': 0.951654350087416,
            'eval_metric': ['mae', 'rmse', 'mape']
        }

        df, disks = self.__merge_dataset(filenames)
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            *self.__get_X_y(df, disks),
            test_size=0.2, random_state=seed, shuffle=True
        )

        train_data = xgb.DMatrix(data=X_train, label=y_train)
        test_data = xgb.DMatrix(data=X_test, label=y_test)

        self.__model = xgb.train(params, train_data, 150)
        y_pred = self.__model.predict(test_data)

        # serialize model with encoders
        dump(self.__model, model_filename)
        dump(self.__label_encoder, label_encoder)
        dump(self.__feature_encoder, feature_encoder)
        dump(self.__target_encoder, target_encoder)

        return r2_score(y_test, y_pred), mean_squared_error(y_test,
                                                            y_pred)

    def inc_train(self,
                  filenames: list[str]) -> tuple[float, float]:
        """Implements incremential learning xgboost model

        Args:
            filenames (list[str]): names of csv files

        Raises:
            ValueError:  raises if call incremential learning
            before training model

        Returns:
            tuple[float, float]: **R<sup>2</sup>** score, **MSE**
        """
        if self.__model is None:
            raise ValueError('model must be trained before incrementional'
                             'training')
        seed = 67947
        params = {
            'updater': 'refresh',
            'process_type': 'update',
            'refresh_leaf': True
        }

        df, disks = self.__merge_dataset(filenames)
        X_train, X_test, y_train, y_test = train_test_split(
            *self.__get_X_y(df, disks),
            test_size=0.2, random_state=seed, shuffle=True
        )

        train_data = xgb.DMatrix(data=X_train, label=y_train)
        test_data = xgb.DMatrix(data=X_test, label=y_test)

        self.__model = xgb.train(params,
                                 train_data,
                                 30,
                                 xgb_model=self.__model)
        y_pred = self.__model.predict(test_data)

        dump(self.__model, self.__filenames[0])
        dump(self.__label_encoder, self.__filenames[1])
        dump(self.__feature_encoder, self.__filenames[2])
        dump(self.__target_encoder, self.__filenames[3])

        return r2_score(y_test, y_pred), mean_squared_error(y_test,
                                                            y_pred)
