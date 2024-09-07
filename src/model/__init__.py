import datetime as dt
import warnings

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from joblib import dump, load
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


class ModelInterface:
    def __init__(self,
                 model_filename: str | None = None,
                 label_encoder: str | None = None,
                 feature_encoder: str | None = None,
                 target_encoder: str | None = None):
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
        df = pd.read_csv(filename, sep=',')
        try:
            df = df.loc[df['failure'] == 0]
            date = dt.date(*[int(part)
                             for part in df['date'].iloc[0].split('-')])
            ids = df['serial_number'].to_list()
            df = df[self._columns + [self._target_column,
                                     self._cat_column]]
            return df, date, ids
        except IndexError:
            raise ValueError(f'Unknown rows in dataset: {filename}')

    def __to_datetime(self,
                      date: dt.datetime,
                      pred: np.ndarray) -> np.ndarray:
        relu = np.vectorize(lambda x: x if x >= 0 else 0)
        date = np.datetime64(date)
        return relu(pred).astype('timedelta64[h]') + date

    def __interpret_predictions(self,
                                date: dt.datetime,
                                ids: list[str],
                                pred: np.ndarray) -> pd.DataFrame:
        datetimes = self.__to_datetime(date,
                                       pred)
        df = pd.DataFrame()
        df['serial_numbers'] = ids
        df['failure_date'] = datetimes
        return df

    def predict(self, filename: str) -> float:
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
        for i in range(len(disks)):
            if disks[i] not in self.__label_encoder.classes_:
                disks[i] = 'unknown'
        if self.__label_encoder is None:
            self.__label_encoder = LabelEncoder().fit([[disk]
                                                       for disk in disks])
        X = df.loc[:, ~df.columns.isin([self._target_column,
                                        self._cat_column])].to_numpy()
        X = np.hstack([X,
                       self.__label_encoder.transform(
                           df[self._cat_column].to_numpy().reshape(-1, 1)
                       ).reshape(-1, 1)])
        y = df[self._target_column].to_numpy().reshape(-1, 1)
        if self.__feature_encoder is None:
            self.__feature_encoder = MinMaxScaler().fit(X)
            self.__target_encoder = MinMaxScaler().fit(y)
        X = self.__feature_encoder.transform(X)
        y = self.__target_encoder.transform(
            y.reshape(-1, 1)
        )
        return (X, y)

    def __get_X(self, df: pd.DataFrame) -> np.ndarray:
        for i in range(len(df[self._cat_column])):
            if df[self._cat_column].iloc[i] \
               not in self.__label_encoder.classes_:
                df[self._cat_column].iloc[i] = 'unknown'
        X = df.loc[:, ~df.columns.isin([self._target_column,
                                        self._cat_column])].to_numpy()
        X = np.hstack([X,
                       self.__label_encoder.transform(
                           df[self._cat_column].to_numpy().reshape(-1, 1)
                       ).reshape(-1, 1)])
        X = self.__feature_encoder.transform(X)
        return X

    def train(self,
              filenames: list[str],
              model_filename: str,
              label_encoder: str,
              feature_encoder: str,
              target_encoder: str) -> tuple[float, float]:
        seed = 67947
        params = {'random_state': seed,
                  'tree_method': 'hist',
                  'objective': 'reg:squarederror',
                  'eval_metric': ['mae', 'rmse', 'mape']}

        df, disks = self.__merge_dataset(filenames)
        X_train, X_test, y_train, y_test = train_test_split(
            *self.__get_X_y(df, disks),
            test_size=0.2, random_state=seed, shuffle=True
        )

        train_data = xgb.DMatrix(data=X_train, label=y_train)
        test_data = xgb.DMatrix(data=X_test, label=y_test)

        self.__model = xgb.train(params, train_data, 150)
        y_pred = self.__model.predict(test_data)

        dump(self.__model, model_filename)
        dump(self.__label_encoder, label_encoder)
        dump(self.__feature_encoder, feature_encoder)
        dump(self.__target_encoder, target_encoder)

        return r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test,
                                                                        y_pred)

    def inc_train(self,
                  filenames: list[str]) -> tuple[float, float]:
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

        return r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test,
                                                                        y_pred)
