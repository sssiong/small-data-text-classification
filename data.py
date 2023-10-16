import time
from dataclasses import dataclass

import pandas as pd
from skmultilearn.model_selection import IterativeStratification

from constants import DATASOURCES


@dataclass
class DataCollector:
    n_train: int = 1000   # max no of records for train data
    n_test: int = 200     # max no of records for test data
    n_labels: int = None  # max no of label columns


    @property
    def datasets(self):
        for name, meta in self._datasets.items():
            yield (name, meta)

    def get_datasets(self, name: str):
        return self._datasets[name]

    def load(self):

        self._datasets = {}
        for name, meta in DATASOURCES.items():
            
            filepath, textcol, labelcols = meta['filepath'], meta['textcol'], meta['labelcols']

            t = time.time()
            print(f'Loading {name} - {filepath.name} ...')

            # load dataframe
            df = pd.read_csv(filepath)
            X, y = df[[textcol]], df[labelcols].fillna(0)

            # limit no of label cols (if required)
            if self.n_labels is not None:
                y = y.iloc[:, :self.n_labels]

                # throw away records that end up with all 0 due to the slicing
                y = y[~(y==0).all(axis=1)]
                X = X.loc[y.index, :]

            # subset to required sample size
            nrows = self.n_train + self.n_test
            y = y.sample(n=nrows, random_state=123)
            X = X.loc[y.index, :]

            # train test split
            test_size = self.n_test / nrows
            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1-test_size])
            X, y = X.reset_index(drop=True), y.reset_index(drop=True) # need index to be running order
            train_indexes, test_indexes = next(stratifier.split(X, y))
            X_train, y_train = X.loc[train_indexes, textcol], y.loc[train_indexes]
            X_test, y_test = X.loc[test_indexes, textcol], y.loc[test_indexes]

            # output
            self._datasets[name] = (X_train, y_train, X_test, y_test)

            print(f'Done in {time.time()-t:.1f}s with {X_train.shape[0]} records in train & {X_test.shape[0]} records in test')

    def print_label_distribution(self):
        for name, (_, y_train, _, y_test) in self._datasets.items():
            out = pd.concat([
                y_train.sum().to_frame('train'),
                y_test.sum().to_frame('test'),
            ], axis=1).astype(int)
            print(f'\n*** Dataset: {name} (train={y_train.shape[0]}, test={y_test.shape[0]}) ***\n')
            print(out)
