# import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgb


class Processor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y=None):

        cat_cols = ["Product_Info_2"]

        cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            [('cat', cat_pipe, cat_cols)],
            remainder='passthrough')

        self.preprocessor.fit(X, y)

        return self

    def transform(self, X, y=None):

        if y is None:
            return self.preprocessor.transform(X)
        else:
            return self.preprocessor.transform(X), y[:, 1]


def build_model():

    return Pipeline([
        ("preprocessor", Processor()),
        ("model", lgb.LGBMClassifier(
            num_leaves=45,
            learning_rate=0.04,
            n_estimators=300,
            min_data_in_leaf=100,
            class_weights="balanced"))
    ])
