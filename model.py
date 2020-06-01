import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgb


class Processor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        if y is None:
            return X

        return X, y

def build_model():
    cat_cols = ["Product_Info_2"]
    cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        [('cat', cat_pipe, cat_cols)],
        remainder='passthrough'
    )

    return Pipeline([("preprocessor", preprocessor), 
                     ("model", lgb.LGBMClassifier(
                        num_leaves=45,
                        learning_rate=0.04,
                        n_estimators=300,
                        min_data_in_leaf=100,
                        class_weights="balanced")
                      )
                    ]
    )