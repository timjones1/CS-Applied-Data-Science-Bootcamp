import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgb

cat_cols = ["Product_Info_2"]

def build_model():
    
    cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        [('cat', cat_pipe, cat_cols)],
        remainder='passthrough'

    )

    lgb_model = lgb.LGBMClassifier(
        num_leaves=45,
        learning_rate=0.04,
        n_estimators=300,
        min_data_in_leaf=100,
        class_weights="balanced")
    
    return Pipeline([("preprocessor", preprocessor), ("model", lgb_model)])
