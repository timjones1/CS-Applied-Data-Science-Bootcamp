from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """
    drop_features = ['Id', 'Medical_History_10', 'Medical_History_24', 'Medical_History_32']
    categorical_features = ['Product_Info_2']
    numeric_features = ['Product_Info_1', 'Product_Info_3', 'Product_Info_4', 'Product_Info_5',
                        'Product_Info_6', 'Product_Info_7', 'Ins_Age', 'Ht', 'Wt', 'BMI',
                        'Employment_Info_1', 'Employment_Info_2', 'Employment_Info_3',
                        'Employment_Info_4', 'Employment_Info_5', 'Employment_Info_6',
                        'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
                        'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                        'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3',
                        'Insurance_History_4', 'Insurance_History_5', 'Insurance_History_7',
                        'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
                        'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
                        'Medical_History_1', 'Medical_History_2', 'Medical_History_3',
                        'Medical_History_4', 'Medical_History_5', 'Medical_History_6',
                        'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
                        'Medical_History_11', 'Medical_History_12', 'Medical_History_13',
                        'Medical_History_14', 'Medical_History_15', 'Medical_History_16',
                        'Medical_History_17', 'Medical_History_18', 'Medical_History_19',
                        'Medical_History_20', 'Medical_History_21', 'Medical_History_22',
                        'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
                        'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
                        'Medical_History_30', 'Medical_History_31', 'Medical_History_33',
                        'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
                        'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
                        'Medical_History_40', 'Medical_History_41', 'Medical_Keyword_1',
                        'Medical_Keyword_2', 'Medical_Keyword_3', 'Medical_Keyword_4',
                        'Medical_Keyword_5', 'Medical_Keyword_6', 'Medical_Keyword_7',
                        'Medical_Keyword_8', 'Medical_Keyword_9', 'Medical_Keyword_10',
                        'Medical_Keyword_11', 'Medical_Keyword_12', 'Medical_Keyword_13',
                        'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16',
                        'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19',
                        'Medical_Keyword_20', 'Medical_Keyword_21', 'Medical_Keyword_22',
                        'Medical_Keyword_23', 'Medical_Keyword_24', 'Medical_Keyword_25',
                        'Medical_Keyword_26', 'Medical_Keyword_27', 'Medical_Keyword_28', 'Medical_Keyword_29',
                        'Medical_Keyword_30', 'Medical_Keyword_31', 'Medical_Keyword_32',
                        'Medical_Keyword_33', 'Medical_Keyword_34', 'Medical_Keyword_35',
                        'Medical_Keyword_36', 'Medical_Keyword_37', 'Medical_Keyword_38',
                        'Medical_Keyword_39', 'Medical_Keyword_40', 'Medical_Keyword_41',
                        'Medical_Keyword_42', 'Medical_Keyword_43', 'Medical_Keyword_44',
                        'Medical_Keyword_45', 'Medical_Keyword_46']
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(remainder='passthrough',
                                     transformers=[
                                         ('drop_columns', 'drop', drop_features),
                                         ('categoricals', categorical_transformer, categorical_features),
                                         ('numericals', numeric_transformer, numeric_features)
                                     ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", LGBMClassifier())])
    return pipeline






'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgb


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
  '''