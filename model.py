# imports
import json
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


def preprocess(data):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training set: X and y together
    with the preprocessed evaluation set: X_eval.

    Make sure you return X_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]

    For y you can either return a pd.DataFrame with one column or pd.Series.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    data.blurb.fillna("missing", inplace=True)
    data.name.fillna("missing", inplace=True)

    data['cat_slug'] = data['category'].apply(
        lambda x: json.loads(x).get('slug'))

    data['cat_parent'] = data['category'].apply(
        lambda x: json.loads(x).get('parent_id'))

    data['goal'] = np.log1p((data.goal*data.static_usd_rate))

    data['blurb_length'] = data['blurb'].apply(len)
    data['name_length'] = data['name'].apply(len)

    # convert time dependent fields into datetime and add extra date fields
    from datetime import datetime

    for col in ['launched_at', 'deadline', 'created_at']:
        data[col] = data[col].apply(
            lambda ts: datetime.utcfromtimestamp(ts))
        data[f'{col}_day'] = pd.DatetimeIndex(data[col]).day
        data[f'{col}_month'] = pd.DatetimeIndex(data[col]).month
        data[f'{col}_year'] = pd.DatetimeIndex(data[col]).year

    # extract location info
    data['loc_state'] = data['location'].apply(
        lambda x: json.loads(x).get('state') if (np.all(pd.notnull(x))) else "Other")
    data['loc_type'] = data['location'].apply(
        lambda x: json.loads(x).get('type') if (np.all(pd.notnull(x))) else "Other")

    data['loc_name'] = data['location'].apply(
        lambda x: json.loads(x).get('name') if (np.all(pd.notnull(x))) else "Other")
    # get only states with > 5 examples and names > 50 examples
    top_names = data['loc_name'].value_counts()[
        data['loc_name'].value_counts() > 10]
    top_states = data['loc_state'].value_counts()[
        data['loc_state'].value_counts() > 2]

    data['loc_name'] = data['loc_name'].apply(
        lambda n: n if n in top_names else "other")
    data['loc_state'] = data['loc_state'].apply(
        lambda n: n if n in top_states else "other")

    # Calculates campaign length, thanks to @paulo
    data['campaign_active_length'] = pd.to_timedelta(data['deadline'] - data['launched_at'], unit='s').dt.days
    data['campaign_total_length'] = pd.to_timedelta(data['deadline'] - data['created_at'], unit='s').dt.days
    data['campaign_prep_length'] = pd.to_timedelta(data['launched_at'] - data['created_at'], unit='s').dt.days

    data['goal_per_active_day'] = data['goal'] / data['campaign_active_length']
    msk_eval = data.evaluation_set

    X = data[~msk_eval].drop(["state"], axis=1)
    y = data[~msk_eval]["state"]
    X_eval = data[msk_eval].drop(["state"], axis=1)

    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """

    NGRAM_RANGE = (1, 2)
    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = 'word'
    # Minimum document/corpus frequency below which a token will be discarded.
    MIN_DOCUMENT_FREQUENCY = 2
    # Limit on the number of features. We use the top 10K features.
    TOP_K = 5000
    # Create keyword arguments to pass to the vectorizer.
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
    }

    numeric_features = ['goal', "blurb_length", "name_length", "launched_at_day",
                       "launched_at_month", "launched_at_year", "deadline_day",
                       "deadline_month", "deadline_year", "created_at_day",
                       "created_at_month", "created_at_year", "campaign_active_length",
                       "campaign_total_length", "campaign_prep_length",
                       'goal_per_active_day']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    text_features = 'blurb'
    text_transformer = Pipeline([
        ('vect', CountVectorizer(**kwargs)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('selector', SelectKBest(chi2, TOP_K))
    ])

    # name_features = 'name'
    # name_transformer = Pipeline([
    #     ('vect_n', CountVectorizer(ngram_range=(1,2))),
    #     ('tfidf_n', TfidfTransformer()),
    # ])

    categorical_features = ['country', 'cat_slug', 'loc_name', 'loc_state']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text_blurb', text_transformer, text_features),
            # ('text_name', name_transformer, name_features),

        ], transformer_weights={
            'num': 1.2,
            'cat': 1.0,
            'text_blurb': 1.8,
            'text_name': 0.8,
        }
    )
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('xgb', XGBClassifier(
                                silent=False,
                                scale_pos_weight=1,
                                learning_rate=0.1,
                                colsample_bytree=0.25,
                                subsample=0.8,
                                objective='binary:logistic',
                                n_estimators=500,
                                reg_alpha=0.3,
                                max_depth=4,
                                gamma=10))])

    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with onecolumn
    or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """

    y_pred = model.predict(X_test)
    return y_pred
