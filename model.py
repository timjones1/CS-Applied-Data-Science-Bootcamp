import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def preprocess(df):
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

    msk_eval = df.evaluation_set

    df = df[["blurb", "state"]]
    df.blurb.fillna("",inplace=True)
    X_train = df[~msk_eval].drop(["state"], axis=1)
    y_train = df[~msk_eval]["state"]
    X_test = df[msk_eval].drop(["state"], axis=1)
    #create Countvectorizer object and create a vector of word counts
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train.blurb)
    #create Tf/idf transformer and transform train set
    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(X_train_counts)
    # transform test set
    X_test_counts = count_vect.transform(X_test.blurb)
    X_test_tf = tf_transformer.transform(X_test_counts)
    
    X = X_train_tf
    y = y_train
    X_eval = X_test_tf
    
    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """

    model = MultinomialNB()
    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well 
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """

    y_pred = model.predict(X_test)
    return y_pred
