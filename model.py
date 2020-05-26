from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class Processor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = ["Product_Info_4"]

        if y is None:
            return X[cols]

        return X[cols], y


def build_model():
    preprocessor = Processor()
    model = DecisionTreeClassifier()
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

'''
def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """
    raise NotImplementedError
'''