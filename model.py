class KickstarterModel:

    def __init__(self):

        raise NotImplementedError

    def preprocess_training_data(self, df):

        raise NotImplementedError

    def fit(self, x, y):

        raise NotImplementedError

    def preprocess_unseen_data(self, df):

        raise NotImplementedError

    def predict(self, x):

        raise NotImplementedError
