class InsuranceModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        raise NotImplementedError

    def fit(self, X, y):

        raise NotImplementedError

    def preprocess_unseen_data(self, df):

        raise NotImplementedError

    def predict(self, X):

        raise NotImplementedError
