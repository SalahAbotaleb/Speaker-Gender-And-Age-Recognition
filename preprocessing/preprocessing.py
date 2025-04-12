from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor class to handle missing values and categorical encoding.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Subclasses must implement this method.")