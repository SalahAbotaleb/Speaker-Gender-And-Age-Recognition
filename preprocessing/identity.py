from .preprocessing import Preprocessor

class Identity(Preprocessor):
    """
    Identity Preprocessor.
    This preprocessor does not perform any transformation on the input data.
    It is useful when you want to bypass preprocessing for certain datasets or tasks.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X