from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class QDA(Model):
    """
    Quadratic Discriminant Analysis (QDA) classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the QDA model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the QDA model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the QDA model
        qda = QuadraticDiscriminantAnalysis()

        # Fit the model directly
        qda.fit(X, y)

        self.model = qda
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict labels for the given feature matrix.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X) if self.model else None
