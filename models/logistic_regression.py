from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(Model):
    """
    Logistic Regression classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the Logistic Regression model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Logistic Regression model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the Logistic Regression model
        lr = LogisticRegression(max_iter=1000, random_state=42)

        if self.config.get('grid_search', False):
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'saga']
            }

            # Perform grid search with cross-validation
            grid = GridSearchCV(lr, param_grid, cv=10, n_jobs=-1)
            grid.fit(X, y)

            self.model = grid.best_estimator_
        else:
            # Fit the model directly without grid search
            lr.fit(X, y)

            self.model = lr

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
