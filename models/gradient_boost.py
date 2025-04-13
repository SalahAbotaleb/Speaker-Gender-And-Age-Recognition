from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoost(Model):
    """
    Gradient Boosting classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the Gradient Boosting model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Gradient Boosting model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the Gradient Boosting model
        gbc = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )

        if self.config.get('grid_search', False):
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }

            # Perform grid search with cross-validation
            grid = GridSearchCV(gbc, param_grid, cv=10, n_jobs=-1)
            grid.fit(X, y)

            self.model = grid.best_estimator_
        else:
            # Fit the model directly without grid search
            gbc.fit(X, y)

            self.model = gbc

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