from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

class ExtraTree(Model):
    """
    Extra Trees Classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the Extra Trees model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Extra Trees model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the Extra Trees Classifier
        etc = ExtraTreesClassifier(n_estimators=100, random_state=42)

        if self.config.get('grid_search', False):
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Perform grid search with cross-validation
            grid = GridSearchCV(etc, param_grid, cv=10, n_jobs=-1)
            grid.fit(X, y)

            self.model = grid.best_estimator_
        else:
            # Fit the model directly without grid search
            etc.fit(X, y)

            self.model = etc

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