from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class KNN(Model):
    """
    K-Nearest Neighbors (KNN) classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the KNN model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)

        if self.config.get('grid_search', False):
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }

            # Perform grid search with cross-validation
            grid = GridSearchCV(knn, param_grid, cv=10, n_jobs=-1)
            grid.fit(X, y)

            self.model = grid.best_estimator_
        else:
            # Fit the model directly without grid search
            knn.fit(X, y)

            self.model = knn

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