from .model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVM(Model):
    """
    Support Vector Machine (SVM) classifier.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = None  # Placeholder for the SVM model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """

        # Initialize the SVC model
        svm = SVC(kernel='rbf', C=1000, gamma=0.0001, random_state=42)

        if self.config.get('grid_search', False):
        # Define the parameter grid for hyperparameter tuning
          param_grid = {
              'C': [1, 10, 100, 1000],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'poly']
          }

          # Perform grid search with cross-validation
          grid = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1)
          grid.fit(X, y)

          self.model = grid.best_estimator_
        else: 
          # Fit the model directly without grid search
          svm.fit(X, y)

          self.model = svm
        
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