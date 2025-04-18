import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam  # you can choose your optimizer
from sklearn.base import BaseEstimator, ClassifierMixin

class NN(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_dim=None,
                 hidden_units=64,
                 hidden_activation='relu',
                 output_activation='softmax',  # Use 'softmax' for multi-class output
                 optimizer='adam',
                 loss='sparse_categorical_crossentropy',  # using integer labels
                 epochs=10,
                 batch_size=32,
                 verbose=1):
        # Parameters for constructing the network:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        # This will hold the compiled Keras model after fit is called:
        self.model_ = None

    def _build_model(self, X):
        # Determine the number of input features from X if not provided
        input_dim = self.input_dim if self.input_dim is not None else X.shape[1]
        model = Sequential()
        # Hidden layer: you can add dropout or more layers as needed
        model.add(Dense(self.hidden_units, activation=self.hidden_activation, input_dim=input_dim))
        # Output layer with 4 units for 4 classes
        model.add(Dense(4, activation=self.output_activation))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def fit(self, X, y):
        """
        Fit the neural network model.
        Parameters:
          X : array-like of shape (n_samples, n_features)
              Training data.
          y : array-like of shape (n_samples,)
              Target class labels as integers (0, 1, 2, or 3).
        Returns:
          self : object
              Returns self.
        """
        # Build the model if not already built
        self.model_ = self._build_model(X)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Returns:
          predictions : array-like of shape (n_samples,)
              The predicted class label for each sample.
        """
        if self.model_ is None:
            raise ValueError("The model has not been fitted yet. Please call fit() first.")
        # Get predicted probabilities
        pred_probs = self.model_.predict(X, batch_size=self.batch_size, verbose=self.verbose)
        # Return the index (i.e. class label) with the highest probability for each sample
        return np.argmax(pred_probs, axis=1)

    def predict_proba(self, X):
        """
        Predict probability estimates for samples in X.
        Returns:
          probabilities : array-like of shape (n_samples, n_classes)
              The class probability estimates.
        """
        if self.model_ is None:
            raise ValueError("The model has not been fitted yet. Please call fit() first.")
        return self.model_.predict(X, batch_size=self.batch_size, verbose=self.verbose)
