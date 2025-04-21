from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
import numpy as np

class GenderAgePipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, gender_model=None, male_age_model=None, female_age_model=None):
        self.gender_model = gender_model
        self.male_age_model = male_age_model
        self.female_age_model = female_age_model

    def fit(self, X, y):
        gender = y % 2
        age = y // 2

        self.gender_model_ = clone(self.gender_model)
        self.male_age_model_ = clone(self.male_age_model)
        self.female_age_model_ = clone(self.female_age_model)

        self.gender_model_.fit(X, gender)

        self.male_age_model_.fit(X[gender == 0], age[gender == 0])
        self.female_age_model_.fit(X[gender == 1], age[gender == 1])

        return self

    def predict(self, X):
        predicted_gender = self.gender_model_.predict(X)

        male_mask = predicted_gender == 0
        female_mask = ~male_mask

        age_preds = np.empty_like(predicted_gender, dtype=float)
        if np.any(male_mask):
            age_preds[male_mask] = self.male_age_model_.predict(X[male_mask])
        if np.any(female_mask):
            age_preds[female_mask] = self.female_age_model_.predict(X[female_mask])

        final_preds = age_preds * 2 + predicted_gender
        return final_preds.astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

    def get_params(self, deep=True):
        return {
            'gender_model': self.gender_model,
            'male_age_model': self.male_age_model,
            'female_age_model': self.female_age_model
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
