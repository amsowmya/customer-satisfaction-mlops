import logging 
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): _description_
            y_train (pd.Series): _description_
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e