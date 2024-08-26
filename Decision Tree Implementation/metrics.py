from typing import Union
import numpy as np
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    return np.sum(y_hat == y)/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # assert y_hat.size == y.size
    # cls_series = pd.Series([cls] * len(y_hat))
    # true_positive = np.sum((y == y_hat) & (y == cls_series))
    # true_predicted = np.sum(y_hat == cls_series)
    # prec = float(true_positive / true_predicted) if true_predicted > 0 else 0.0
    # return prec

    assert y_hat.size == y.size, "Size of y_hat and y must be the same"
    
    # Align the indices
    y, y_hat = y.align(y_hat)
    
    # Create a boolean Series for the class
    cls_series = pd.Series([cls] * len(y_hat), index=y_hat.index)
    
    # Calculate true positives and true predicted
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_predicted = np.sum(y_hat == cls_series)
    
    # Calculate precision
    prec = float(true_positive / true_predicted) if true_predicted > 0 else 0.0
    
    return prec

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    y, y_hat = y.align(y_hat)
    cls_series = pd.Series([cls] * len(y_hat), index = y_hat.index)
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_actual = np.sum(y == cls_series)
    rec = float(true_positive / true_actual) if true_actual > 0 else 0.0
    return rec


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    return np.sqrt(np.mean((y_hat - y)**2))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return np.mean(np.abs(y_hat - y))