"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

np.random.seed(42)

class Node():
    def __init__(self, feature=None, threshold=None, info_gain=None, value = None,left=None,right=None):

      self.feature = feature
      self.threshold = threshold
      self.info_gain = info_gain
      self.children = {}
      self.value = value
      self.split_pt= None
      self.left = left
      self.right = right


class DecisionTree:
    def __init__(self, criterion, max_depth=5):  # Fixed __init__ typo
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = list(X.columns)  # Ensure features is a list
        self.root = self._grow_tree(X, y, features, depth=0)

    def _grow_tree(self, X, y, features, depth):
        if len(set(y)) == 1 or depth == self.max_depth or len(features) == 0:
            return Node(value=y.mode()[0] if not check_ifreal(y) else y.mean())
        
        best_feature = opt_split_attribute(X, y, self.criterion, features)
        if check_ifreal(X[best_feature]):
            threshold = X[best_feature].median()
            X_left, y_left, X_right, y_right = split_data(X, y, best_feature, threshold)
        else:
            threshold = None
            X_left, y_left, X_right, y_right = split_data(X, y, best_feature, y.mode()[0])
        
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value=y.mode()[0] if not check_ifreal(y) else y.mean())
        
        remaining_features = [f for f in features if f != best_feature]  # Corrected feature removal
        left_child = self._grow_tree(X_left, y_left, remaining_features, depth + 1)
        right_child = self._grow_tree(X_right, y_right, remaining_features, depth + 1)
        return Node(feature=best_feature, threshold=threshold, left=left_child, right=right_child)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._traverse_tree, axis=1, args=(self.root,))
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if node.threshold is None:
            if x[node.feature] == 1:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        
        if check_ifreal(x[node.feature]):
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def plot(self, node=None, indent=""):
        if node is None:
            node = self.root
        
        if node.value is not None:
            print(indent + "Leaf:", node.value)
        else:
            condition = f"({node.feature} <= {node.threshold})" if node.threshold else f"({node.feature})"
            print(indent + f"? {condition}")
            print(indent + "Y:", end=" ")
            self.plot(node.left, indent + "    ")
            print(indent + "N:", end=" ")
            self.plot(node.right, indent + "    ")




