"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, info_gain=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.value = value
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = list(X.columns)
        self.root = self._grow_tree(X, y,features, depth=0)

    def _grow_tree(self, X, y,features, depth):
        
        if len(y.unique()) == 1 or depth == self.max_depth or X.shape[1] == 0:
            leaf_value = y.mode()[0] if not check_ifreal(y) else y.mean()
            return Node(value=leaf_value)

        best_feature = opt_split_attribute(X, y, self.criterion,features)

        if best_feature is None:
            leaf_value = y.mode()[0] if not check_ifreal(y) else y.mean()
            return Node(value=leaf_value)
        
        if check_ifreal(X[best_feature]):
            threshold = X[best_feature].median()
            left_idxs = X[best_feature] <= threshold
        else:
            threshold = None
            left_idxs = X[best_feature] == X[best_feature].mode()[0]

        X_left, y_left = X[left_idxs], y[left_idxs]
        X_right, y_right = X[~left_idxs], y[~left_idxs]

        if X_left.empty or X_right.empty:
            leaf_value = y.mode()[0] if not check_ifreal(y) else y.mean()
            return Node(value=leaf_value)

        remaining_features = [f for f in features if f != best_feature]
        left_child = self._grow_tree(X_left, y_left,remaining_features, depth + 1)
        right_child = self._grow_tree(X_right, y_right,remaining_features, depth + 1)
        return Node(feature=best_feature, threshold=threshold, left=left_child, right=right_child)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._traverse, axis=1)

    def _traverse(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        if node.threshold is None:
            if x[node.feature] == 1:
                return self._traverse(x, node.left)
            else:
                return self._traverse(x, node.right)

        if check_ifreal(x[node.feature]):
            if x[node.feature] <= node.threshold:
                return self._traverse(x, node.left)
            else:
                return self._traverse(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse(x, node.left)
            else:
                return self._traverse(x, node.right)

    def plot(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{indent}Leaf: {node.value}")
        else:
            condition = f"({node.feature} <= {node.threshold})" if node.threshold else f"({node.feature})"
            print(f"{indent}? {condition}")
            print(f"{indent}Yes ->", end=" ")
            self.plot(node.left, indent + "    ")
            print(f"{indent}No  ->", end=" ")
            self.plot(node.right, indent + "    ")





