"""
File Name: DecisionTree.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /DecisionTree.py
File Created: Sunday, 22nd October 2023 4:14:14 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 18th January 2024 11:10:30 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2023 - 2023, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations
from functools import partial
import numpy as np
from .utils import Node, DataTypes, BaseTreeEstimator
from time import time


class DecisionTree(BaseTreeEstimator):
    def __init__(
        self,
        max_depth: int = None,
        max_features: int = None,
        min_impurity_decrease: float = 0.0,
        impurity_method: str = "gini",
        cp: float = 0.01,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.max_depth = np.iinfo(np.int32).max if max_depth is None else max_depth
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity_method = impurity_method
        self.cp = cp

        self.random_state = random_state
        self.tree = None

    def _split_data(self, X, y, random_state: int = None) -> tuple:
        n, p = X.shape
        if n < 1:
            return None

        # initialize impurity function
        _impurity = partial(self.impurity_measure, method=self.impurity_method)
        # get the impurity of the current node
        impurity_base = _impurity(y)
        # if the current node is pure, return None
        if impurity_base == 0:
            return None, None

        best_idx, best_threshold = None, None
        np.random.seed(random_state)
        feature_indices = np.random.choice(
            range(self.n_features), self.max_features, replace=False
        )

        for feature_idx in feature_indices:
            # get all possible thresholds for the current feature
            possible_thresholds = self._all_possible_thresholds(X[:, feature_idx], y)

            # if no possible thresholds, return None
            if len(possible_thresholds) == 0:
                return None, None

            for threshold in possible_thresholds:
                # split the data into left & right by the index & threshold
                left_y = y[X[:, feature_idx] < threshold]
                right_y = y[X[:, feature_idx] >= threshold]

                # calculate the impurity of the left & right nodes
                left_ratio, right_ratio = len(left_y) / n, len(right_y) / n
                impurity_current = left_ratio * _impurity(
                    left_y
                ) + right_ratio * _impurity(right_y)

                # if impurity reduced > min_impurity_decrease, update the best threshold
                if impurity_current < impurity_base - self.min_impurity_decrease:
                    best_idx, best_threshold = feature_idx, threshold
                    impurity_base = impurity_current

        return best_idx, best_threshold

    def _grow_tree(
        self,
        X,
        y,
        depth: int = 1,
        random_state: int = None,
    ) -> Node:
        sample_per_class = [np.sum(y == c) for c in range(self.n_classes)]
        predicted_class = np.argmax(sample_per_class)

        # setup node
        node = Node(predicted_class, n_samples=len(y))

        # stop if max_depth is reached
        if (self.max_depth is None) or (depth < self.max_depth):
            idx, threshold = self._split_data(X, y, random_state)
            if idx is not None:
                if random_state is not None:
                    random_state += 1
                idx_left = X[:, idx] < threshold
                X_left, y_left = X[idx_left, :], y[idx_left]
                X_right, y_right = X[~idx_left, :], y[~idx_left]

                node.feature_index = idx
                node.threshold = threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1, random_state)
                node.right = self._grow_tree(X_right, y_right, depth + 1, random_state)

        return node

    # prune the tree based on the prediction classes
    # cost complexity pruning
    def _prune_tree(
        self,
        node: Node,
        X: DataTypes,
        y: DataTypes,
    ) -> Node:
        # if the node is a leaf, return the node
        if node.left is None and node.right is None:
            return node
        # otherwise, recursively prune the tree
        else:
            idx_left = X[:, node.feature_index] < node.threshold
            # split the data into left & right by the index & threshold
            X_left, y_left = X[idx_left, :], y[idx_left]
            X_right, y_right = X[~idx_left, :], y[~idx_left]
            # recursively prune the tree on the left & right nodes
            node.left = self._prune_tree(node.left, X_left, y_left, y_left)
            node.right = self._prune_tree(node.right, X_right, y_right, y_right)
            # if both children are leaves, calculate the cost complexity
            if node.left.left is None and node.right.right is None:
                # calculate the cost complexity of current node and its children
                # misclassification of leaf nodes
                # TODO: add more cost complexity measures
                leaf_mis = self.risk(y_left, node.left.predicted_class) + self.risk(
                    y_right, node.right.predicted_class
                )
                parent_mis = self.risk(y, node.predicted_class)
                # prune the current node if cost complexity measure is reduced
                if parent_mis - leaf_mis < self.cp * self.n_observations:
                    node.left = None
                    node.right = None
                    return node
                # otherwise, return the current node
                else:
                    return node
            # otherwise, return the current node
            else:
                return node

    def fit(self, X, y) -> DecisionTree:
        self.n_classes = len(np.unique(y))
        self.n_observations = X.shape[0]
        self.n_features = X.shape[1]

        # set max_features if not set
        if self.max_features is None:
            self.max_features = self.n_features
        # if max_features is float, set as proportion
        elif 0 < self.max_features <= 1:
            self.max_features = int(self.max_features * self.n_features)

        # format data types
        X, y = self._format_types(X), self._format_types(y)

        # grow tree
        self.tree = self._grow_tree(X, y, random_state=self.random_state)
        # prune tree

    def _predict_instance(self, X) -> int:
        node = self.tree
        while node.left:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

    def predict(self, X) -> np.ndarray:
        return np.array([self._predict_instance(x) for x in X.values])
