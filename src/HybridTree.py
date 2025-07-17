"""
File Name: HybridTree.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /HybridTree.py
File Created: Tuesday, 16th January 2024 11:12:42 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 12th July 2025 5:26:47 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2024 - 2024, Panyi Dong

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
import pandas as pd
import statsmodels.api as sm
from .utils import (
    DataTypes,
    Node,
    BaseTreeEstimator,
    ZeroEstimator,
    MeanEstimator,
    GLMEstimator,
    GLMNetEstimator,
    ProbGLMEstimator,
    ProbGLMNetEstimator,
)


class HybridTree(BaseTreeEstimator):

    GLM_estimator = {
        "GLMNet": GLMNetEstimator,
        "GLM": GLMEstimator,
        "ProGLMNet": ProbGLMNetEstimator,
        "ProGLM": ProbGLMEstimator,
    }
    default_GLM_family = {
        "GLMNet": "gaussian",
        "GLM": sm.families.Gaussian,
        "ProGLMNet": "gaussian",
        "ProGLM": sm.families.Gaussian,
    }

    def __init__(
        self,
        max_depth: int = None,
        max_features: int = None,
        min_impurity_decrease: float = 0.0,
        min_samples_split: int = 2,
        impurity_method: str = "gini",
        cp: float = 0.0001,
        pruning_criterion: str = "misclassification",
        GLM_type: str = "GLM",
        GLM_family: str = None,
        GLM_zero_threshold: float = 0.90,
        GLM_min_samples_leaf: int = 40,
        zero_std_rate: float = 0.0,
        mean_std_rate: float = 0.0,
        GLM_std_rate: float = 0.0,
        random_state: int = None,
        **GLM_kwargs,
    ) -> None:
        super().__init__()
        self.max_depth = np.iinfo(np.int32).max if max_depth is None else max_depth
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.impurity_method = impurity_method
        self.cp = cp
        self.pruning_criterion = pruning_criterion
        self.GLM_type = GLM_type
        self.GLM_family = (
            self.default_GLM_family[GLM_type] if GLM_family is None else GLM_family
        )
        self.GLM_zero_threshold = GLM_zero_threshold
        self.GLM_min_samples_leaf = GLM_min_samples_leaf
        self.zero_std_rate = zero_std_rate
        self.mean_std_rate = mean_std_rate
        self.GLM_std_rate = GLM_std_rate
        self.GLM_kwargs = GLM_kwargs

        self.random_state = random_state
        self.tree = None

    def _check_input(self, X: DataTypes, y_clf: DataTypes, y_reg: DataTypes) -> tuple:
        X, y_clf, y_reg = (
            self._format_types(X),
            self._format_types(y_clf),
            self._format_types(y_reg),
        )
        if len(y_clf.shape) > 1:
            y_clf = y_clf.flatten()
        if len(y_reg.shape) > 1:
            y_reg = y_reg.flatten()

        return X, y_clf, y_reg

    def get_model(
        self,
        y: DataTypes,
    ):
        # if the proportion of zeros is greater than the threshold, return ZeroEstimator
        if sum(y < 1e-6) / len(y) >= self.GLM_zero_threshold:
            return ZeroEstimator(std_rate=self.zero_std_rate)
        # if the number of samples is less than the threshold, return MeanEstimator
        elif len(y) <= self.GLM_min_samples_leaf:
            return MeanEstimator(std_rate=self.mean_std_rate)
        # otherwise, return GLMNetEstimator
        else:
            return self.GLM_estimator[self.GLM_type](
                self.GLM_family, self.GLM_std_rate, **self.GLM_kwargs
            )

    def _fit_reg_model(self, X: DataTypes, y_clf: DataTypes, y_reg: DataTypes):
        _model = self.get_model(
            y_clf,
        )
        _model.fit(X, y_reg)
        return _model

    def _split_data(
        self, X: DataTypes, y: DataTypes, random_state: int = None
    ) -> tuple:
        n, p = X.shape
        if n < 1:
            return None, None

        # initialize impurity function
        _impurity = partial(self.impurity_measure, method=self.impurity_method)
        # get the impurity of the current node
        impurity_base = _impurity(y)
        # if the current node is pure, return None
        if impurity_base == 0:
            return None, None

        # initialize best feature index, threshold & feature indices
        best_idx, best_threshold = None, None
        np.random.seed(random_state)
        feature_indices = (
            np.random.choice(range(self.n_features), self.max_features, replace=False)
            if self.max_features < self.n_features
            else range(self.n_features)
        )
        for feature_idx in feature_indices:
            # if only one unique value, skip the current feature
            if len(set(X[:, feature_idx])) == 1:
                continue

            # get all possible thresholds for the current feature
            possible_thresholds = self._all_possible_thresholds(X[:, feature_idx], y)

            for threshold in possible_thresholds:
                # split the data into left & right by the threshold
                idx_left = X[:, feature_idx] < threshold
                left_y = y[idx_left]
                right_y = y[~idx_left]

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
        X: DataTypes,
        y_clf: DataTypes,
        y_reg: DataTypes,
        depth: int = 1,
        random_state: int = None,
    ) -> Node:
        sample_per_class = [np.sum(y_clf == c) for c in range(self.n_classes)]
        predicted_class = np.argmax(sample_per_class)

        # initialize node
        node = Node(predicted_class, n_samples=len(y_clf))
        # initialize idx
        idx = None
        # max depth criterion
        depth_criterion = (self.max_depth is None) or (depth < self.max_depth)
        # min samples split criterion
        split_criterion = len(y_clf) >= self.min_samples_split

        # stop if max_depth is reached
        if depth_criterion and split_criterion:
            idx, threshold = self._split_data(X, y_clf, random_state)
            # continue splitting if stopping criterion is not met
            if idx is not None:
                # if idx > 0:
                # update random_state
                if random_state is not None:
                    random_state += 1
                # split the data into left & right by the index & threshold
                idx_left = X[:, idx] < threshold
                X_left, y_clf_left, y_reg_left = (
                    X[idx_left, :],
                    y_clf[idx_left],
                    y_reg[idx_left],
                )
                X_right, y_clf_right, y_reg_right = (
                    X[~idx_left, :],
                    y_clf[~idx_left],
                    y_reg[~idx_left],
                )

                # assign the feature index & threshold to the node
                node.feature_index = idx
                node.threshold = threshold
                # recursively grow the tree on the left & right nodes
                node.left = self._grow_tree(
                    X_left, y_clf_left, y_reg_left, depth + 1, random_state
                )
                node.right = self._grow_tree(
                    X_right, y_clf_right, y_reg_right, depth + 1, random_state
                )
        # train corresponding regression estimator if stopping criterion is met
        if depth >= self.max_depth or idx is None:
            # if depth >= self.max_depth or idx < 0:
            node.prediction_model = self._fit_reg_model(X, y_clf, y_reg)

        return node

    # prune the tree based on the prediction classes
    # cost complexity pruning
    def _prune_tree(
        self, node: Node, X: DataTypes, y_clf: DataTypes, y_reg: DataTypes
    ) -> Node:
        # if the node is a leaf, return the node
        if node.left is None and node.right is None:
            return node
        # otherwise, recursively prune the tree
        else:
            idx_left = X[:, node.feature_index] < node.threshold
            # split the data into left & right by the index & threshold
            X_left, y_clf_left, y_reg_left = (
                X[idx_left, :],
                y_clf[idx_left],
                y_reg[idx_left],
            )
            X_right, y_clf_right, y_reg_right = (
                X[~idx_left, :],
                y_clf[~idx_left],
                y_reg[~idx_left],
            )
            # recursively prune the tree on the left & right nodes
            node.left = self._prune_tree(node.left, X_left, y_clf_left, y_reg_left)
            node.right = self._prune_tree(node.right, X_right, y_clf_right, y_reg_right)
            # if both children are leaves, calculate the cost complexity
            if node.left.left is None and node.right.right is None:
                # calculate the cost complexity of current node and its children
                # misclassification of leaf nodes
                # TODO: add more cost complexity measures
                leaf_cost = self.risk(
                    node.left, X_left, y_clf_left, y_reg_left, self.pruning_criterion
                ) + self.risk(
                    node.right,
                    X_right,
                    y_clf_right,
                    y_reg_right,
                    self.pruning_criterion,
                )
                # temporarily train a new prediction model at parent node
                node.prediction_model = self._fit_reg_model(X, y_clf, y_reg)
                parent_cost = self.risk(node, X, y_clf, y_reg, self.pruning_criterion)
                # prune the current node if cost complexity measure is reduced
                if parent_cost - leaf_cost < self.cp * self.n_observations:
                    node.left = None
                    node.right = None
                    return node
                # otherwise, return the current node
                else:
                    # remove the temporary prediction model
                    node.prediction_model = None
                    return node
            # otherwise, return the current node
            else:
                return node

    # y_clf is the classification label for tree growing
    # y_reg is the regression label for prediction
    def fit(self, X: DataTypes, y_clf: DataTypes, y_reg: DataTypes) -> HybridTree:
        self.n_classes = len(np.unique(y_reg))
        self.n_observations = X.shape[0]
        self.n_features = X.shape[1]

        # set max_features if not set
        if self.max_features is None:
            self.max_features = self.n_features
        # if max_features is float, set as proportion
        elif 0 < self.max_features <= 1:
            self.max_features = int(self.max_features * self.n_features)

        # if y_clf is not provided, binarized y_reg as y_clf
        y_clf = (y_reg < 1e-6).astype(int) if y_clf is None else y_clf

        # format data types
        X, y_clf, y_reg = self._check_input(X, y_clf, y_reg)

        # grow the tree
        self.tree = self._grow_tree(X, y_clf, y_reg, random_state=self.random_state)
        # prune the tree if cp > 0
        if self.cp > 0:
            self.tree = self._prune_tree(self.tree, X, y_clf, y_reg)

        return self

    def _predict_instance(self, X: DataTypes) -> float:
        # initialize node
        node = self.tree
        # traverse the tree until reaching a leaf node
        while node.left:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction_model.predict(X.reshape(1, -1))

    def predict(self, X: DataTypes) -> np.ndarray:
        X = self._format_types(X)
        return np.array([self._predict_instance(x) for x in X])
