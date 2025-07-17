"""
File Name: utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /utils.py
File Created: Tuesday, 16th January 2024 11:13:07 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 13th July 2025 11:25:28 pm
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
import graphviz
from itertools import combinations
from typing import Union, Callable
import numpy as np
from numpy import ndarray
import pandas as pd
import statsmodels.api as sm
from glum import GeneralizedLinearRegressorCV as cvglmnet
from sklearn.linear_model import LogisticRegression
from .splitter import impurity_measure, _all_possible_thresholds

DataTypes = Union[np.ndarray, pd.Series, pd.DataFrame]


def visualize_binary_tree(tree, feature_names: list, name: str = "tree"):
    def get_node_label(node):
        return str(feature_names[node.feature_index]) + "<={:.2f}".format(
            node.threshold
        )

    def get_leaf_label(node):
        if node.prediction_model:
            return "Class: {}\nSamples: {}\n{}".format(
                node.predicted_class, node.n_samples, str(node.prediction_model)
            )
        else:
            return "Class: {}\nSamples: {}".format(node.predicted_class, node.n_samples)

    dot = graphviz.Digraph("G", filename="{}.gv".format(name), format="png")
    dot.attr(size="45,80")
    dot.edge_attr.update(weight="1.2")
    dot.node_attr.update(
        nodesep="0.75", ranksep="0.75", color=None, shape="box", fontsize="15"
    )
    dot.node(str(tree), label=get_node_label(tree))

    def add_nodes_edges(node):
        if node.left and node.left.left is not None:
            dot.node(str(node.left), label=get_node_label(node.left))
            dot.edge(str(node), str(node.left), label="True")
            add_nodes_edges(node.left)
        if node.left.left is None:
            dot.node(str(node.left), label=get_leaf_label(node.left))
            dot.edge(str(node), str(node.left), label="True")
        if node.right and node.right.right is not None:
            dot.node(str(node.right), label=get_node_label(node.right))
            dot.edge(str(node), str(node.right), label="False")
            add_nodes_edges(node.right)

        if node.right.right is None:
            dot.node(str(node.right), label=get_leaf_label(node.right))
            dot.edge(str(node), str(node.right), label="False")

    add_nodes_edges(tree)
    return dot


# get all unique & sorted decision nodes of a tree
def get_all_nodes(tree) -> list:
    nodes = []

    def _exhaust_node(tree) -> None:
        # if the current node is not a leaf, continue to traverse
        if tree.left is not None and tree.left.left is not None:
            nodes.append(str(tree.feature_index) + "_{:.2f}".format(tree.threshold))
            _exhaust_node(tree.left)
        # if the child of current node is a leaf, return the path
        if tree.left is not None and tree.left.left is None:
            nodes.append(str(tree.feature_index) + "_{:.2f}".format(tree.threshold))
        if tree.right is not None and tree.right.right is not None:
            nodes.append(str(tree.feature_index) + "_{:.2f}".format(tree.threshold))
            _exhaust_node(tree.right)
        if tree.right is not None and tree.right.right is None:
            nodes.append(str(tree.feature_index) + "_{:.2f}".format(tree.threshold))

    _exhaust_node(tree)
    return sorted(set(nodes))


# get all paths from root to leaf of a tree
# return a list of sorted paths in form of "idx_threshold"
def get_all_paths(tree) -> list:

    def get_node_name(tree) -> str:
        return str(tree.feature_index) + "_{:.2f}".format(tree.threshold)

    paths = []

    def _exhaust_path(tree, path: list) -> None:
        # if the current node is not a leaf, continue to traverse
        if tree.left is not None and tree.left.left is not None:
            _exhaust_path(tree.left, [*path, get_node_name(tree)])
        # if the child of current node is a leaf, return the path
        elif tree.left is not None and tree.left.left is None:
            paths.append("_".join(sorted([*path, get_node_name(tree)])))
            return None
        if tree.right is not None and tree.right.right is not None:
            _exhaust_path(tree.right, [*path, get_node_name(tree)])
        elif tree.right is not None and tree.right.right is None:
            paths.append("_".join(sorted([*path, get_node_name(tree)])))
            return None

    _exhaust_path(tree, [])
    return paths


# get all sharing decision paths from a list of trees
def get_decision_path(trees: list, threshold: float = 1.0) -> Union[list, str]:
    # check if decision nodes are in one tree
    def _check_nodes_tree(tree, nodes: set) -> bool:
        # get all decision paths of the tree
        decision_paths = get_all_paths(tree)
        # check if all decision nodes are in any of the decision paths
        return any(all(node in path for node in nodes) for path in decision_paths)

    # check if decision nodes are in all trees
    def _check_nodes_all_trees(trees: list, nodes: set) -> bool:
        return (
            True
            if sum(_check_nodes_tree(tree, nodes) for tree in trees) / len(trees)
            >= threshold
            else False
        )

    # get all decision nodes of all trees
    # convert to set to find intersection
    all_nodes = [set(get_all_nodes(tree)) for tree in trees]
    # get common decision nodes
    # sharing_nodes = sorted(set.intersection(*all_nodes))
    # get unique counts
    unique_nodes, counts = np.unique(
        np.concatenate([list(nodes) for nodes in all_nodes]), return_counts=True
    )
    # get all nodes with percentage larger than threshold
    sharing_nodes = sorted(unique_nodes[counts / len(trees) >= threshold])
    print("Common decision nodes: {}".format(sharing_nodes))

    # if there is no common decision nodes, raise error
    if len(sharing_nodes) == 0:
        raise ValueError("No common decision nodes found.")

    # check if the common decision nodes are in all trees
    paths = []
    # loop decreasing length of common decision nodes to find the longest common decision nodes
    for leng in range(len(sharing_nodes), 0, -1):
        # if only length of 1 is found, return the common decision nodes
        if leng == 1:
            return sharing_nodes
        # otherwise, loop through all combinations of common decision nodes
        for nodes in combinations(sharing_nodes, leng):
            if _check_nodes_all_trees(trees, nodes):
                paths.append("_".join(sorted(nodes)))

        if len(paths) > 0:
            return list(set(paths))

    if len(paths) == 0:
        raise ValueError("No common decision paths found.")


# get common paths from a list of path lists
def common_paths(paths: list) -> str:
    common_path = set(paths[0])
    for s in paths[1:]:
        common_path.intersection_update(s)
    return sorted(common_path)


class Node:
    def __init__(
        self,
        predicted_class: int,
        **kwargs,
    ) -> None:
        self.predicted_class = predicted_class
        self.n_samples = kwargs.get("n_samples", 0)
        self.prediction_model = None
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class BaseEstimator:
    """
    Base Estimator for prediction models
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: DataTypes, y: DataTypes) -> BaseEstimator:
        return self

    def predict(self, X: DataTypes) -> np.ndarray:
        raise NotImplementedError


class ZeroEstimator(BaseEstimator):
    """
    Predicts all labels to be 0
    """

    def __init__(
        self,
        std_rate: float = 0.0,
    ) -> None:
        self.std_rate = std_rate
        super().__init__()

    def __str__(self) -> str:
        # return "Prediction: 0"
        return "Prediction: {:.2f}".format(self.std_rate * self._std)

    def fit(self, X: DataTypes, y: DataTypes) -> ZeroEstimator:
        self._std = np.std(y)
        return self

    def predict(self, X: DataTypes) -> np.ndarray:
        result = np.zeros(X.shape[0]) + self.std_rate * self._std
        # make sure the prediction is positive
        make_positive = lambda x: 0 if x < 1e-6 else x
        return np.array(list(map(make_positive, result)))


class MeanEstimator(BaseEstimator):
    """
    Predicts all labels to be the mean of the training labels
    """

    def __init__(
        self,
        std_rate: float = 0.0,
    ) -> None:
        self.std_rate = std_rate
        super().__init__()

    def __str__(self) -> str:
        return "Prediction: {:.2f}".format(self.predictions + self.std_rate * self._std)

    def fit(self, X: DataTypes, y: DataTypes) -> MeanEstimator:
        self.predictions = np.mean(y)
        self._std = np.std(y)
        return self

    def predict(self, X: DataTypes) -> np.ndarray:
        result = np.full(X.shape[0], self.predictions + self.std_rate * self._std)
        # make sure the prediction is positive
        make_positive = lambda x: 0 if x < 1e-6 else x
        return np.array(list(map(make_positive, result)))


class GLMEstimator(BaseEstimator):
    """
    Train GLM model
    """

    def __init__(
        self,
        family: sm.families = sm.families.Gaussian,
        std_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.family = family
        self.std_rate = std_rate
        self.kwargs = kwargs

    def __str__(self) -> str:
        return "Prediction: GLM"

    def fit(self, X: DataTypes, y: DataTypes) -> GLMEstimator:
        self.features = np.std(X, axis=0) > 0
        self._model = sm.GLM(
            y,
            sm.add_constant(X[:, self.features], has_constant="add"),
            family=self.family(),
            **self.kwargs,
        )
        self._model_result = self._model.fit()
        self._std = np.std(y)
        return self

    def predict(self, X: DataTypes) -> ndarray:
        result = (
            self._model_result.predict(
                sm.add_constant(X[:, self.features], has_constant="add")
            )
            + self.std_rate * self._std
        )
        # make sure the prediction is positive
        make_positive = lambda x: 0 if x < 1e-6 else x
        return np.array(list(map(make_positive, result)))


class GLMNetEstimator(BaseEstimator, cvglmnet):
    """
    Train GLMNet model

    To deal with highly sparse data, columns with zero variance are removed.
    """

    def __init__(
        self,
        family: str = "normal",
        std_rate: float = 0.0,
        **kwargs,
    ) -> None:
        self.std_rate = std_rate
        super(BaseEstimator, self).__init__(
            family=family,
            # solver="lbfgs",
            max_iter=1000,
            **kwargs,
        )

    def __str__(self) -> str:
        return "Prediction: {} GLMnet".format(self.family)

    def fit(self, X: DataTypes, y: DataTypes) -> GLMNetEstimator:
        self.features = np.std(X, axis=0) > 0
        super(BaseEstimator, self).fit(X[:, self.features], y)
        self._std = np.std(y)
        return self

    def predict(self, X: DataTypes) -> ndarray:
        result = (
            super(BaseEstimator, self).predict(X[:, self.features])
            + self.std_rate * self._std
        )
        # make sure the prediction is positive
        make_positive = lambda x: 0 if x < 1e-6 else x
        return np.array(list(map(make_positive, result)))


class ProbEstimator(BaseEstimator):
    """
    Return a probability of no-claim and regression on claims model
    """

    def __init__(
        self,
        classification_type: str = "soft",
        std_rate: float = 0.0,
    ) -> None:
        self.classification_type = classification_type
        self.std_rate = std_rate
        super().__init__()

    @staticmethod
    def _check_classifier(clf: Callable):
        # make sure classifier has "predict_proba" method
        if not hasattr(clf, "predict_proba"):
            raise NotImplementedError(
                "Classifier must have 'predict_proba' method. Use 'hard' classification type."
            )

    # def _soft_fit(self, X: DataTypes, y: DataTypes) -> ProbEstimator:
    #     self._check_classifier(self.classifier)

    #     # get binary response from y by checking if it is zero
    #     y_clf = (y > 1e-6).astype(int)
    #     # fit classifier
    #     self.classifier.fit(X, y_clf)
    #     # get predicted probabilities
    #     probs = self.classifier.predict_proba(X)
    #     # if probs is 2d, get the second column
    #     probs = probs[:, 1] if probs.shape[1] == 2 else probs
    #     # make sure the probability is positive
    #     probs = np.maximum(probs, 1e-6)

    #     # select non-zero claims
    #     non_zero_idx = y > 1e-6
    #     # divide response by probability to get the regression target
    #     # so that the predictions by classifier * regressor are unbiased
    #     # fit regressor
    #     self.regressor.fit(X[non_zero_idx], y[non_zero_idx] / probs[non_zero_idx])

    #     return self

    # def _hard_fit(self, X: DataTypes, y: DataTypes) -> ProbEstimator:

    #     # get binary response from y by checking if it is zero
    #     y_clf = (y > 1e-6).astype(int)
    #     # fit classifier
    #     self.classifier.fit(X, y_clf)
    #     # select non-zero claims
    #     non_zero_idx = y > 1e-6
    #     # fit regressor
    #     self.regressor.fit(X[non_zero_idx], y[non_zero_idx])

    #     return self

    def _soft_predict(self, X: DataTypes) -> np.ndarray:
        # get predicted probabilities
        probs = self.classifier.predict_proba(X)
        # if probs is 2d, get the second column
        probs = probs[:, 1] if probs.shape[1] == 2 else probs
        # make sure the probability is positive
        probs = np.maximum(probs, 1e-6)

        # predict regression
        return self.regressor.predict(sm.add_constant(X, has_constant="add")) * probs

    def _hard_predict(self, X: DataTypes) -> np.ndarray:
        # get predicted probabilities
        pred_clf = self.classifier.predict(X)
        # predict regression
        return self.regressor.predict(sm.add_constant(X, has_constant="add")) * pred_clf

    def fit(self, X: DataTypes, y: DataTypes) -> ProbEstimator:

        self._std = np.std(y)

        if self.classification_type == "soft":
            return self._soft_fit(X, y)
        elif self.classification_type == "hard":
            return self._hard_fit(X, y)
        else:
            raise ValueError("Invalid classification type.")

    def predict(self, X: DataTypes) -> np.ndarray:
        if self.classification_type == "soft":
            return self._soft_predict(X) + self.std_rate * self._std
        elif self.classification_type == "hard":
            return self._hard_predict(X) + self.std_rate * self._std
        else:
            raise ValueError("Invalid classification type.")


class ProbGLMEstimator(ProbEstimator):
    """
    A probability estimator with GLM model as regressor
    """

    def __init__(
        self,
        family: sm.families = sm.families.Gaussian,
        std_rate: float = 0.0,
        classifier: Callable = LogisticRegression(),
        classification_type: str = "hard",
        **kwargs,
    ) -> None:
        self.classifier = classifier
        self.family = family
        self.classification_type = classification_type
        self.std_rate = std_rate
        self.kwargs = kwargs
        super().__init__(
            classification_type=classification_type,
            std_rate=std_rate,
        )

    def _soft_fit(self, X: DataTypes, y: DataTypes) -> ProbGLMEstimator:
        self._check_classifier(self.classifier)

        # get binary response from y by checking if it is zero
        y_clf = (y > 1e-6).astype(int)
        # fit classifier
        self.classifier.fit(X, y_clf)
        # get predicted probabilities
        probs = self.classifier.predict_proba(X)
        # if probs is 2d, get the second column
        probs = probs[:, 1] if probs.shape[1] == 2 else probs
        # make sure the probability is positive
        probs = np.maximum(probs, 1e-6)

        # select non-zero claims
        non_zero_idx = y > 1e-6
        # divide response by probability to get the regression target
        # so that the predictions by classifier * regressor are unbiased
        # fit regressor
        self.regressor = sm.GLM(
            y[non_zero_idx] / probs[non_zero_idx],
            sm.add_constant(X[non_zero_idx], has_constant="add"),
            family=self.family(),
            **self.kwargs,
        ).fit()

        return self

    def _hard_fit(self, X: DataTypes, y: DataTypes) -> ProbGLMEstimator:
        # get binary response from y by checking if it is zero
        y_clf = (y > 1e-6).astype(int)
        # fit classifier
        self.classifier.fit(X, y_clf)
        # select non-zero claims
        non_zero_idx = y > 1e-6
        # fit regressor
        self.regressor = sm.GLM(
            y[non_zero_idx],
            sm.add_constant(X[non_zero_idx], has_constant="add"),
            family=self.family(),
            **self.kwargs,
        ).fit()

        return self


class ProbGLMNetEstimator(ProbEstimator):
    """
    A probability estimator with GLMNet model as regressor
    """

    def __init__(
        self,
        family: sm.families = sm.families.Gaussian,
        std_rate: float = 0.0,
        classifier: Callable = LogisticRegression(),
        classification_type: str = "hard",
        **kwargs,
    ) -> None:
        self.classifier = classifier
        self.family = family
        self.classification_type = classification_type
        self.std_rate = std_rate
        self.kwargs = kwargs
        super().__init__(
            classifier=classifier,
            classification_type=classification_type,
            std_rate=std_rate,
        )

    def _soft_fit(self, X: DataTypes, y: DataTypes) -> ProbGLMEstimator:
        self._check_classifier(self.classifier)

        # get binary response from y by checking if it is zero
        y_clf = (y > 1e-6).astype(int)
        # fit classifier
        self.classifier.fit(X, y_clf)
        # get predicted probabilities
        probs = self.classifier.predict_proba(X)
        # if probs is 2d, get the second column
        probs = probs[:, 1] if probs.shape[1] == 2 else probs
        # make sure the probability is positive
        probs = np.maximum(probs, 1e-6)

        # select non-zero claims
        non_zero_idx = y > 1e-6
        # divide response by probability to get the regression target
        # so that the predictions by classifier * regressor are unbiased
        self.regressor = cvglmnet(
            family=self.family,
            max_iter=1000,
            **self.kwargs,
        )
        # fit regressor
        self.regressor.fit(X[non_zero_idx], y[non_zero_idx] / probs[non_zero_idx])

        return self

    def _hard_fit(self, X: DataTypes, y: DataTypes) -> ProbGLMEstimator:

        # get binary response from y by checking if it is zero
        y_clf = (y > 1e-6).astype(int)
        # fit classifier
        self.classifier.fit(X, y_clf)
        # select non-zero claims
        non_zero_idx = y > 1e-6
        # fit regressor
        self.regressor = cvglmnet(
            family=self.family,
            max_iter=1000,
            **self.kwargs,
        )
        self.regressor.fit(X[non_zero_idx], y[non_zero_idx])

        return self


class BaseTreeEstimator(BaseEstimator):
    """
    Base Tree Estimator
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _format_types(X: DataTypes) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        else:
            return X

    @staticmethod
    def risk(
        node: Node,
        X: DataTypes,
        y_clf: DataTypes,
        y_reg: DataTypes,
        method="misclassification",
    ) -> float:
        if method == "misclassification":
            return np.sum(y_clf != node.predicted_class)
        elif method == "mae":
            return np.sum(np.abs(y_reg - node.prediction_model.predict(X)))
        elif method == "mse":
            return np.sum(np.square(y_reg - node.prediction_model.predict(X)))

    # TODO: add more impurity measures
    # @staticmethod
    # def impurity_measure(y: DataTypes, method="gini") -> float:
    #     unique_classes, counts = np.unique(y, return_counts=True)
    #     total_samples = len(y)
    #     if method == "gini":
    #         impurity = 1 - np.sum(np.square(counts / total_samples))
    #     elif method == "entropy":
    #         impurity = -np.sum(
    #             (counts / total_samples) * np.log2(counts / total_samples)
    #         )
    #     elif method == "misclassification":
    #         y_pred = np.argmax(np.bincount(y))
    #         impurity = np.sum(y != y_pred) / total_samples

    #     return impurity

    @staticmethod
    def impurity_measure(y: DataTypes, method="gini") -> float:
        return impurity_measure(y, method=method)

    # @staticmethod
    # def _all_possible_thresholds(
    #     X: DataTypes, y: DataTypes, max_thresholds: int = 10
    # ) -> np.ndarray:
    #     # not-nan sorted & unique feature values
    #     nanunique = lambda arr: sorted(np.unique(arr[~np.isnan(arr)]))
    #     # not-nan sort
    #     nansort = lambda arr: np.sort(arr[~np.isnan(arr)])
    #     # not-nan argsort
    #     nanargsort = lambda arr1, arr2: arr2[~np.isnan(arr1)][
    #         np.argsort(arr1[~np.isnan(arr1)])
    #     ]
    #     # get sorted & unique feature values
    #     feature_values = nanunique(X)
    #     # if the number of unique values is less than the threshold, return all possible thresholds
    #     if len(feature_values) <= max_thresholds:
    #         return np.array(
    #             [np.mean([i, j]) for i, j in zip(feature_values, feature_values[1:])]
    #         )
    #     # otherwise, return thresholds that have impact on the impurity
    #     else:
    #         X_sort = nansort(X)
    #         y_sort = nanargsort(X, y)
    #         return np.array(
    #             [
    #                 np.mean([X_sort[i], X_sort[i + 1]])
    #                 for i in range(len(X_sort) - 1)
    #                 if y_sort[i] != y_sort[i + 1]
    #             ]
    #         )

    @staticmethod
    def _all_possible_thresholds(
        X: DataTypes, y: DataTypes, max_thresholds: int = 10
    ) -> np.ndarray:
        return _all_possible_thresholds(X, y, max_thresholds=max_thresholds)


# Functions of validation measures
import random


def giniTest(y, py):
    data = pd.DataFrame({"y": y, "py": py})
    random.seed(1)
    n = len(y)
    data["rand_unif"] = np.random.uniform(0.0, 1.0, size=len(y))
    sorted_y = data.iloc[np.argsort(data.py, data.rand_unif), 0]
    i = np.array(range(1, n + 1))
    giniIndex = 1 - 2 / (n - 1) * (n - sum(sorted_y * i) / sum(sorted_y))
    return giniIndex


def me(y, py):
    ME = None
    ME = sum(y - py) / len(y)
    return ME


def mae(y, py):
    MAE = None
    MAE = sum(abs(y - py)) / len(y)
    return MAE


def mse(y, py):
    MSE = None
    MSE = sum((y - py) ** 2) / len(y)
    return MSE


def rmse(y, py):
    RMSE = None
    RMSE = (sum((y - py) ** 2) / len(y)) ** 0.5
    return RMSE


def pe(y, py):
    PE = None
    PE = (sum(y) - sum(py)) / sum(y)
    return PE


def ccc(y, py):
    CCC = None
    CCC = (
        2
        * np.corrcoef(py, y)[0, 1]
        * np.std(y, ddof=1)
        * np.std(py, ddof=1)
        / (
            np.var(y, ddof=1)
            + np.var(py, ddof=1)
            + (sum(y) / len(y) - sum(py) / len(py)) ** 2
        )
    )
    return CCC


def r2(y, py):
    r2 = None
    r2 = 1 - sum((py - y) ** 2) / sum((y - np.mean(y)) ** 2)
    return r2


# higher GINI is better
# higher R2 is better
# higher CCC is better
# lower MSE is better
# lower RMSE is better
# lower |ME| is better
# lower |PE| is better
# lower MAE is better


def PrintPredictionError(y, py):
    print("GINI:", giniTest(y, py))
    print("R2:", r2(y, py))
    print("CCC:", ccc(y, py))
    print("MSE:", mse(y, py))
    print("RMSE:", rmse(y, py))
    print("ME:", me(y, py))
    print("PE:", pe(y, py))
    print("MAE:", mae(y, py))


def StorePredictionError(y, py):
    temp_vals = [
        round(giniTest(y, py), 4),
        round(r2(y, py), 4),
        round(ccc(y, py), 4),
        round(mse(y, py), 4),
        round(rmse(y, py), 4),
        round(me(y, py), 4),
        round(pe(y, py), 4),
        round(mae(y, py), 4),
    ]
    return temp_vals
