"""
File Name: splitter.pyx
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /splitter.pyx
File Created: Wednesday, 17th January 2024 9:32:25 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 30th January 2024 1:44:32 pm
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

from typing import Callable, Union
from functools import partial
import numpy as np # Normal NumPy import
cimport numpy as cnp # Import for NumPY C-APIfrom 
cnp.import_array()

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

# TODO: add more impurity measures
def impurity_measure(y: np.ndarray, method: str ="gini") -> float:
    cdef cnp.ndarray[cnp.int_t, ndim=1] unique_classes
    cdef cnp.ndarray[cnp.int_t, ndim=1] counts
    cdef int total_samples
    cdef float impurity
    
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    if method == "gini":
        impurity = 1 - np.sum(np.square(counts / total_samples))
    elif method == "entropy":
        impurity = -np.sum(
            (counts / total_samples) * np.log2(counts / total_samples)
        )
    elif method == "misclassification":
        y_pred = np.argmax(np.bincount(y))
        impurity = np.sum(y != y_pred) / total_samples

    return impurity

def _all_possible_thresholds(
    X: np.ndarray, y: np.ndarray, max_thresholds: int = 10
) -> np.ndarray:
    cdef cnp.ndarray[cnp.float_t, ndim=1] feature_values
    cdef cnp.ndarray[cnp.float_t, ndim=1] X_sort
    cdef cnp.ndarray[cnp.int_t, ndim=1] y_sort
    
    # NOTE: no need to handle missing values
    # # not-nan sorted & unique feature values
    # nanunique = lambda arr: np.sort(np.unique(arr[~np.isnan(arr)]))
    # # not-nan sort
    # nansort = lambda arr: np.sort(arr[~np.isnan(arr)])
    # # not-nan argsort arr1 by arr2
    # nanargsort = lambda arr1, arr2: arr2[~np.isnan(arr1)][
    #     np.argsort(arr1[~np.isnan(arr1)])
    # ]
    # get sorted & unique feature values
    # feature_values = nanunique(X)
    feature_values = np.sort(np.unique(X))
    # if the number of unique values is less than the threshold, return all possible thresholds
    if len(feature_values) <= max_thresholds:
        return np.array(
            [np.mean([i, j]) for i, j in zip(feature_values, feature_values[1:])]
        )
    # otherwise, return thresholds that have impact on the impurity
    # only when both features & labels change
    else:
        # X_sort = nansort(X)
        # y_sort = nanargsort(X, y)
        X_sort = np.sort(X)
        y_sort = y[np.argsort(X)]
        return np.unique(
            [
                np.mean([X_sort[i], X_sort[i + 1]])
                for i in range(len(X_sort) - 1)
                if (y_sort[i] != y_sort[i + 1]) & (X_sort[i] != X_sort[i + 1])
            ]
        )

cpdef tuple splitter(
    X: np.ndarray, 
    y: np.ndarray, 
    n_features: int,
    max_features: int,
    min_impurity_decrease: float = 0.0,
    impurity_method: str = "gini",
    random_state: int = None,
) :
    # define variable types
    cdef int n, p, best_idx, feature_idx
    cdef float impurity_base, impurity_current, best_threshold, threshold, left_ratio, right_ratio
    cdef cnp.ndarray[cnp.int_t, ndim=1] feature_indices
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] idx_left
    cdef cnp.ndarray[cnp.float_t, ndim=1] possible_thresholds
    cdef cnp.ndarray[cnp.int_t, ndim=1] left_y, right_y
    # cdef Callable _impurity
    
    # get the number of samples & features
    n, p = X.shape
    # if no samples, return None
    if n < 1:
        return -1, -1.0

    # initialize impurity function
    _impurity = partial(impurity_measure, method=impurity_method)
    # get the impurity of the current node
    impurity_base = _impurity(y)
    # if the current node is pure, return None
    if impurity_base == 0:
        return -1, -1.0

    # initialize best feature index, threshold & feature indices
    best_idx, best_threshold = -1, -1.0
    np.random.seed(random_state)
    feature_indices = np.random.choice(
        range(n_features), max_features, replace=False
    )
    for feature_idx in feature_indices:
        # if only one unique value, skip the current feature
        if len(set(X[:, feature_idx])) == 1:
            continue

        # get all possible thresholds for the current feature
        possible_thresholds = _all_possible_thresholds(X[:, feature_idx], y)

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
            if impurity_current < impurity_base - min_impurity_decrease:
                best_idx, best_threshold = feature_idx, threshold
                impurity_base = impurity_current

    return best_idx, best_threshold

def _grow_tree(
    X: np.ndarray,
    y_clf: np.ndarray,
    y_reg: np.ndarray,
    _fit_reg_model: Callable,
    n_features: int,
    max_features: int,
    max_depth: int = None,
    min_samples_split: int = 2,
    n_classes: int = 2,
    depth: int = 1,
    min_impurity_decrease: float = 0.0,
    impurity_method: str = "gini",
    random_state: int = None,
) -> Node:
    cdef int predicted_class, idx
    cdef float threshold
    cdef cnp.npy_bool depth_criterion, split_criterion
    cdef cnp.ndarray[cnp.int_t, ndim=1] sample_per_class, y_clf_left, y_clf_right
    cdef cnp.ndarray[cnp.float_t, ndim=2] X_left, X_right
    cdef cnp.ndarray[cnp.float_t, ndim=1] y_reg_left, y_reg_right
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] idx_left
    
    sample_per_class = np.array([np.sum(y_clf == c) for c in range(n_classes)])
    predicted_class = np.argmax(sample_per_class)

    # initialize node
    node = Node(predicted_class, n_samples=len(y_clf))
    # initialize idx
    idx = -1
    # max depth criterion
    depth_criterion = (max_depth is None) or (depth < max_depth)
    # min samples split criterion
    split_criterion = len(y_clf) >= min_samples_split

    # stop if max_depth is reached
    if depth_criterion and split_criterion:
        idx, threshold = splitter(
            X, y_clf, n_features, max_features, min_impurity_decrease,impurity_method, random_state
        )
        # continue splitting if stopping criterion is not met
        if idx > 0:
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
            node.left = _grow_tree(
                X_left, y_clf_left, y_reg_left, _fit_reg_model, n_features,
                max_features, max_depth, min_samples_split, n_classes, depth, 
                min_impurity_decrease, impurity_method, random_state,
            )
            node.right = _grow_tree(
                X_right, y_clf_right, y_reg_right, _fit_reg_model, n_features,
                max_features, max_depth, min_samples_split, n_classes, depth, 
                min_impurity_decrease, impurity_method, random_state,
            )
    # train corresponding regression estimator if stopping criterion is met
    if depth >= max_depth or idx < 0:
        node.prediction_model = _fit_reg_model(X, y_clf, y_reg)

    return node