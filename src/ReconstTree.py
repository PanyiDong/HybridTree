"""
File Name: ReconstTree.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /ReconstTree.py
File Created: Monday, 9th September 2024 1:58:13 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th July 2025 11:12:14 am
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
from typing import List, Dict
from functools import partial
import numpy as np
import pandas as pd
from .utils import DataTypes, Node, MeanEstimator

from .HybridTree import HybridTree


def _convert_split_rules(split_rules: List) -> List:
    rules = []
    for i, rule in enumerate(split_rules):
        pairs = rule.split("_")
        # in case more than 1 pair
        for feature, threshold in zip(pairs[::2], pairs[1::2]):
            rules.append((int(feature), float(threshold)))
    return list(set(rules))


class ReconstTree(HybridTree):
    """
    Using a set of feature+threshold pairs to reconstruct best-performing hybrid tree
    """

    def __init__(
        self,
        split_rules: List,
        **kwargs,
    ) -> None:
        self.split_rules = _convert_split_rules(split_rules)
        super().__init__(**kwargs)

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

        for feature_idx, threshold in self.split_rules:
            # split the data into left & right by the threshold
            idx_left = X[:, feature_idx] < threshold
            left_y = y[idx_left]
            right_y = y[~idx_left]

            # calculate the impurity of the left & right nodes
            left_ratio, right_ratio = len(left_y) / n, len(right_y) / n
            impurity_current = left_ratio * _impurity(left_y) + right_ratio * _impurity(
                right_y
            )

            # if impurity reduced > min_impurity_decrease, update the best threshold
            if impurity_current < impurity_base - self.min_impurity_decrease:
                best_idx, best_threshold = feature_idx, threshold
                impurity_base = impurity_current

        return best_idx, best_threshold
