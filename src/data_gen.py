"""
File Name: data_gen.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: src
Latest Version: <<projectversion>>
Relative Path: /data_gen.py
File Created: Tuesday, 16th January 2024 1:47:45 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 29th August 2025 1:24:13 pm
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

import numpy as np
import pandas as pd


def data_gen(nSample, nRealCatX, nFakeCatX, nRealConX, nFakeConx, pho, seed=42):
    np.random.seed(seed)

    # shape of data
    n = nSample  # number of observations
    p = nRealCatX + nFakeCatX + nRealConX + nFakeConx  # number of features
    p_Con = nRealConX + nFakeConx  # number of continuous features
    p_Cat = nRealCatX + nFakeCatX  # number of categorical features
    pho = float(pho)

    # generate continuous features
    # covariance matrix of continuous features
    Cov_Con = pho ** np.abs(
        [[row - col for col in range(p_Con)] for row in range(p_Con)]
    )
    mean_Con = np.zeros(p_Con)
    xCon = np.random.multivariate_normal(mean_Con, Cov_Con, size=n)  # .T

    # generate categorical features
    xCat = np.random.choice(
        [-3, -2, 1, 4], size=(n, p_Cat), replace=True, p=[0.25, 0.25, 0.25, 0.25]
    )

    # Poisson variables
    mu_poi = (
        -0.1
        # + 0.5 * np.sum(xCon[:, : int(nRealConX / 2)], axis=1)
        # - 0.5 * np.sum(xCat[:, : int(nRealCatX / 2)], axis=1)
        # - 0.1 * np.sum(xCon[:, (int(nRealConX / 2) + 1) : nRealConX], axis=1)
        # - 0.1 * np.sum(xCat[:, (int(nRealCatX / 2) + 1) : nRealCatX], axis=1)
        + np.sum(
            pd.DataFrame(xCon)
            .mul([0.05 * (idx + 1) - 0.02 * nRealConX for idx in range(nRealConX)])
            .values,
            axis=1,
        )
        + np.sum(
            pd.DataFrame(xCat)
            .mul([0.05 * (idx + 1) - 0.02 * nRealCatX for idx in range(nRealCatX)])
            .values,
            axis=1,
        )
    )
    mu_poi = np.exp(mu_poi)
    mu_true_poi = mu_poi / np.mean(mu_poi)

    # Gamma variables
    mu_gam = (
        6.0
        # + 0.5 * np.sum(xCon[:, : int(nRealConX / 2)], axis=1)
        # + 0.5 * np.sum(xCat[:, : int(nRealCatX / 2)], axis=1)
        # - 0.1 * np.sum(xCon[:, (int(nRealConX / 2) + 1) : nRealConX], axis=1)
        # - 0.1 * np.sum(xCat[:, (int(nRealCatX / 2) + 1) : nRealCatX], axis=1)
        + np.sum(
            pd.DataFrame(xCon)
            .mul([0.01 * (idx + 1) - 0.004 * nRealConX for idx in range(nRealConX)])
            .values,
            axis=1,
        )
        + np.sum(
            pd.DataFrame(xCat)
            .mul([0.01 * (idx + 1) - 0.004 * nRealCatX for idx in range(nRealCatX)])
            .values,
            axis=1,
        )
    )
    mu_gam = np.exp(mu_gam)
    mu_true_gamma = 10000 * mu_gam / np.mean(mu_gam)

    # generate Tweedie response
    power = 1.5
    phi = 2
    _lambda = mu_true_poi ** (2 - power) / (phi * (2 - power))
    alpha = (2 - power) / (1 - power)
    gam = phi * (power - 1) * mu_true_gamma ** (power - 1)

    y_poisson = np.random.poisson(_lambda)
    y = np.random.gamma(np.abs(-y_poisson * alpha), gam)
    y = np.array(
        [
            item * (1 + 0.25 * np.abs(np.random.normal())) if item > 0 else item
            for item in y
        ]
    )

    # return concatenated data
    data = np.concatenate((xCon, xCat, y.reshape(-1, 1)), axis=1)

    return pd.DataFrame(
        data,
        columns=[
            *["Con_{:.2f}".format(0.05 * (idx + 1) - 0.02 * 20) for idx in range(20)],
            *["Cat_{:.2f}".format(0.05 * (idx + 1) - 0.02 * 20) for idx in range(20)],
            "y",
        ],
    )
