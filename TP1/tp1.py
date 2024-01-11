import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import eig, inv
from scipy.stats import t
from sklearn.linear_model import LinearRegression

PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    # Q1:
    df = pd.read_csv(PATH + "/data/invest.csv", sep=" ")
    print("Q1:")
    print(df.head())

    # Q2:
    # plt.scatter(df["gnp"], df["invest"])
    # plt.xlabel("gnp")
    # plt.ylabel("invest")
    # plt.show()

    lgnp = np.log(df["gnp"])
    linvest = np.log(df["invest"])

    # plt.scatter(lgnp, linvest)
    # plt.xlabel("log(gnp)")
    # plt.ylabel("log(invest)")
    # plt.show()

    # Q3:
    print("\nQ3:")
    x_bar = lgnp.mean()
    y_bar = linvest.mean()

    # slope
    beta1 = np.sum((lgnp - x_bar) * (linvest - y_bar)) / np.sum((lgnp - x_bar) ** 2)
    # intercept
    beta0 = y_bar - beta1 * x_bar
    print("beta0:", beta0)
    print("beta1:", beta1)

    sigma2 = (1 / (len(lgnp) - 2)) * np.sum((linvest - beta0 - beta1 * lgnp) ** 2)
    print("sigma2:", sigma2)

    std_beta0 = np.sqrt(
        sigma2 * ((1 / len(lgnp)) + (x_bar**2) / np.sum((lgnp - x_bar) ** 2))
    )
    std_beta1 = np.sqrt(sigma2 / np.sum((lgnp - x_bar) ** 2))
    print("std_beta0:", std_beta0)
    print("std_beta1:", std_beta1)

    r_squared = 1 - np.sum((linvest - beta0 - beta1 * lgnp) ** 2) / np.sum(
        (linvest - y_bar) ** 2
    )
    print("r_squared:", r_squared)

    # Q4:
    print("\nQ4:")
    n = len(lgnp)
    p = 1
    alpha = 0.1

    c = t.ppf(1 - alpha / 2, n - p - 1)
    print("c:", c)

    t_stat = beta1 / std_beta1
    print("T-statistic:", t_stat)

    p_value = 2 * (1 - t.cdf(t_stat, n - p - 1))
    print("p_value:", p_value)

    if -c <= t_stat <= c:
        print("Fail to reject H0")
    else:
        print("H0 is rejected")

    # Q5:
    # GNP = 1000
    CI = [
        (
            beta0
            + beta1 * np.log(1000)
            - t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 / n + (np.log(1000) - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        ),
        (
            beta0
            + beta1 * np.log(1000)
            + t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 / n + (np.log(1000) - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        ),
    ]

    PI = [
        (
            beta0
            + beta1 * np.log(1000)
            - t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(
                1 + 1 / n + (np.log(1000) - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2)
            )
        ),
        (
            beta0
            + beta1 * np.log(1000)
            + t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(
                1 + 1 / n + (np.log(1000) - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2)
            )
        ),
    ]
    y_est = beta0 + beta1 * np.log(1000)

    print("\nQ5:")
    print("Predicted Investment for GNP = 1000:", y_est)
    print("90% CI:", CI)
    print("90% PI:", PI)

    # Q6:
    CI_lower = [
        (
            beta0
            + beta1 * lgnp[i]
            - t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 / n + (lgnp[i] - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        )
        for i in range(n)
    ]
    CI_upper = [
        (
            beta0
            + beta1 * lgnp[i]
            + t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 / n + (lgnp[i] - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        )
        for i in range(n)
    ]

    PI_lower = [
        (
            beta0
            + beta1 * lgnp[i]
            - t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 + 1 / n + (lgnp[i] - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        )
        for i in range(n)
    ]
    PI_upper = [
        (
            beta0
            + beta1 * lgnp[i]
            + t.ppf(1 - alpha / 2, n - 2)
            * np.sqrt(sigma2)
            * np.sqrt(1 + 1 / n + (lgnp[i] - x_bar) ** 2 / np.sum((lgnp - x_bar) ** 2))
        )
        for i in range(n)
    ]

    # plt.scatter(lgnp, linvest)
    # plt.xlabel("log(gnp)")
    # plt.ylabel("log(invest)")
    # plt.plot(lgnp, beta0 + beta1 * lgnp, color="red")
    # plt.fill_between(
    #     lgnp, CI_lower, CI_upper, color="orange", alpha=0.3, label="Confidence Interval"
    # )
    # plt.fill_between(
    #     lgnp, PI_lower, PI_upper, color="green", alpha=0.3, label="Prediction Interval"
    # )
    # plt.title("Linear regression")
    # plt.legend()
    # plt.show()

    # Q7:
    print("\nQ7:")
    x = np.log(1000)
    model = LinearRegression().fit(lgnp.values.reshape(-1, 1), linvest)
    y_pred = model.predict(x.reshape(1, -1))

    print("beta0:", model.intercept_)
    print("beta1:", model.coef_[0])
    print("r_squared:", model.score(lgnp.values.reshape(-1, 1), linvest))
    print("Predicted Investment for GNP = 1000:", y_pred)

    # Q8:
    # plt.scatter(lgnp, linvest)
    # plt.xlabel("log(gnp)")
    # plt.ylabel("log(invest)")
    # plt.plot(lgnp, model.intercept_ + model.coef_[0] * lgnp, color="red")
    # plt.plot(np.log(1000), y_pred, "o", color="green")
    # plt.title("Linear regression with sklearn")
    # plt.show()

    # Q9:
    print("\nQ9:")
    gnp = df["gnp"]
    interest = df["interest"]

    X = np.column_stack((gnp, interest))

    G = np.dot(X.T, X)
    eig_vals, eig_vecs = eig(np.dot(G.T, G))
    n_eig_zero = np.count_nonzero(eig_vals == 0)
    rank_G = G.shape[0] - n_eig_zero

    print("Gram Matrix:", G)
    print("Rank of Gram Matrix:", rank_G)

    # Check if the Gram matrix is of full rank
    if rank_G == min(X.shape):
        print("The Gram matrix is of full rank.")
    else:
        print("The Gram matrix is not of full rank.")

    # Q10:
    print("\nQ10:")
    y_bar = linvest.mean()
    X = np.column_stack((np.ones(n), lgnp, interest))

    beta = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, linvest))
    print("beta0:", beta[0])
    print("beta1:", beta[1])
    print("beta2:", beta[2])

    sigma2 = (1 / (n - 3)) * np.sum(
        (linvest - beta[0] - beta[1] * lgnp - beta[2] * interest) ** 2
    )
    C = sigma2 * inv(np.dot(X.T, X))
    std_beta0 = np.sqrt(C[0, 0])
    std_beta1 = np.sqrt(C[1, 1])
    std_beta2 = np.sqrt(C[2, 2])
    print("std_beta0:", std_beta0)
    print("std_beta1:", std_beta1)
    print("std_beta2:", std_beta2)

    r_squared = 1 - np.sum(
        (linvest - beta[0] - beta[1] * lgnp - beta[2] * interest) ** 2
    ) / np.sum((linvest - y_bar) ** 2)
    print("r_squared:", r_squared)

    n = len(lgnp)
    alpha = 0.1

    c = t.ppf(1 - alpha / 2, n - 3)
    print("c:", c)

    for i in range(len(beta)):
        print(f"\nBeta {i}:")
        t_stat = beta[i] / np.sqrt(C[i, i])
        print("T-statistic:", t_stat)

        p_value = 2 * (1 - t.cdf(t_stat, n - 3))
        print("p_value:", p_value)

        if -c <= t_stat <= c:
            print("Fail to reject H0")
        else:
            print("H0 is rejected")

    # Q11:
    print("\nQ11:")
    alpha = 0.001
    c = t.ppf(1 - alpha / 2, n - 3)

    x_pred = np.array([1, np.log(1000), 10])
    y_pred = np.dot(x_pred, beta)

    CI = [
        y_pred
        - c
        * np.sqrt(sigma2)
        * np.sqrt(np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T)),
        y_pred
        + c
        * np.sqrt(sigma2)
        * np.sqrt(np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T)),
    ]

    PI = [
        y_pred
        - c
        * np.sqrt(sigma2)
        * np.sqrt(1 + np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T) + 1 / n),
        y_pred
        + c
        * np.sqrt(sigma2)
        * np.sqrt(1 + np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T) + 1 / n),
    ]

    print("Predicted Investment for GNP = 1000 and Interest = 10:", y_pred)
    print("99.9% CI:", CI)
    print("99.9% PI:", PI)

    # Q12:
    y_pred = np.dot(X, beta)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot data points
    ax.scatter(lgnp, interest, linvest)
    ax.set_xlabel("log(gnp)")
    ax.set_ylabel("interest")
    ax.set_zlabel("log(invest)")

    # plot predicted values
    ax.scatter(lgnp, interest, y_pred, color="green", label="Predicted values")

    # plot regression plane
    x_surf = np.linspace(min(lgnp), max(lgnp), 50)
    y_surf = np.linspace(min(interest), max(interest), 50)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = beta[0] + beta[1] * x_surf + beta[2] * y_surf
    ax.plot_surface(
        x_surf, y_surf, z_surf, color="red", alpha=0.3, label="Regression Plane"
    )

    # plot confidence interval plane
    CI_lower = np.zeros((50, 50))
    CI_upper = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            x_pred = np.array([1, x_surf[i, j], y_surf[i, j]])
            CI_lower[i, j] = np.dot(x_pred, beta) - c * np.sqrt(sigma2) * np.sqrt(
                np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T)
            )
            CI_upper[i, j] = np.dot(x_pred, beta) + c * np.sqrt(sigma2) * np.sqrt(
                np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T)
            )

    ax.plot_surface(
        x_surf,
        y_surf,
        CI_lower,
        color="orange",
        alpha=0.3,
        label="Confidence Interval",
    )
    ax.plot_surface(
        x_surf,
        y_surf,
        CI_upper,
        color="orange",
        alpha=0.3,
    )

    # plot prediction interval plane
    PI_lower = np.zeros((50, 50))
    PI_upper = np.zeros((50, 50))

    for i in range(50):
        for j in range(50):
            x_pred = np.array([1, x_surf[i, j], y_surf[i, j]])
            PI_lower[i, j] = np.dot(x_pred, beta) - c * np.sqrt(sigma2) * np.sqrt(
                1 + np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T) + 1 / n
            )
            PI_upper[i, j] = np.dot(x_pred, beta) + c * np.sqrt(sigma2) * np.sqrt(
                1 + np.dot(np.dot(x_pred, inv(np.dot(X.T, X))), x_pred.T) + 1 / n
            )

    ax.plot_surface(
        x_surf,
        y_surf,
        PI_lower,
        color="green",
        alpha=0.3,
        label="Prediction Interval",
    )

    ax.plot_surface(
        x_surf,
        y_surf,
        PI_upper,
        color="green",
        alpha=0.3,
    )

    ax.legend()
    plt.show()

    # Q13:
    print("\nQ13:")
    model = LinearRegression().fit(X, linvest)
    y_pred = model.predict(np.array([1, np.log(1000), 10]).reshape(1, -1))

    print("beta0:", model.intercept_)
    print("beta1:", model.coef_[1])
    print("beta2:", model.coef_[2])
    print("r_squared:", model.score(X, linvest))
    print("Predicted Investment for GNP = 1000, Interest = 10:", y_pred)


if __name__ == "__main__":
    main()
