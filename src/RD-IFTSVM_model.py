"""
A Novel Relative Density and Nonmembership-Based Intuitionistic Fuzzy Twin SVM for Class Imbalance Learning
"""

import numpy as np
from numpy import linalg as la
from sklearn.neighbors import NearestNeighbors
from cvxopt import matrix, solvers

# ---------------------------------------------------
# Fuzzy membership based on relative density
# ---------------------------------------------------
def fuzzy_membership_relative_density(X):
    """
    Compute fuzzy membership values using relative density.
    """
    k = int(np.sqrt(X.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    Dk = distances[:, k]
    Dk[Dk == 0] = 1e-10  # avoid division by zero

    density = 1 / Dk
    membership = density / np.sum(density)
    return membership


# ---------------------------------------------------
# Non-membership ratio
# ---------------------------------------------------
def calculate_non_membership_ratio(Xc, Xo, delta):
    """
    Xc : current class samples
    Xo : opposite class samples
    delta : neighborhood threshold
    """
    center = np.mean(Xc, axis=0)

    d_class_other = np.linalg.norm(
        Xc[:, None, :] - Xo[None, :, :], axis=2
    )
    d_center = np.linalg.norm(Xc - center, axis=1)
    d_within = np.abs(d_center[:, None] - d_center)

    P1 = np.sum(d_class_other <= delta, axis=1)
    P2 = np.sum(d_within <= delta, axis=1)

    total = P1 + P2
    rho = np.divide(P1, total, out=np.zeros_like(P1, dtype=float), where=total > 0)
    return rho


# ---------------------------------------------------
# Final membership score
# ---------------------------------------------------
def calculate_score(Xc, Xo, delta):
    mu = fuzzy_membership_relative_density(Xc)
    rho = calculate_non_membership_ratio(Xc, Xo, delta)

    nu = (1 - mu) * rho
    score = np.where(
        nu == 0,
        mu,
        np.where(mu <= nu, 0, (1 - nu) / (2 - mu - nu))
    )
    return score


# ---------------------------------------------------
# Proposed memberships for both classes
# ---------------------------------------------------
def calculate_proposed_membership(A, B, delta):
    sA = calculate_score(A, B, delta)
    sB = calculate_score(B, A, delta)

    r = min(len(A), len(B)) / max(len(A), len(B))

    if len(A) < len(B):
        return sA * r, np.ones(len(B))
    else:
        return np.ones(len(A)), sB * r


# ---------------------------------------------------
# Twin SVM optimization (QP)
# ---------------------------------------------------
def fit_positive(A, B, u, C1, C2):
    m = B.shape[0]
    e = np.ones(A.shape[0])

    G = np.c_[A, e]
    Q = np.c_[B, np.ones(m)]

    P1 = np.linalg.inv(G.T @ G + C1 * np.eye(G.shape[1]))
    P2 = Q @ P1 @ Q.T

    q = matrix(-np.ones(m))
    Gm = matrix(np.r_[np.eye(m), -np.eye(m)])
    hm = matrix(np.r_[C2 * u, np.zeros(m)])

    sol = solvers.qp(matrix(P2), q, Gm, hm)
    alpha = np.array(sol['x'])

    w = -P1 @ Q.T @ alpha
    return w[:-1], w[-1]


def fit_negative(A, B, u, C1, C2):
    m = A.shape[0]
    e = np.ones(B.shape[0])

    P = np.c_[B, e]
    H = np.c_[A, np.ones(m)]

    P1 = np.linalg.inv(P.T @ P + C1 * np.eye(P.shape[1]))
    P2 = H @ P1 @ H.T

    q = matrix(-np.ones(m))
    Gm = matrix(np.r_[np.eye(m), -np.eye(m)])
    hm = matrix(np.r_[C2 * u, np.zeros(m)])

    sol = solvers.qp(matrix(P2), q, Gm, hm)
    alpha = np.array(sol['x'])

    w = P1 @ H.T @ alpha
    return w[:-1], w[-1]


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
def predict(X, W, B):
    dist = np.abs(X @ W + B) / la.norm(W, axis=0)
    idx = np.argmin(dist, axis=1)
    return np.where(idx == 1, -1, 1)
