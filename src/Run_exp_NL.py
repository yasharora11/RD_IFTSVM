"""
Run RD-IFTSVM Non-Linear (RBF Kernel) on WPBC dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import RD_IFTSVM_NL_model as p5   


# ---------------------------------------------------
# Load WPBC dataset
# ---------------------------------------------------
def load_wpbc(path):
    df = pd.read_excel(path, header=None)
    df = df[0].str.split(',', expand=True)
    df.iloc[:, -1] = df.iloc[:, -1].str.strip()
    df.iloc[:, -1] = df.iloc[:, -1].replace({'positive': 1, 'negative': -1}).astype(float)
    return df.astype(float).to_numpy()


def compute_auc_from_cm(y_true, y_pred):
    """
    Compute AUC using sensitivity and false positive rate (paper-aligned).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = 1 - specificity

    auc = (1 + sensitivity - fpr) / 2
    return auc


# ---------------------------------------------------
# Hyperparameters (Choose the optimal hyperparameters)
# ---------------------------------------------------
C1 = 0.01
C2 = 0.01
delta = 1.5
sigma = 0.5   


# ---------------------------------------------------
# Main experiment
# ---------------------------------------------------
X = load_wpbc("data/WPBC.xlsx")   

kf = KFold(n_splits=10, shuffle=True, random_state=10)

auc_scores = []

for train_idx, test_idx in kf.split(X):
    train, test = X[train_idx], X[test_idx]

    A_tr = train[train[:, -1] == 1][:, :-1]
    B_tr = train[train[:, -1] == -1][:, :-1]

    A_te = test[test[:, -1] == 1][:, :-1]
    B_te = test[test[:, -1] == -1][:, :-1]

    if len(A_te) == 0 or len(B_te) == 0:
        continue

    scaler_A = MinMaxScaler()
    scaler_B = MinMaxScaler()

    A_tr = scaler_A.fit_transform(A_tr)
    A_te = scaler_A.transform(A_te)

    B_tr = scaler_B.fit_transform(B_tr)
    B_te = scaler_B.transform(B_te)

    uA, uB = p5.calculate_proposed_membership(A_tr, B_tr, delta, sigma)

    w1, b1 = p5.fit1(A_tr, B_tr, uB, C1, C2, sigma)
    w2, b2 = p5.fit2(A_tr, B_tr, uA, C1, C2, sigma)

    W = np.c_[w1, w2]
    B = np.c_[b1, b2]

    C = np.r_[A_tr, B_tr]
    AT = np.r_[A_te, B_te]
    ker = p5.rbf_kernel_between(AT, C, sigma)

    y_pred = p5.predict(AT, ker, W, B)
    y_true = np.r_[np.ones(len(A_te)), -np.ones(len(B_te))]

    auc = compute_auc_from_cm(y_true, y_pred)
    auc_scores.append(auc)

print("===================================")
print("RESULTS (10-fold CV)")
print("AUC :", np.mean(auc_scores) * 100)
print("===================================")
