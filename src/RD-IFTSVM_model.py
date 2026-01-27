"""
A Novel Relative Density and Nonmembership-Based Intuitionistic Fuzzy Twin SVM for Class Imbalance Learning
"""
import numpy as np
from numpy import linalg as la
from sklearn.neighbors import NearestNeighbors
from cvxopt import matrix, solvers

def fuzzy_membership_relative_density(X_class):
     nbd=int(np.sqrt(X_class.shape[0]))
     """This will give the k nearest neigbor for every point in X_class"""
     nbrs = NearestNeighbors(n_neighbors=nbd + 1).fit(X_class)
     """ It will give the distance of x to its k neigbors"""
     distances, _ = nbrs.kneighbors(X_class)
     """Get the kth nearest distance for each instance"""
     D_k = distances[:, nbd]
     """ Handle zero distances by replacing them with a small positive value """
     epsilon = 1e-10  # Small constant to avoid division by zero
     D_k = np.where(D_k == 0, epsilon, D_k)
     " relative density for each pattern"
     relative_density = 1 / D_k
     " sum of patterns in same class"
     sum_relative_density=sum(relative_density)
     # max_relative_density=max(relative_density)
     membership=relative_density/sum_relative_density
     
     return membership

def calculate_center(data):
    """Calculate the center (mean) of a set of data."""
    return np.mean(data, axis=0)

def distance(data, center):
    """Calculate the Euclidean distance between a point and the center."""
    return la.norm(data - center)

def calculate_non_membership_ratio(X_class, X_other_class, delta):
    # Calculate the class center for X_class
    class_center = calculate_center(X_class)  # Mean of all points in the class
    
    # Compute pairwise distances between all points in X_class and X_other_class
    distances_class_other = np.linalg.norm(X_class[:, np.newaxis, :] - X_other_class, axis=2)
    
    # Compute distances of each point in X_class to the class center
    distances_to_center = np.linalg.norm(X_class - class_center, axis=1)
    
    # Compute pairwise distances within X_class
    distances_within_class = np.abs(distances_to_center[:, np.newaxis] - distances_to_center)
    
    # Determine neighbors within delta for P1
    distances_within_delta_P1 = distances_class_other <= delta
    
    # Determine neighbors within delta for P2
    distances_within_delta_P2 = distances_within_class <= delta
    np.fill_diagonal(distances_within_delta_P2, False)  # Exclude the point itself (i != j)
    
    # Count neighbors for P1 and P2
    P1_cardinalities = np.sum(distances_within_delta_P1, axis=1)
    P2_cardinalities = np.sum(distances_within_delta_P2, axis=1)
    
    # Compute rho values
    total_cardinalities = P1_cardinalities + P2_cardinalities
    rho_values = np.divide(
        P1_cardinalities, total_cardinalities, out=np.zeros_like(P1_cardinalities, dtype=float), where=total_cardinalities > 0
    )
    
    return rho_values

def calculate_score(X_class,X_other_class,delta):
    membership=fuzzy_membership_relative_density(X_class)
    non_membership_ratio=calculate_non_membership_ratio(X_class,X_other_class,delta)
    nu=(1-membership)*non_membership_ratio
    score = np.where(nu == 0, membership, np.where(membership <= nu, 0, (1 - nu) / (2 - membership - nu)))
    return score

def calculate_proposed_membership(X_class,X_other_class,delta):
    score_positive = calculate_score(X_class, X_other_class, delta)
    score_negative = calculate_score(X_other_class,X_class, delta)
    num_positive_samples = len(X_class)
    num_negative_samples = len(X_other_class)
    imbalance_ratio = min(num_positive_samples, num_negative_samples) / max(num_positive_samples, num_negative_samples)    
    # Determine the minority class
    minority_class = 1 if num_positive_samples < num_negative_samples else -1
    
    if minority_class==-1:
        prop_mem_pos=score_positive*imbalance_ratio
        prop_mem_neg=np.ones(num_negative_samples)
    else:
        prop_mem_neg=score_negative*imbalance_ratio
        prop_mem_pos=np.ones(num_positive_samples)
        
    return prop_mem_pos,prop_mem_neg

#%%
# A-class +1, B=class -1, u membership for class -1 i.e u2, M1,M1 mpdels parameter
def fit1(A,B,u,C1,C2):
    m1=A.shape[0]
    m2=B.shape[0]
    e1=np.ones(m1) # vector of ones of class +1
    e2=np.ones(m2)
    G=np.c_[A,e1]
    Q=np.c_[B,e2]

    P1=np.linalg.inv(np.matmul(G.T,G)+C1*np.identity(G.shape[1]))
    P2=np.linalg.multi_dot([Q,P1,Q.T])
    
    e=np.repeat([-1],m2)
    g=np.r_[np.identity(m2),np.diag(np.zeros(m2))-(np.identity(m2))]
    h=np.r_[C2*u,np.repeat([0],m2)]
    
    P = matrix(P2)
    q = matrix(e,tc='d')
    G = matrix(g)
    H = matrix(h,tc='d')
    
    sol = solvers.qp(P, q, G,H)
    dual = np.array(sol["x"])
    
    w=np.linalg.multi_dot([-P1,Q.T,dual])
    
    w1=w[:-1]
    b1=w[-1]
    
    return w1,b1

# pass u membership of class +1 i.e u1
def fit2(A,B,u,C1,C2):
    m1=A.shape[0]
    m2=B.shape[0]
    e1=np.ones(m1) # vector of ones of class +1
    e2=np.ones(m2)
    P=np.c_[B,e2]
    H1=np.c_[A,e1]

    P1=np.linalg.inv(np.matmul(P.T,P)+C1*np.identity(P.shape[1]))
    P2=np.linalg.multi_dot([H1,P1,H1.T])
    
    e=np.repeat([-1],m1)
    g=np.r_[np.identity(m1),np.diag(np.zeros(m1))-(np.identity(m1))]
    h=np.r_[C2*u,np.repeat([0],m1)]
    
    P = matrix(P2)
    q = matrix(e,tc='d')
    G = matrix(g)
    H = matrix(h,tc='d')
    
    sol = solvers.qp(P, q, G,H)
    dual = np.array(sol["x"])
    
    w=np.linalg.multi_dot([P1,H1.T,dual])
    
    w2=w[:-1]
    b2=w[-1]
    
    return w2,b2
#%%
def predict(x,w,b):
    norm_W = la.norm(w, axis=0)
    distances = np.abs(np.matmul(x, w) + b)/norm_W
    labels = np.argmin(distances, axis=1)
    pred = np.where(labels == 1, -1, +1)
    return pred
