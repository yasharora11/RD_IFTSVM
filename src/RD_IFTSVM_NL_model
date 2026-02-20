"""
A Novel Relative Density and Nonmembership-Based Intuitionistic Fuzzy Twin SVM
Non-linear (RBF kernel) version
"""

import numpy as np
from numpy import linalg as la
from cvxopt import matrix, solvers
from scipy.spatial.distance import pdist, squareform, cdist

#%%
def rbf_kernel(A, sigma):
"""
RBF kernel matrix for a single set: K(i,j)=exp(-sigma*||Ai-Aj||^2)
"""
    pairwise_sq_dists = squareform(pdist(A, 'sqeuclidean'))
    M = np.exp(-sigma * pairwise_sq_dists)
    return M

#%%
def rbf_kernel_between(X, C, sigma):
    """
    RBF kernel between two sets: K(x,c)=exp(-sigma*||x-c||^2)
    """
    pairwise_sq_dists = cdist(X, C, 'sqeuclidean')
    M = np.exp(-sigma * pairwise_sq_dists)  
    return M

def distance_to_same_class_center(kernel):
"""
    Kernel-space distance from each point to its class center:
    ||phi(x_i) - m|| where m is the mean in feature space.
  """
    m = kernel.shape[0]  # Number of training points
    dist = np.zeros(m)
    row_sum = np.sum(kernel, axis=0) 
    # Total sum of all elements in the kernel matrix
    total_sum = np.sum(kernel)
    # Compute the squared distance for each point
    for i in range(m):
        dist[i] = np.sqrt(kernel[i, i] - (2 / m) * row_sum[i] + (1 / m**2) * total_sum)
    return dist

#%%
def fuzzy_membership_relative_density_kernel(X_class, sigma):
    # Compute the RBF kernel matrix for X_class
    Kernel = rbf_kernel(X_class, sigma)
    
    # Determine the number of neighbors
    nbd = int(np.sqrt(X_class.shape[0]))
    
    # Calculate kernel-based distances for each pair of points
    # np.diag(Kernel)=K(x,x),[:, np.newaxis] this is performed to reshape the diagonal into column vector 
    kernel_distances = np.sqrt(np.diag(Kernel)[:, np.newaxis] + np.diag(Kernel) - 2 * Kernel) 
    
    # Sort distances for each point and find the k-th nearest neighbor distance
    kernel_distances.sort(axis=1)
    D_k = kernel_distances[:, nbd]  # k-th nearest distance for each instance
    epsilon = 1e-10  # Small constant to avoid division by zero
    D_k = np.where(D_k == 0, epsilon, D_k)
    # Calculate relative density and normalize by maximum relative density
    relative_density = 1 / D_k
    sum_relative_density=sum(relative_density)
    membership=relative_density/sum_relative_density
    
    return membership

#%%
def calculate_non_membership_ratio(X_class, X_other_class, delta,sigma):
   
    kernel_self =rbf_kernel(X_class,sigma) # Self-kernel matrix for positve class
    kernel_other =rbf_kernel(X_other_class,sigma) # kernel matrix for negative class
    
    kernel_between  = rbf_kernel_between(X_class, X_other_class, sigma) #Cross-kernel between X_class and X_other_class

    # Calculate the pairwise distance matrix between each point in x_pos and each point in x_neg i.e |phi(x_pos)-phi(x_neg)|
    kernel_distances_P1  = np.sqrt(np.diag(kernel_self)[:, np.newaxis] - 2 * kernel_between + np.diag(kernel_other))
    # Find pairs within delta distance (neighboring points) for P1.
    distances_within_delta_P1  = kernel_distances_P1 <= delta
    
    # Calculate distance to class center for each point in X_class 
    dis_class_center_self  = distance_to_same_class_center(kernel_self)    
    # calculate the difference betweeen class centers for each pair i.e |D(x_pos_i)-D(x_pos_j)|
    difference_between_each_self =np.abs(dis_class_center_self[:, np.newaxis] - dis_class_center_self)
    # Find pairs within delta distance for P2.
    distances_within_delta_P2 = difference_between_each_self <= delta

    rho_values = []
    for i in range(len(X_class)):
        # Count neighbors in X_other_class within delta for P1
        P1_cardinality = np.sum(distances_within_delta_P1[i, :])  # Cardinality of P1 for positive class
        P2_cardinality = np.sum(distances_within_delta_P2[i, :]) - 1  # Exclude the point itself (i != j)
        rho_i = P1_cardinality / (P1_cardinality + P2_cardinality) if (P1_cardinality + P2_cardinality) > 0 else 0
        rho_values.append(rho_i)
    
    return rho_values
    

def calculate_score(X_class,X_other_class,delta,sigma):
    membership=fuzzy_membership_relative_density_kernel(X_class,sigma)
    non_membership_ratio=calculate_non_membership_ratio(X_class,X_other_class,delta,sigma)
    nu=(1-membership)*non_membership_ratio
    score = np.where(nu == 0, membership, np.where(membership <= nu, 0, (1 - nu) / (2 - membership - nu)))
    return score

def calculate_proposed_membership(X_class,X_other_class,delta,sigma):
    score_positive = calculate_score(X_class, X_other_class, delta,sigma)
    score_negative = calculate_score(X_other_class,X_class, delta,sigma)
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
def fit1(A,B,u,C1,C2,sigma):
    m1=A.shape[0]
    m2=B.shape[0]
    e1=np.ones(m1) # vector of ones of class +1
    e2=np.ones(m2)
    C=np.r_[A,B]
    K1=rbf_kernel_between(A,C,sigma)
    K2=rbf_kernel_between(B,C,sigma)
    G=np.c_[K1,e1]
    Q=np.c_[K2,e2]

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

def fit2(A,B,u,C1,C2,sigma):
    m1=A.shape[0]
    m2=B.shape[0]
    e1=np.ones(m1) # vector of ones of class +1
    e2=np.ones(m2)
    C=np.r_[A,B]
    K1=rbf_kernel_between(A,C,sigma)
    K2=rbf_kernel_between(B,C,sigma)
    P=np.c_[K2,e2]
    H1=np.c_[K1,e1]

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
def predict(x,ker,w,b):
    norm_W = la.norm(w, axis=0)
    distances = np.abs(np.matmul(ker, w) + b)/norm_W
    labels = np.argmin(distances, axis=1)
    pred = np.where(labels == 1, -1, +1)
    return pred
