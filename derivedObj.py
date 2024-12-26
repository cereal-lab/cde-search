
import numpy as np
import warnings

# def matrix_factorization(tests: list[list[float]], k: int, steps: int, alpha: float, beta: float, epsilon = 0.001) -> tuple[np.ndarray, np.ndarray, float]:
#     m = len(tests)
#     n = len(tests[0])
#     P = np.random.rand(m, k)
#     Q = np.random.rand(n, k)
#     for step in range(steps):
#         indices = [(i, j) for i in range(m) for j in range(n)]
#         np.random.shuffle(indices)

#         for i, j in indices:
#             # P2 = np.copy(P)
#             # Q2 = np.copy(Q)
#             r = tests[i][j]
#             eij = r - np.dot(P[i, :], Q[j, :])
#             for s in range(k):
#                 P[i, s] = P[i, s] + alpha * (2 * eij * Q[j, s] - beta * P[i, s])
#                 Q[j, s] = Q[j, s] + alpha * (2 * eij * P[i, s] - beta * Q[j, s])
#             # P = P2
#             # Q = Q2
#         e = 0
#         for i in range(m):
#             for j in range(n):
#                 r = tests[i][j]
#                 e += (r - np.dot(P[i, :], Q[j, :])) ** 2
#                 for s in range(k):
#                     e = e / 2 + beta * (P[i, s] ** 2 + Q[j, s] ** 2)
#         if e < epsilon:
#             break
#     return P, Q, e

# def matrix_factorization(tests: list[list[float]], k: int, steps: int, alpha: float, beta: float, epsilon=0.001) -> tuple[np.ndarray, np.ndarray, float]:
#     m = len(tests)
#     n = len(tests[0])
    
#     # Convert tests to a PyTomatrix_factorizationyTorch tensors with requires_grad=True
#     P = torch.rand((m, k), dtype=torch.float32, requires_grad=True)
#     Q = torch.rand((n, k), dtype=torch.float32, requires_grad=True)
    
#     # Use the Adam optimizer
#     optimizer = optim.Adam([P, Q], lr=alpha)
    
#     for step in range(steps):
#         optimizer.zero_grad()
        
#         # Calculate the predicted matrix
#         pred = torch.matmul(P, Q.t())
        
#         # Calculate the loss
#         mask = tests > 0
#         loss = torch.sum((tests[mask] - pred[mask]) ** 2)
#         loss += beta * (torch.sum(P ** 2) + torch.sum(Q ** 2))
        
#         # Backpropagate the loss
#         loss.backward()
        
#         # Update the parameters
#         optimizer.step()
        
#         # Check for convergence
#         if loss.item() < epsilon:
#             break
    
#     return P.detach().numpy(), Q.detach().numpy(), loss.item()

from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

def matrix_factorization(tests: list[list[float]], k: int) -> tuple[np.ndarray, np.ndarray, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = ConvergenceWarning)
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(tests)
        H = model.components_
    return W, H, model.reconstruction_err_

from pyclustering.cluster.xmeans import xmeans
import warnings
import numpy as np
np.warnings = warnings
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# from pyclustering.utils import read_sample
# from pyclustering.samples.definitions import SIMPLE_SAMPLES

def xmean_cluster(tests: list[list[float]], kmax: int):
    # initial_centers = kmeans_plusplus_initializer(tests, 2).initialize()
    xmeans_instance = xmeans(tests, kmax = kmax)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return clusters, centers

if __name__ == '__main__':
    tests = [
        [2, 2, 2, 2],
        [1, 1, 2, 2],
        [2, 1, 1, 1]
    ]

    # P, Q, e = matrix_factorization(tests, 2, 10000, 0.05, 0.01)
    W, H, _ = matrix_factorization(tests, 2)
    clusters, centers = xmean_cluster(tests, 2)
    pass