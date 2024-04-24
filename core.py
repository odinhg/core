import gudhi
import numpy as np
from miniball import Miniball
from scipy.spatial import KDTree
from itertools import combinations

# Compute distances to k-nearest neighbors in X
def k_nearest_neighbor_distances(X, k):
    kd_tree = KDTree(X)
    k_core_distances, _ = kd_tree.query(X, k=k, workers=-1)
    return k_core_distances  

# Compute filtration values of faces from squared radius and core values
def core_value(face, squared_radius, core_values):
    max_core = max(core_values[face]) 
    return max(squared_radius ** 0.5, max_core)


# Compute core value for each points
def vertex_values(X, max_k, max_r):
    if max_r is None or max_k <= 1:
        return k_nearest_neighbor_distances(X, k=[max_k])
    k_core_distances = k_nearest_neighbor_distances(X, np.arange(1, max_k + 1))
    line = np.linspace(max_r, 0, num=max_k)
    if max_k>1:
        indices = np.argmax(line <= k_core_distances, axis=1)
        values = k_core_distances[np.arange(len(k_core_distances)), indices]
        values[values > max_r] = max_r
    else:
        values = np.array(k_core_distances)
    return values

# Compute core simplex tree from a simplex tree filtered by squared radius and a point cloud
def core_complex(X, st, max_k, max_r=None):
    k_core_distances = vertex_values(X, max_k, max_r)**2
    for vertex in range(X.shape[0]):
        st.assign_filtration([vertex], k_core_distances[vertex])
    st.make_filtration_non_decreasing()
    return st

# Compute simplex tree of a point cloud filtered by squared radius
def cech_squared_radius(X, max_dim=1):
    st = gudhi.SimplexTree()
    for dim in range(max_dim+1):
        for face in combinations(range(len(X)),dim+1):
            val = Miniball(X[list(face)]).squared_radius()
            st.insert(face, val)
    return st


def core_cech(X, max_k=10, max_r=None, max_dim=1):
    """
    Compute the core Čech filtration for a fixed k if max_r is None. Otherwise, compute the filtration along the line passing through (0, k) and (max_r, 0).

    Inputs:
        X: Input point cloud of shape (n, d).
        k: Value of k as described above.
        max_dim: Add simplices of dimensions up to and including max_dim + 1.
        max_r: Value of max_r as described above.
    Returns:
        The core Čech filtration as a simplex tree.
    """
    st = cech_squared_radius(X, max_dim=max_dim)
    return core_complex(X, st, max_k, max_r)

def alpha_core(X, max_k=10, max_r=None, precision='safe'):
    """
    Compute the alpha-core filtration for a fixed k if max_r is None. Otherwise, compute the filtration along the line passing through (0, k) and (max_r, 0).

    Args:
        X: Input point cloud of shape (n, d).
        k: Value of k as described above.
        max_r: Value of max_r as described above.
        precision: Precision for Gudhis Alpha complex, can be ‘fast’, ‘safe’ or ‘exact’.
    Returns:
        The alpha-core filtration as a simplex tree.
    """
    st = gudhi.AlphaComplex(points=X, precision=precision).create_simplex_tree()
    return core_complex(X, st, max_k, max_r)

def plot_sqrt_persistence(st, **kwargs):
    persistence = st.persistence()
    persistence = [(dim, (birth**.5, death**.5)) for dim, (birth, death) in persistence]
    return gudhi.plot_persistence_diagram(persistence, **kwargs)
