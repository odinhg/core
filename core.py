import gudhi
import numpy as np
from miniball import Miniball
from scipy.spatial import KDTree
from itertools import combinations


def core_cech(points, k, max_dim=1, max_r=None, return_squared=False):
    """
    Compute the core Čech filtration for a fixed k (if max_r is None) or along the line passing through (0, k) and (max_r, 0).

    TODO: Write docstring.
    """

    kd_tree = KDTree(points)
    k_core_distances, _ = kd_tree.query(points, k=k)

    if k == 1:
        k_core_distances = k_core_distances[..., np.newaxis]

    if max_r is None:
        vertex_values = k_core_distances[:, -1]
    else:
        line = np.linspace(max_r, 0, num=k)
        vertex_indices = np.argmax(line <= k_core_distances, axis=1)
        vertex_values = k_core_distances[np.arange(len(points)), vertex_indices]
    
    st = gudhi.SimplexTree()
    for dim in range(max_dim + 1):
        for face in combinations(range(len(points)), dim + 1):
            face_list = list(face)
            max_core = max(vertex_values[face_list])
            squared_radius = Miniball(points[face_list]).squared_radius()
            if not return_squared:
                st.insert(face, max(squared_radius**0.5, max_core))
            else:
                st.insert(face, max(squared_radius, max_core**2))
    return st


def alpha_core(points, k=1):
    kd_tree = KDTree(points)
    k_core_distances, _ = kd_tree.query(points, k=k)
    if k > 1:
        k_core_distances = k_core_distances[:, -1]
    st_alpha = gudhi.AlphaComplex(points=points).create_simplex_tree()
    for face, val in st_alpha.get_filtration():
        max_core = max(k_core_distances[list(face)])
        st_alpha.assign_filtration(face, max(val**0.5, max_core))
    return st_alpha


def alpha_core_slope(points, max_k=None, max_r=None):
    if max_k is None:
        max_k = len(points)
    if max_r is None:
        max_r = 2 * np.sqrt(Miniball(points).squared_radius())
    kd_tree = KDTree(points)
    k_core_distances, _ = kd_tree.query(points, k=max_k)
    line = np.linspace(max_r, 0, num=max_k)
    if max_k > 1:
        vertex_indices = np.argmax(line <= k_core_distances, axis=1)
        vertex_values = k_core_distances[
            np.arange(len(k_core_distances)), vertex_indices
        ]
    else:
        vertex_values = np.array(k_core_distances)
    st_alpha = gudhi.AlphaComplex(points=points).create_simplex_tree()
    for face, val in st_alpha.get_filtration():
        face_list = list(face)
        max_core = max(vertex_values[face_list])
        new_val = min(max(val**0.5, max_core), max_r)
        st_alpha.assign_filtration(face, new_val)
    return st_alpha
