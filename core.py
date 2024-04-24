import gudhi
import numpy as np
from miniball import Miniball
from scipy.spatial import KDTree
from itertools import combinations
from math import sqrt
from typing import Union


def core_cech(
    points: np.ndarray,
    k: int,
    max_dim: int = 1,
    max_r: Union[float, None] = None,
    return_squared: bool = False,
) -> gudhi.SimplexTree:
    """
    Compute the core Čech filtration for a fixed k if max_r is None. Otherwise, compute the filtration along the line passing through (0, k) and (max_r, 0).

    Args:
        points: Input point cloud of shape (n, d).
        k: Value of k as described above.
        max_dim: Add simplices of dimensions up to and including max_dim + 1.
        max_r: Value of max_r as described above.
        return_squared: Return squared filtration values if True, default value is False.
    Returns:
        The core Čech filtration as a simplex tree.
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
            if return_squared:
                filtration_value = max(squared_radius, max_core**2)
            else:
                filtration_value = max(sqrt(squared_radius), max_core)
            st.insert(face, filtration_value)
    return st


def alpha_core(
    points: np.ndarray,
    k: int,
    max_r: Union[float, None] = None,
    return_squared: bool = False,
) -> gudhi.SimplexTree:
    """
    Compute the alpha-core filtration for a fixed k if max_r is None. Otherwise, compute the filtration along the line passing through (0, k) and (max_r, 0).

    Args:
        points: Input point cloud of shape (n, d).
        k: Value of k as described above.
        max_r: Value of max_r as described above.
        return_squared: Return squared filtration values if True, default value is False.
    Returns:
        The alpha-core filtration as a simplex tree.
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

    st = gudhi.AlphaComplex(points=points).create_simplex_tree()
    for face, val in st.get_filtration():
        face_list = list(face)
        max_core = max(vertex_values[face_list])
        if return_squared:
            new_val = max(val, max_core**2)
        else:
            new_val = max(sqrt(val), max_core)
        st.assign_filtration(face, new_val)
    return st
