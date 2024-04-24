import numpy as np
from typing import Tuple


def sample_circle(
    n: int,
    generator: np.random.Generator,
    r: float = 1.0,
    std: float = 0.0,
) -> np.ndarray:
    t = generator.uniform(0, 2 * np.pi, n)
    X = r * np.c_[np.cos(t), np.sin(t)]
    if std > 0.0:
        X += generator.normal(loc=0.0, scale=std, size=(n, 2))
    return X

def sample_flat_torus(
        n: int,
        generator: np.random.Generator,
        std: float = 0.0,
        ) -> np.ndarray:
    return np.c_[sample_circle(n, generator, r=1, std=std),
                 sample_circle(n, generator, r=1, std=std)]

def sample_torus(
    n: int,
    generator: np.random.Generator,
    a: float = 0.25,
    b: float = 0.75,
    std: float = 0.0,
) -> np.ndarray:
    ...
    # C1 = sample_circle(n, generator, r=a, std=0)
    # C2 = sample_circle(n, generator, r=1, std=0)
    # r = a * C1[:, 0] + b
    # X = np.c_[r * C2[:, 0], r * C2[:, 1], a * C1[:, 1]]
    t1 = generator.uniform(0, 2 * np.pi, n)
    t2 = generator.uniform(0, 2 * np.pi, n)
    r = a * np.cos(t1) + b
    X = np.c_[r * np.cos(t2), r * np.sin(t2), a * np.sin(t1)]
    if std > 0.0:
        X += generator.normal(loc=0.0, scale=std, size=(n, 3))
    return X


def sample_cube(
    n: int,
    generator: np.random.Generator,
    lower_left_corner: Tuple[float, ...] = (-1, -1, -1),
    upper_right_corner: Tuple[float, ...] = (1, 1, 1),
) -> np.ndarray:
    X = generator.uniform(0, 1, (n, len(lower_left_corner)))
    X = X * (np.array(upper_right_corner) - np.array(lower_left_corner))
    X += np.array(lower_left_corner)
    return X

def sample_rectangle(
    n: int,
    generator: np.random.Generator,
    lower_left_corner: Tuple[float, float] = (-1, -1),
    upper_right_corner: Tuple[float, float] = (1, 1),
) -> np.ndarray:
    return sample_cube(n, generator, lower_left_corner, upper_right_corner)


def sample_sphere(
    n: int,
    generator: np.random.Generator,
    std: float = 0.1,
) -> np.ndarray:
    X = generator.uniform(-1, 1, (n, 3))
    X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
    X = X + generator.normal(loc=0.0, scale=std, size=(n, 3))
    return X
