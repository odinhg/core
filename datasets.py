import numpy as np
from typing import Tuple


def sample_circle(
    n: int,
    generator: np.random.Generator,
    r: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    variance: float = 0.0,
) -> np.ndarray:
    t = generator.uniform(0, 2 * np.pi, n)
    X = r * np.c_[np.cos(t), np.sin(t)]
    if variance > 0.0:
        X += generator.normal(loc=0.0, scale=np.sqrt(variance), size=(n, 2))
    return X


def sample_rectangle(
    n: int,
    generator: np.random.Generator,
    lower_left_corner: Tuple[float, float] = (-1, -1),
    upper_right_corner: Tuple[float, float] = (1, 1),
) -> np.ndarray:
    x1, y1 = lower_left_corner
    x2, y2 = upper_right_corner
    X = np.c_[generator.uniform(x1, x2, n), generator.uniform(y1, y2, n)]
    return X


def sample_torus(
    n: int,
    generator: np.random.Generator,
    a: float = 0.25,
    b: float = 0.75,
    variance: float = 0.0,
) -> np.ndarray:
    ...
    t1 = generator.uniform(0, 2 * np.pi, n)
    t2 = generator.uniform(0, 2 * np.pi, n)
    r = a * np.cos(t1) + b
    X = np.c_[r * np.cos(t2), r * np.sin(t2), a * np.sin(t1)]
    if variance > 0.0:
        X += generator.normal(loc=0.0, scale=np.sqrt(variance), size=(n, 3))
    return X


def sample_cube(
    n: int,
    generator: np.random.Generator,
    lower_left_corner: Tuple[float, float, float] = (-1, -1, -1),
    upper_right_corner: Tuple[float, float, float] = (1, 1, 1),
) -> np.ndarray:
    x1, y1, z1 = lower_left_corner
    x2, y2, z2 = upper_right_corner
    X = np.c_[
        generator.uniform(x1, x2, n),
        generator.uniform(y1, y2, n),
        generator.uniform(z1, z2, n),
    ]
    return X


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(seed=0)

    # Torus with noise
    X = sample_torus(5000, generator, variance=0.01)
    Y = sample_cube(1000, generator)
    Z = np.r_[X, Y]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*Z.T, alpha=0.2, color="black", s=5)
    plt.show()

    # Circle with noise
    X = sample_circle(200, generator, variance=0.01)
    Y = sample_rectangle(100, generator)
    Z = np.r_[X, Y]
    plt.scatter(*Z.T, color="black", s=5)
    plt.show()
