import numpy as np


def sample_circle(
    n: int,
    generator: np.random.Generator,
    r: float = 1.0,
    center: tuple[float] = (0.0, 0.0),
    variance: float = 0.0,
) -> np.ndarray:
    #t = generator.uniform(0, 2 * np.pi, n)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = r * np.c_[np.cos(t), np.sin(t)]
    if variance > 0.0:
        X += generator.normal(loc=0.0, scale=np.sqrt(variance), size=(n, 2))
    return X


def sample_rectangle(
    n: int,
    generator: np.random.Generator,
    lower_left_corner: tuple[float] = (-1, -1),
    upper_right_corner: tuple[float] = (1, 1),
) -> np.ndarray:
    x1, y1 = lower_left_corner
    x2, y2 = upper_right_corner
    X = np.c_[generator.uniform(x1, x2, n), generator.uniform(y1, y2, n)]
    return X


if __name__ == "__main__":
    """
    Example usage:
    """
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(seed=0)
    X = sample_circle(50, generator, variance=0.01)
    Y = sample_rectangle(50, generator)
    Z = np.r_[X, Y]
    plt.scatter(*Z.T)
    plt.show()
