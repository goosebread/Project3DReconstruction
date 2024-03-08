"""
Utilities for working with homographies / projective transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def projective_transform(
    x: npt.ArrayLike, transform: npt.ArrayLike
) -> npt.NDArray[Any]:
    """
    Apply a project transform to heterogeneous/non-homogenous coordinates.

    Parameters
    ----------
    x : array-like
        Heterogeneous/non-homogenous coordinates
    transform : array-like
        Projective transform.

    Returns
    -------
    numpy.ndarray
        Transformed coordinates.
    """

    proj = transform @ homogenize(x)[..., np.newaxis]
    proj = np.squeeze(proj, axis=-1)

    return dehomogenize(proj)


def homogenize(x: npt.ArrayLike) -> npt.NDArray[Any]:
    """
    Convert heterogeneous/non-homogenous coordinates to homogenous coordinates.

    >>> homogenize([2, 3, 4])
    array([2., 3., 4., 1.])
    >>> homogenize([(2, 3), (3, 3), (4, 4)])
    array([[2., 3., 1.],
           [3., 3., 1.],
           [4., 4., 1.]])
    """
    x = np.asarray(x)
    return np.append(x, np.ones((*x.shape[:-1], 1)), axis=-1)


def dehomogenize(x: npt.ArrayLike) -> npt.NDArray[Any]:
    """
    Convert homogenous coordinates to heterogeneous/non-homogenous coordinates.

    >>> dehomogenize([10, 20, 30, 10])
    array([1., 2., 3.])
    >>> dehomogenize([[2.0, 3.0, 1.0], [6.0, 6.0, 2.0], [12.0, 12.0, 3.0]])
    array([[2., 3.],
           [3., 3.],
           [4., 4.]])
    """
    x = np.asarray(x)
    scaled = x / x[..., [-1]]
    return scaled[..., :-1]  # type: ignore
