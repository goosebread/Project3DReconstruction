"""
Tools for working 3D plots in `matplotlib`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project_3d_reconstruction import homography

if TYPE_CHECKING:
    import numpy.typing as npt
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-not-found]


def axes_proj_image_coord(
    axes: Axes3D, coordinates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Convert the given world 3D coordinates to their corresponding 2D image coordinates
    in the given Axes' view box.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError(
            f"coordinates must be an Nx3 array, got shape {coordinates.shape}"
        )

    proj = axes.get_proj()
    _transformed = homography.projective_transform(coordinates, proj)

    raise NotImplementedError
