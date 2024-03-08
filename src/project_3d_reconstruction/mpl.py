"""
Tools for working 3D plots in `matplotlib`.
"""
# Useful references:
# - https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from project_3d_reconstruction import homography

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


def fig_export_rgba(fig: Figure) -> npt.NDArray[np.uint8]:
    """Export matplotlib figure as RGBA array."""
    # image_data = np.array(fig.canvas.buffer_rgba()).copy()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="rgba", dpi="figure")
    buffer.seek(0)

    image_data = np.frombuffer(buffer.getbuffer(), dtype=np.uint8)
    return image_data.reshape((*fig.canvas.get_width_height()[::-1], 4))


def axes_coords_data2display(
    axes: Axes, coordinates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Convert the given 2D data coordinates to their corresponding display coordinates.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:  # noqa: PLR2004
        raise ValueError(
            f"coordinates must be an Nx3 array, got shape {coordinates.shape}"
        )
    return axes.transData.transform(coordinates)  # type: ignore


def axes_coords_3d2display(
    axes: Axes3D, coordinates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Convert the given world 3D coordinates to their corresponding 2D display
    coordinates.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError(
            f"coordinates must be an Nx3 array, got shape {coordinates.shape}"
        )

    # TODO: determine how to correct the transformation for non-square canvases.
    fig_dims = axes.figure.canvas.get_width_height()
    if fig_dims[0] != fig_dims[1]:
        raise ValueError(
            f"figures with unequal height and width are not yet supported"
            f" - got size {fig_dims}"
        )

    proj = axes.get_proj()
    ax_data_coords = homography.projective_transform(coordinates, proj)
    return axes_coords_data2display(axes, ax_data_coords[..., :-1])
